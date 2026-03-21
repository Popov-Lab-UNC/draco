"""dynamics.py – Explicit-solvent OpenMM MD engine for Draco.

Role in the Pipeline
--------------------
This is the **first stage** of the Draco pipeline.  It takes a prepared
protein PDB and runs a full explicit-solvent MD simulation on a CUDA GPU.
During production MD it continuously monitors the protein backbone (Cα)
RMSD and extracts a new **conformational frame** whenever the structure has
drifted by more than *rmsd_threshold* Å from the last extracted frame.

Extracted frames (protein-only, stripped of water/ions) are returned as
:class:`DynamicsFrame` objects, which downstream stages consume:

    dynamics.py  →  pocketeer  →  pocket_coloring  →  overlay  →  local_minimization

MD Protocol (vacuum-free)
-------------------------
1. Add rectangular TIP3P-FB water box (padding *box_padding_nm*).
2. Add neutralising Na⁺/Cl⁻ ions.
3. Energy-minimise the full solvated system.
4. NVT equilibration (protein heavy-atom restraints, *nvt_steps* steps).
5. NPT equilibration (MonteCarloBarostat, *npt_steps* steps).
6. NVT production (no restraints, *production_steps* steps).
   – Every *report_interval_steps* an RMSD check is performed;
     frames passing the threshold are saved.

Public API
----------
    result = run_dynamics(
        "6pbc-prepared.pdb",
        simulation_time_ps=5000,
        rmsd_threshold_angstrom=1.5,
        platform_name="CUDA",
    )
    for frame in result.frames:
        # frame.protein_pdb_string  – PDB string (protein only, no water/ions)
        # frame.atomarray            – biotite AtomArray for pocketeer
        # frame.simulation_time_ps  – time-stamp
        ...

"""
from __future__ import annotations

import copy
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pocketeer as pt

from protein_preparation import PreparedProtein, prepare_protein

try:
    from openmm import (
        LangevinMiddleIntegrator,
        MonteCarloBarostat,
        Platform,
        CustomExternalForce,
        unit,
    )
    from openmm.app import (
        DCDReporter,
        ForceField,
        HBonds,
        Modeller,
        PDBFile,
        PME,
        Simulation,
        StateDataReporter,
        Topology,
    )
except ImportError:  # pragma: no cover
    from simtk.openmm import (  # type: ignore
        LangevinMiddleIntegrator,
        MonteCarloBarostat,
        Platform,
        CustomExternalForce,
        unit,
    )
    from simtk.openmm.app import (  # type: ignore
        DCDReporter,
        ForceField,
        HBonds,
        Modeller,
        PDBFile,
        PME,
        Simulation,
        StateDataReporter,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2: float = 418.4
_DEFAULT_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")
_DEFAULT_WATER_MODEL = "tip3pfb"  # chemistry model; placement via Modeller (see below)

# Ion / water residue names that should be excluded when identifying protein atoms.
_SOLVENT_ION_RESNAMES: set[str] = {
    "HOH", "WAT", "TIP3", "SPC", "T3P", "T4P", "T5P",
    "NA", "CL", "Na+", "Cl-", "NA+", "CL-",
    "K", "K+", "MG", "CA", "ZN", "FE",
}

# OpenMM Modeller.addSolvent only accepts tip3p, spce, tip4pew, tip5p, swm4ndp.
# TIP3P-FB uses the same 3-site geometry as TIP3P; parameters come from the FF XML.
_MODELLER_SOLVENT_MODEL_ALIASES: dict[str, str] = {"tip3pfb": "tip3p"}


def _modeller_solvent_model(water_model: str) -> str:
    """Return the ``model=`` string required by ``Modeller.addSolvent``."""
    return _MODELLER_SOLVENT_MODEL_ALIASES.get(water_model, water_model)


# ─────────────────────────────────────────────────────────────────────────────
# Public data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DynamicsFrame:
    """A single conformational snapshot extracted from the MD trajectory.

    Only protein atoms are stored (water/ions stripped).  This object is
    everything the downstream pocketeer → overlay → minimization pipeline needs.
    """

    frame_index: int
    """Sequential index of this extracted frame (0-based)."""

    simulation_time_ps: float
    """Simulation time in picoseconds when this frame was captured."""

    rmsd_from_prev_angstrom: float
    """Backbone Cα RMSD from the previously extracted frame (Å).
    The very first frame always has RMSD = 0.0."""

    protein_pdb_string: str
    """PDB text of the protein-only structure (no water, no ions).
    Pass this to ``local_minimization.minimize_overlay_pose`` via
    ``protein_preparation.prepare_protein``."""

    protein_positions_nm: npt.NDArray[np.float64]
    """Protein atom positions in nanometres, shape (N_protein, 3)."""

    atomarray: object
    """biotite AtomArray of the protein-only snapshot.
    Pass directly to ``pocketeer.find_pockets`` and ``pocket_coloring.color_pockets``."""


@dataclass
class DynamicsResult:
    """Aggregated output of a completed MD run."""

    frames: list[DynamicsFrame]
    """Conformationally distinct frames extracted during production MD."""

    trajectory_dcd: Path | None
    """Path to the DCD trajectory file (full solvated system), or None if disabled."""

    topology_pdb: Path | None
    """Path to a solvated-system topology PDB used as DCD reference, or None if disabled."""

    prepared_protein: PreparedProtein
    """The PreparedProtein used for MD (re-use for per-frame minimization)."""

    n_protein_atoms: int
    """Number of protein (non-water, non-ion) atoms in the solvated topology."""

    simulation_time_ps: float
    """Total production MD time in picoseconds."""


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def run_dynamics(
    protein_pdb_path: str | Path,
    *,
    ph: float = 7.4,
    # Solvation
    box_padding_nm: float = 1.0,
    ionic_strength_molar: float = 0.15,
    # Simulation lengths
    minimize_max_iterations: int = 10_000,
    nvt_steps: int = 50_000,        # 100 ps @ 2 fs
    npt_steps: int = 50_000,        # 100 ps @ 2 fs
    production_steps: int = 2_500_000,  # 5 ns @ 2 fs
    # Integrator
    temperature_kelvin: float = 300.0,
    pressure_bar: float = 1.0,
    friction_per_ps: float = 1.0,
    timestep_fs: float = 2.0,
    nvt_timestep_fs: float = 1.0,   # smaller timestep for NVT equil (safer)
    # Conformational-change detection
    report_interval_steps: int = 5_000,   # check RMSD every 10 ps
    rmsd_threshold_angstrom: float = 1.5,
    # NVT equilibration restraints
    equil_restraint_k_kcal: float = 5.0,  # softer spring → less first-step force spike
    # Platform
    platform_name: str = "CUDA",
    cuda_precision: str = "mixed",
    # I/O
    output_dir: str | Path | None = None,
    save_trajectory: bool = True,
    forcefield_files: tuple[str, ...] = _DEFAULT_FORCEFIELD,
    water_model: str = _DEFAULT_WATER_MODEL,
    verbose: bool = True,
) -> DynamicsResult:
    """Run explicit-solvent NVT/NPT MD and return conformational frames.

    Parameters
    ----------
    protein_pdb_path:
        Path to a protein-only PDB file (no ligand, no water).
        PDBFixer will repair missing atoms and add hydrogens.
    ph:
        pH for protonation state assignment (default 7.4).
    box_padding_nm:
        Distance (nm) between protein surface and box edge (default 1.0 nm).
    ionic_strength_molar:
        NaCl ionic strength in mol/L (default 0.15 M physiological).
    minimize_max_iterations:
        Max L-BFGS steps for initial energy minimization (default 5000).
    nvt_steps:
        Steps of NVT equilibration with protein heavy-atom restraints (default 50k = 100 ps).
    npt_steps:
        Steps of NPT equilibration (MonteCarloBarostat, default 50k = 100 ps).
    production_steps:
        Steps of unrestrained NVT production MD (default 2.5M = 5 ns).
    temperature_kelvin:
        Simulation temperature (default 300 K).
    pressure_bar:
        Target pressure for NPT barostat (default 1.0 bar).
    friction_per_ps:
        Langevin friction coefficient 1/ps (default 1.0).
    timestep_fs:
        Integration timestep in femtoseconds (default 2 fs).
    report_interval_steps:
        How often (in steps) to check for conformational change (default 5000 = 10 ps).
    rmsd_threshold_angstrom:
        Backbone Cα RMSD threshold in Å; a new frame is extracted whenever
        drift from the last extracted frame exceeds this value (default 1.5 Å).
    equil_restraint_k_kcal:
        Harmonic restraint spring constant (kcal/mol/Å²) on protein heavy atoms
        during NVT equilibration (default 10.0).
    platform_name:
        OpenMM platform ('CUDA', 'OpenCL', 'CPU').
    cuda_precision:
        CUDA precision mode: 'mixed' (default), 'single', or 'double'.
    output_dir:
        Directory for trajectory/topology files.  Defaults to a subdirectory
        named 'dynamics_output' in the current working directory.
    save_trajectory:
        If True, write the full solvated trajectory as DCD (default True).
    forcefield_files:
        Force-field XML files (default AMBER14-all + TIP3P-FB water).
    water_model:
        Water chemistry label (default ``'tip3pfb'``, paired with
        ``amber14/tip3pfb.xml``).  Passed to ``Modeller.addSolvent`` after
        mapping known aliases (e.g. *tip3pfb* → *tip3p* for placement), because
        OpenMM only ships pre-equilibrated boxes for a fixed set of models.
    verbose:
        Print progress messages (default True).

    Returns
    -------
    DynamicsResult
        Contains the list of extracted :class:`DynamicsFrame` objects and
        metadata for downstream pipeline stages.
    """
    outdir = Path(output_dir) if output_dir else Path("dynamics_output")
    outdir.mkdir(parents=True, exist_ok=True)

    log = _Logger(verbose)

    timestep_fs = float(timestep_fs)
    steps_per_ps = 1000.0 / timestep_fs

    # ── Step 1: Prepare protein (PDBFixer + H addition) ────────────────────
    log("── Step 1/6: Preparing protein (PDBFixer) …")
    prepared_protein = prepare_protein(protein_pdb_path, ph=ph)
    base_atomarray = pt.load_structure(str(protein_pdb_path))

    # ── Snapshot: prepared protein (before solvation) ──────────────────────
    step1_pdb = outdir / "step1_prepared_protein.pdb"
    _save_pdb(prepared_protein.topology, prepared_protein.positions, step1_pdb)
    log(f"   Snapshot saved: {step1_pdb}")

    # ── Step 2: Build solvated system ──────────────────────────────────────
    log(f"── Step 2/6: Adding {water_model} water box (padding {box_padding_nm} nm, "
        f"[NaCl]={ionic_strength_molar} M) …")
    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(prepared_protein.topology, prepared_protein.positions)

    # Clear any inherited periodic box vectors (e.g. from an incorrect CRYST1 record
    # in the input PDB). This forces Modeller.addSolvent to automatically compute
    # a bounding box that fully encloses the protein plus the requested padding,
    # avoiding massive steric clashes when a too-small box is specified.
    modeller.topology.setPeriodicBoxVectors(None)

    modeller.addSolvent(
        forcefield,
        model=_modeller_solvent_model(water_model),
        padding=box_padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_molar * unit.molar,
    )
    log(f"   Solvated system: {modeller.topology.getNumAtoms()} atoms")

    # Identify protein atom indices in the solvated system (before any simulation)
    all_atoms = list(modeller.topology.atoms())
    protein_atom_indices = _identify_protein_atoms(all_atoms)
    n_protein_atoms = len(protein_atom_indices)
    log(f"   Protein atoms identified: {n_protein_atoms}")

    # Save solvated topology PDB (needed as DCD reference)
    topo_pdb_path = outdir / "step2_solvated_full.pdb"
    _save_pdb(modeller.topology, modeller.positions, topo_pdb_path)

    # ── Snapshot: solvated system + protein-only (before minimization) ─────
    step2_prot_pdb = outdir / "step2_solvated_protein_only.pdb"
    _save_protein_only_pdb(
        all_atoms, protein_atom_indices,
        np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64),
        step2_prot_pdb,
    )
    log(f"   Snapshot saved: {topo_pdb_path}  (full solvated)")
    log(f"   Snapshot saved: {step2_prot_pdb}  (protein only)")

    # ── Step 3: Create OpenMM system ───────────────────────────────────────
    log("── Step 3/6: Building OpenMM system (PME, HBonds constraints) …")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
    )

    # Identify Cα indices for RMSD monitoring (protein only)
    ca_indices = [
        a.index for a in all_atoms
        if a.name == "CA"
        and a.residue.name not in {"HOH", "WAT", "NA", "CL"}
        and a.element is not None
    ]
    log(f"   Cα atoms for RMSD tracking: {len(ca_indices)}")

    # ── Step 4: Build simulation and minimise ──────────────────────────────
    log(f"── Step 4/6: Energy minimisation (max {minimize_max_iterations} steps) …")

    integrator = LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    simulation = _build_simulation(
        modeller, system, integrator, platform_name, cuda_precision, log
    )
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature_kelvin * unit.kelvin)

    simulation.minimizeEnergy(maxIterations=minimize_max_iterations)
    # Run a second round if energy is still very high — some large proteins
    # need more iterations to fully relax crystal contacts.
    _state_post_min = simulation.context.getState(getEnergy=True)
    _pe_post_min = _state_post_min.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    if not np.isfinite(_pe_post_min) or _pe_post_min > 0:
        log(f"   Post-minimisation PE = {_pe_post_min:.0f} kJ/mol — running additional "
            f"{minimize_max_iterations} steps …")
        simulation.minimizeEnergy(maxIterations=minimize_max_iterations)
    log("   Minimisation complete.")

    # ── Snapshot: post-minimisation ────────────────────────────────────────
    _min_state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    _min_pos_nm = np.asarray(
        _min_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    step4_full = outdir / "step4_minimized_full.pdb"
    _save_pdb(modeller.topology, _min_state.getPositions(), step4_full)
    step4_prot = outdir / "step4_minimized_protein_only.pdb"
    _save_protein_only_pdb(all_atoms, protein_atom_indices, _min_pos_nm, step4_prot)
    log(f"   Snapshot saved: {step4_full}  (full solvated)")
    log(f"   Snapshot saved: {step4_prot}  (protein only)")

    # Minimization updates coordinates but does not reliably initialize velocities
    # for subsequent dynamics. Reset them so the first integration step starts
    # from a valid Maxwell-Boltzmann distribution.
    simulation.context.setVelocitiesToTemperature(temperature_kelvin * unit.kelvin)
    if verbose:
        # Fail fast with a clear error if CUDA produced/retained non-finite
        # state. Without this, OpenMM throws "Particle coordinate is NaN"
        # later during stepping.
        state0 = simulation.context.getState(getPositions=True, getVelocities=True)
        pos_nm = np.asarray(
            state0.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        if not np.isfinite(pos_nm).all():
            raise ValueError("NaN/Inf detected in particle positions after minimization.")
        vel_nm_ps = np.asarray(
            state0.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            dtype=np.float64,
        )
        if not np.isfinite(vel_nm_ps).all():
            raise ValueError("NaN/Inf detected in particle velocities after minimization.")

    # ── Step 5: Equilibration (NVT with restraints → NPT) ─────────────────
    if nvt_steps > 0:
        log(f"── Step 5a/6: NVT equilibration ({nvt_steps} steps = "
            f"{nvt_steps/steps_per_ps:.1f} ps, protein restrained) …")
        # Add heavy-atom restraints (release before production)
        protein_heavy_set = set(protein_atom_indices)
        protein_heavy_indices = [
            a.index for a in all_atoms
            if a.index in protein_heavy_set and a.element.symbol != "H"
        ]
        # Use the current (minimized) coordinates as restraint reference positions.
        # Otherwise, restraint targets may be "stale" (from before minimization),
        # creating large initial forces and risking numerical instability.
        current_positions = simulation.context.getState(getPositions=True).getPositions()

        # Pre-NVT sanity check: verify energy is finite before adding restraints.
        _check_state_finite(simulation, label="pre-NVT (after minimization)")

        _add_restraints(
            system,
            simulation,
            current_positions,
            protein_heavy_indices,
            equil_restraint_k_kcal,
        )

        # ── Re-minimise with restraints in place ────────────────────────────
        # Adding restraint forces changes the energy landscape.  Even though
        # we use the post-minimisation coordinates as reference positions, the
        # mixed-precision rounding on CUDA means particles are NOT exactly at
        # their reference positions, so the restraint potential contributes
        # non-trivially.  A quick re-minimisation relaxes these micro-clashes
        # before any integration step occurs.
        log("   Re-minimising with restraints in place …")
        simulation.minimizeEnergy(maxIterations=2000)
        _check_state_finite(simulation, label="post-restraint re-minimization")

        # ── Staged timestep + temperature warmup ────────────────────────────
        # Large solvated systems (>100k atoms) can blow up on the very first
        # integration step even after minimization, because mixed-precision
        # CUDA rounds coordinates slightly, causing tiny inter-atomic clashes
        # that hit stiff restraints or PME cutoffs.  Running a short warmup at
        # progressively larger timesteps AND starting from a low temperature
        # (small velocities → smaller initial displacements) lets those micro-
        # clashes dissipate before committing to the full NVT run.
        prod_timestep = integrator.getStepSize().value_in_unit(unit.femtoseconds)

        # Temperature ramp: start at 50 K and ramp to target.
        _WARMUP_TEMP_K = 50.0
        simulation.context.setVelocitiesToTemperature(
            _WARMUP_TEMP_K * unit.kelvin
        )
        log(f"   Starting temperature ramp from {_WARMUP_TEMP_K} K → "
            f"{temperature_kelvin} K")

        # Each warmup stage changes EITHER timestep OR temperature, never both
        # simultaneously. Changing both at once doubles the displacement per step
        # and velocity magnitude together, greatly increasing the risk of a force
        # spike and NaN coordinates.
        warmup_stages: list[tuple[float, int, str, float]] = [
            # (dt_fs,  n_steps,  label,  temperature_K)
            (0.1,   200,  "warmup 0.1 fs × 200 steps",        _WARMUP_TEMP_K),
            (0.25,  200,  "warmup 0.25 fs × 200 steps",       _WARMUP_TEMP_K),
            (0.5,   500,  "warmup 0.5 fs × 500 steps",        _WARMUP_TEMP_K),  # dt only
            (0.5,   500,  "warmup 0.5 fs × 500 steps @100K",  100.0),           # T only
            (1.0,   500,  "warmup 1.0 fs × 500 steps @100K",  100.0),           # dt only
            (1.0,   500,  "warmup 1.0 fs × 500 steps @200K",  200.0),           # T only
            (nvt_timestep_fs, nvt_steps,
             f"NVT {nvt_timestep_fs} fs × {nvt_steps} steps",
             temperature_kelvin),
        ]
        # Collapse consecutive stages that are identical in (dt, T) so we don't
        # double-count when nvt_timestep_fs happens to match a warmup entry.
        seen_dt_T: set[tuple[float, float]] = set()
        deduped_stages: list[tuple[float, int, str, float]] = []
        for _dt, _n, _lbl, _T in warmup_stages:
            key = (_dt, _T)
            if key not in seen_dt_T or _lbl.startswith("NVT"):
                deduped_stages.append((_dt, _n, _lbl, _T))
                seen_dt_T.add(key)

        for stage_dt, stage_steps, stage_label, stage_T in deduped_stages:
            log(f"   [{stage_label}] timestep={stage_dt} fs, T={stage_T} K …")
            integrator.setStepSize(stage_dt * unit.femtoseconds)
            integrator.setTemperature(stage_T * unit.kelvin)

            # Snapshot positions before stepping so we can restore on NaN crash.
            _pre_stage_state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            _pre_stage_positions = _pre_stage_state.getPositions()

            try:
                simulation.step(stage_steps)
            except Exception as exc:
                # ── Automatic retry: restore checkpoint, halve dt, keep T ───
                retry_dt = stage_dt / 2.0
                retry_steps = stage_steps * 2  # keep total integration time
                log(f"   ⚠ Stage '{stage_label}' crashed: {exc}")
                log(f"     Restoring pre-stage checkpoint and retrying "
                    f"with dt={retry_dt} fs × {retry_steps} steps …")
                # Restore the last known-good positions so we are not stepping
                # from a NaN state (which always re-crashes immediately).
                simulation.context.setPositions(_pre_stage_positions)
                simulation.minimizeEnergy(maxIterations=2000)
                _check_state_finite(simulation, label=f"retry pre-step ({stage_label})")
                simulation.context.setVelocitiesToTemperature(
                    max(50.0, stage_T * 0.5) * unit.kelvin
                )
                integrator.setStepSize(retry_dt * unit.femtoseconds)
                try:
                    simulation.step(retry_steps)
                except Exception as exc2:
                    raise RuntimeError(
                        f"NVT equilibration crashed at stage '{stage_label}' "
                        f"even after retry with dt={retry_dt} fs "
                        f"(k={equil_restraint_k_kcal} kcal/mol/Å², "
                        f"{len(protein_heavy_indices)} restrained atoms). "
                        f"Try: --equil-restraint-k 2.0 or "
                        f"--cuda-precision double. "
                        f"Original error: {exc2}"
                    ) from exc2

        # Restore production timestep and temperature.
        integrator.setStepSize(prod_timestep * unit.femtoseconds)
        integrator.setTemperature(temperature_kelvin * unit.kelvin)

        _remove_restraints(system, simulation)
        log(f"   NVT equilibration done, restraints removed. "
            f"Timestep restored to {prod_timestep} fs, T={temperature_kelvin} K.")

        # ── Snapshot: post-NVT equilibration ──────────────────────────────
        _nvt_state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        _nvt_pos_nm = np.asarray(
            _nvt_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        step5a_full = outdir / "step5a_nvt_equil_full.pdb"
        _save_pdb(modeller.topology, _nvt_state.getPositions(), step5a_full)
        step5a_prot = outdir / "step5a_nvt_equil_protein_only.pdb"
        _save_protein_only_pdb(all_atoms, protein_atom_indices, _nvt_pos_nm, step5a_prot)
        log(f"   Snapshot saved: {step5a_full}  (full solvated)")
        log(f"   Snapshot saved: {step5a_prot}  (protein only)")

    if npt_steps > 0:
        log(f"── Step 5b/6: NPT equilibration ({npt_steps} steps = "
            f"{npt_steps/steps_per_ps:.1f} ps, {pressure_bar} bar) …")
        barostat = MonteCarloBarostat(
            pressure_bar * unit.bar, temperature_kelvin * unit.kelvin
        )
        system.addForce(barostat)
        simulation.context.reinitialize(preserveState=True)
        simulation.step(npt_steps)
        # Remove barostat for production NVT
        system.removeForce(system.getNumForces() - 1)
        simulation.context.reinitialize(preserveState=True)
        log("   NPT equilibration done, barostat removed.")

        # ── Snapshot: post-NPT equilibration ──────────────────────────────
        _npt_state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        _npt_pos_nm = np.asarray(
            _npt_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        step5b_full = outdir / "step5b_npt_equil_full.pdb"
        _save_pdb(modeller.topology, _npt_state.getPositions(), step5b_full)
        step5b_prot = outdir / "step5b_npt_equil_protein_only.pdb"
        _save_protein_only_pdb(all_atoms, protein_atom_indices, _npt_pos_nm, step5b_prot)
        log(f"   Snapshot saved: {step5b_full}  (full solvated)")
        log(f"   Snapshot saved: {step5b_prot}  (protein only)")

    # ── Step 6: Production MD with conformational-change detection ─────────
    production_time_ps = production_steps * timestep_fs / 1000.0
    log(f"── Step 6/6: Production NVT MD ({production_steps} steps = "
        f"{production_time_ps:.1f} ps), "
        f"RMSD threshold {rmsd_threshold_angstrom:.1f} Å …")

    if save_trajectory:
        dcd_path = outdir / "trajectory.dcd"
        simulation.reporters.append(
            DCDReporter(str(dcd_path), report_interval_steps)
        )
    else:
        dcd_path = None

    # Progress reporter to stdout
    _n_reports = max(1, production_steps // report_interval_steps)
    log_interval = max(1, production_steps // min(_n_reports, 20))
    simulation.reporters.append(
        StateDataReporter(
            _PrintWrapper(log),
            log_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=True,
            progress=True,
            totalSteps=production_steps,
            separator="\t",
        )
    )

    # Reference Cα positions for RMSD computation (post-equilibration crystal coords)
    eq_state = simulation.context.getState(getPositions=True)
    eq_positions_nm = np.asarray(
        eq_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    ref_ca_positions_nm = eq_positions_nm[ca_indices].copy()  # reference for RMSD

    extracted_frames: list[DynamicsFrame] = []
    steps_run = 0

    while steps_run < production_steps:
        chunk = min(report_interval_steps, production_steps - steps_run)
        simulation.step(chunk)
        steps_run += chunk

        current_time_ps = (steps_run * timestep_fs / 1000.0)

        state = simulation.context.getState(getPositions=True)
        positions_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )

        # Compute backbone RMSD from the last extracted frame's reference
        ca_positions_nm = positions_nm[ca_indices]
        rmsd_A = _rmsd(ca_positions_nm, ref_ca_positions_nm) * 10.0  # nm → Å

        is_first_frame = len(extracted_frames) == 0
        if is_first_frame or rmsd_A >= rmsd_threshold_angstrom:
            rmsd_report = 0.0 if is_first_frame else rmsd_A
            log(
                f"   → Frame {len(extracted_frames):4d}  "
                f"t={current_time_ps:8.1f} ps  "
                f"RMSD={rmsd_report:.2f} Å  — extracting"
            )

            # Extract protein-only positions and build a PDB string
            prot_pos_nm = positions_nm[protein_atom_indices]
            protein_pdb_str, frame_atomarray = _extract_protein_frame(
                all_atoms=all_atoms,
                modeller=modeller,
                protein_atom_indices=protein_atom_indices,
                positions_nm=positions_nm,
                base_atomarray=base_atomarray,
                n_protein_atoms=n_protein_atoms,
            )

            extracted_frames.append(
                DynamicsFrame(
                    frame_index=len(extracted_frames),
                    simulation_time_ps=current_time_ps,
                    rmsd_from_prev_angstrom=rmsd_report,
                    protein_pdb_string=protein_pdb_str,
                    protein_positions_nm=prot_pos_nm,
                    atomarray=frame_atomarray,
                )
            )

            # Update reference for next RMSD comparison
            ref_ca_positions_nm = ca_positions_nm.copy()

    log(f"   Production complete. Extracted {len(extracted_frames)} conformational frames.")

    return DynamicsResult(
        frames=extracted_frames,
        trajectory_dcd=dcd_path if save_trajectory else None,
        topology_pdb=topo_pdb_path,
        prepared_protein=prepared_protein,
        n_protein_atoms=n_protein_atoms,
        simulation_time_ps=production_time_ps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _identify_protein_atoms(all_atoms: list) -> list[int]:
    """Return indices of protein atoms (excluding water, ions, and solvent).

    Works across varied PDB chain/residue naming conventions.
    """
    protein_indices: list[int] = []
    for a in all_atoms:
        if a.element is None:
            continue
        res_name = a.residue.name.strip().upper()
        if res_name in _SOLVENT_ION_RESNAMES:
            continue
        chain_id = a.residue.chain.id
        # OpenMM places water/ions on chain "W" or similar; skip if the
        # residue name is a water variant we haven't yet listed.
        if chain_id == "W":
            continue
        protein_indices.append(a.index)
    return protein_indices


def _build_simulation(
    modeller: Modeller,
    system: object,
    integrator: object,
    platform_name: str,
    cuda_precision: str,
    log: _Logger,
) -> Simulation:
    """Construct an OpenMM Simulation, falling back to CPU if CUDA unavailable."""
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties: dict[str, str] = {}
        if platform_name.upper() == "CUDA":
            properties["Precision"] = cuda_precision
        elif platform_name.upper() == "OPENCL":
            properties["Precision"] = cuda_precision
        sim = Simulation(modeller.topology, system, integrator, platform, properties)
        log(f"   Platform: {platform_name} ({cuda_precision} precision)")
        return sim
    except Exception as exc:
        log(f"   Warning: {platform_name} unavailable ({exc}), falling back to CPU.")
        return Simulation(modeller.topology, system, integrator)


def _check_state_finite(simulation: Simulation, label: str = "") -> None:
    """Raise ValueError if positions or potential energy contain NaN/Inf.

    Call this before starting any integration phase so that instability is
    caught with a clear message rather than an opaque OpenMM crash.
    """
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    pos_nm = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    if not np.isfinite(pos_nm).all():
        tag = f" [{label}]" if label else ""
        raise ValueError(f"NaN/Inf in particle positions{tag}.")
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(pe):
        tag = f" [{label}]" if label else ""
        raise ValueError(
            f"Non-finite potential energy ({pe} kJ/mol){tag}. "
            "The system is likely in a high-energy configuration that was not "
            "resolved by minimization."
        )


def _add_restraints(
    system: object,
    simulation: Simulation,
    positions: object,
    atom_indices: list[int],
    k_kcal: float,
) -> None:
    """Add harmonic positional restraints to *atom_indices* in-place.

    Adds the force to *system* and re-initialises the context so that the
    force takes effect immediately.
    """
    if not atom_indices or k_kcal <= 0.0:
        return
    positions_nm = np.asarray(
        positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    k_kj = k_kcal * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2
    force = CustomExternalForce(
        "0.5*k_equil*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    force.addGlobalParameter("k_equil", k_kj)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for idx in atom_indices:
        x0, y0, z0 = positions_nm[int(idx)]
        force.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    system.addForce(force)
    simulation.context.reinitialize(preserveState=True)


def _remove_restraints(system: object, simulation: Simulation) -> None:
    """Remove the last force added to *system* (the equilibration restraints)."""
    system.removeForce(system.getNumForces() - 1)
    simulation.context.reinitialize(preserveState=True)


def _extract_protein_frame(
    *,
    all_atoms: list,
    modeller: Modeller,
    protein_atom_indices: list[int],
    positions_nm: npt.NDArray[np.float64],
    base_atomarray: object,
    n_protein_atoms: int,
) -> tuple[str, object]:
    """Return (protein_pdb_string, biotite_atomarray) for the current frame.

    Strips water and ions; keeps only protein atoms.
    """
    # Build a filtered atom list + position list for PDBFile.writeFile
    protein_idx_set = set(protein_atom_indices)

    # Create a minimal topology for the protein-only atoms
    prot_topology = Topology()
    prot_positions_list = []
    global_to_local: dict[int, int] = {}

    chain_map: dict[object, object] = {}
    res_map: dict[object, object] = {}

    from openmm import Vec3
    for atom in all_atoms:
        if atom.index not in protein_idx_set:
            continue
        # Map chains
        orig_chain = atom.residue.chain
        if orig_chain not in chain_map:
            chain_map[orig_chain] = prot_topology.addChain(orig_chain.id)
        # Map residues
        orig_res = atom.residue
        if orig_res not in res_map:
            res_map[orig_res] = prot_topology.addResidue(
                orig_res.name, chain_map[orig_chain], orig_res.id
            )
        prot_topology.addAtom(
            atom.name, atom.element, res_map[orig_res]
        )
        local_idx = len(global_to_local)
        global_to_local[atom.index] = local_idx
        xyz_nm = positions_nm[atom.index]
        prot_positions_list.append(
            Vec3(xyz_nm[0] * 10.0, xyz_nm[1] * 10.0, xyz_nm[2] * 10.0)
        )

    buf = io.StringIO()
    PDBFile.writeFile(prot_topology, prot_positions_list, buf, keepIds=True)
    pdb_str = buf.getvalue()

    # Build biotite AtomArray: copy the base array (protein only) and update coords
    frame_arr = copy.copy(base_atomarray)
    prot_positions_A = positions_nm[protein_atom_indices] * 10.0  # nm → Å
    # base_atomarray has the same atom ordering as the prepared protein
    arr_len = min(len(frame_arr), len(prot_positions_A))
    frame_arr.coord = prot_positions_A[:arr_len].astype(np.float32)

    return pdb_str, frame_arr


def _save_pdb(topology: object, positions: object, path: Path) -> None:
    buf = io.StringIO()
    PDBFile.writeFile(topology, positions, buf)
    path.write_text(buf.getvalue())


def _save_protein_only_pdb(
    all_atoms: list,
    protein_atom_indices: list[int],
    positions_nm: npt.NDArray[np.float64],
    path: Path,
) -> None:
    """Write a protein-only PDB (no water/ions) for easy visualisation in PyMol."""
    from openmm import Vec3

    protein_idx_set = set(protein_atom_indices)
    prot_topology = Topology()
    prot_positions_list = []
    chain_map: dict[object, object] = {}
    res_map: dict[object, object] = {}

    for atom in all_atoms:
        if atom.index not in protein_idx_set:
            continue
        orig_chain = atom.residue.chain
        if orig_chain not in chain_map:
            chain_map[orig_chain] = prot_topology.addChain(orig_chain.id)
        orig_res = atom.residue
        if orig_res not in res_map:
            res_map[orig_res] = prot_topology.addResidue(
                orig_res.name, chain_map[orig_chain], orig_res.id
            )
        prot_topology.addAtom(atom.name, atom.element, res_map[orig_res])
        
        # Multiply by 10 to firmly convert nanometers to Angstroms
        # and feed a raw list of Vec3. PDBFile treats un-unit-ified Vec3s as Angstroms.
        xyz_nm = positions_nm[atom.index]
        prot_positions_list.append(Vec3(xyz_nm[0] * 10.0, xyz_nm[1] * 10.0, xyz_nm[2] * 10.0))

    buf = io.StringIO()
    PDBFile.writeFile(prot_topology, prot_positions_list, buf, keepIds=True)
    path.write_text(buf.getvalue())


def _rmsd(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> float:
    """RMSD between two coordinate arrays (same units, shape (N,3) or (1,3))."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

class _Logger:
    """Callable that prints only when *verbose* is True."""

    def __init__(self, verbose: bool) -> None:
        self._verbose = verbose

    def __call__(self, msg: str) -> None:
        if self._verbose:
            print(msg, flush=True)


class _PrintWrapper:
    """File-like wrapper that passes lines to a :class:`_Logger`."""

    def __init__(self, log: _Logger) -> None:
        self._log = log

    def write(self, text: str) -> None:
        text = text.rstrip()
        if text:
            self._log(f"   [MD]  {text}")

    def flush(self) -> None:
        pass
