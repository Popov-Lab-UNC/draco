from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from draco.ligand_preparation import conformer_to_pdb_block
from deprecated.overlay import OverlayResult
from draco.protein_preparation import PreparedProtein

try:
    from openmm import CustomExternalForce, LangevinIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
except ImportError:  # pragma: no cover
    from simtk.openmm import CustomExternalForce, LangevinIntegrator, Platform, unit  # type: ignore
    from simtk.openmm.app import (  # type: ignore
        ForceField,
        HBonds,
        Modeller,
        NoCutoff,
        PDBFile,
        Simulation,
    )

KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2 = 418.4

# Cache for OpenMM template generator to avoid repeating AM1-BCC parameterization on the same ligand.
_TEMPLATE_GENERATOR_CACHE = {}

@dataclass(frozen=True)
class LocalMinimizationResult:
    initial_energy_kj_per_mol: float
    final_energy_kj_per_mol: float
    ligand_heavy_atom_rmsd_angstrom: float
    protein_atoms_flexible: int
    protein_atoms_restrained: int
    ligand_atoms_restrained: int
    minimized_positions_angstrom: npt.NDArray[np.float64]
    minimized_complex_pdb: str
    # --- Physics-based interaction score (None if unavailable) ---
    interaction_energy_kj_per_mol: float | None = None
    # --- Induced-fit MD results (None if run_induced_fit_md=False) ---
    induced_fit_ligand_rmsd_angstrom: float | None = None
    induced_fit_final_energy_kj_per_mol: float | None = None
    induced_fit_complex_pdb: str | None = None

    def save_pdb(self, output_path: str | Path) -> None:
        Path(output_path).write_text(self.minimized_complex_pdb)


def minimize_overlay_pose(
    prepared_protein: PreparedProtein,
    overlay_result: OverlayResult,
    *,
    protein_forcefield_files: tuple[str, ...] = ("amber14-all.xml", "amber14/tip3pfb.xml"),
    openff_forcefield: str = "openff-2.3.0",
    shell_radius_angstrom: float = 8.0,
    protein_restraint_k_kcal_per_mol_A2: float = 10.0,
    ligand_restraint_k_kcal_per_mol_A2: float = 1.0,
    temperature_kelvin: float = 300.0,
    friction_per_ps: float = 1.0,
    timestep_fs: float = 2.0,
    minimize_max_iterations: int = 2000,
    platform_name: str | None = None,
    ligand_residue_name: str = "LIG",
    ligand_chain_id: str = "L",
    ligand_residue_id: int = 1,
    output_path: str | Path | None = None,
    write_pdb_on_error: bool = True,
    error_suffix: str = "_error",
    compute_interaction_energy: bool = True,
    run_induced_fit_md: bool = False,
    induced_fit_steps: int = 25000,
) -> LocalMinimizationResult:
    """Minimise one overlay pose against a pre-prepared protein.

    The caller is responsible for running ``prepare_protein()`` once and
    passing the resulting ``PreparedProtein`` here.  This avoids repeating
    the (slow) PDBFixer repair step for every pose.

    Parameters
    ----------
    compute_interaction_energy:
        If True (default), compute an approximate protein–ligand interaction
        energy by zeroing NonbondedForce parameters for each component in
        turn.  Returns ``None`` gracefully if the NonbondedForce cannot be
        located (e.g. when GAFF custom forces are used exclusively).
    run_induced_fit_md:
        If True, after minimization the ligand restraints are released and
        protein restraints softened 5×, then ``induced_fit_steps`` MD steps
        are run (default 25 000 × 2 fs = 50 ps) followed by a short
        re-minimization.  The RMSD of the ligand from the minimized pose is
        reported — a large drift indicates the pose is not stable.
    induced_fit_steps:
        Number of MD steps for the induced-fit phase (default 25 000 = 50 ps
        at a 2 fs timestep).
    """
    ligand_pdb = PDBFile(
        io.StringIO(
            conformer_to_pdb_block(
                overlay_result.conformer,
                overlay_result.transformed_all_atom_coords,
                residue_name=ligand_residue_name,
                chain_id=ligand_chain_id,
                residue_id=ligand_residue_id,
            )
        )
    )

    modeller = Modeller(prepared_protein.topology, prepared_protein.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.positions)

    # Write an initial complex PDB early so we get per-pose artifacts even
    # if createSystem/minimization fails (useful for debugging templates).
    initial_out = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, initial_out, keepIds=True)
    initial_complex_pdb = initial_out.getvalue()

    try:
        forcefield = ForceField(*protein_forcefield_files)
        _register_ligand_template(
            forcefield, overlay_result, openff_forcefield=openff_forcefield
        )

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )

        topology_atoms = list(modeller.topology.atoms())
        positions_nm = np.asarray(
            modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64
        )

        ligand_atom_indices = [
            atom.index
            for atom in topology_atoms
            if atom.residue.name == ligand_residue_name
        ]
        if not ligand_atom_indices:
            raise ValueError(
                f"Could not identify ligand residue '{ligand_residue_name}' in topology"
            )

        protein_atom_indices = [
            atom.index
            for atom in topology_atoms
            if atom.residue.name != ligand_residue_name and atom.element is not None
        ]
        if not protein_atom_indices:
            raise ValueError("Could not identify protein atoms in topology")

        restrained_protein_indices, flexible_protein_indices = _partition_protein_atoms_by_shell(
            positions_nm=positions_nm,
            protein_atom_indices=protein_atom_indices,
            ligand_atom_indices=ligand_atom_indices,
            shell_radius_angstrom=shell_radius_angstrom,
        )

        n_protein_restrained = _add_positional_restraints(
            system=system,
            positions_nm=positions_nm,
            atom_indices=restrained_protein_indices,
            k_kcal_per_mol_a2=protein_restraint_k_kcal_per_mol_A2,
            k_param_name="k_protein_posres",
        )
        n_ligand_restrained = _add_positional_restraints(
            system=system,
            positions_nm=positions_nm,
            atom_indices=ligand_atom_indices,
            k_kcal_per_mol_a2=ligand_restraint_k_kcal_per_mol_A2,
            k_param_name="k_ligand_posres",
        )

        integrator = LangevinIntegrator(
            temperature_kelvin * unit.kelvin,
            friction_per_ps / unit.picosecond,
            timestep_fs * unit.femtoseconds,
        )
        simulation = (
            Simulation(
                modeller.topology,
                system,
                integrator,
                Platform.getPlatformByName(platform_name),
            )
            if platform_name
            else Simulation(modeller.topology, system, integrator)
        )
        simulation.context.setPositions(modeller.positions)

        initial_state = simulation.context.getState(getEnergy=True, getPositions=True)
        initial_energy = float(
            initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        initial_positions_nm = np.asarray(
            initial_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )

        simulation.minimizeEnergy(maxIterations=minimize_max_iterations)

        final_state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = float(
            final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        final_positions_nm = np.asarray(
            final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )

        ligand_heavy_relative = list(overlay_result.conformer.heavy_atom_indices)
        ligand_heavy_abs = [
            ligand_atom_indices[idx]
            for idx in ligand_heavy_relative
            if idx < len(ligand_atom_indices)
        ]
        ligand_rmsd = _rmsd(
            initial_positions_nm[ligand_heavy_abs] * 10.0,
            final_positions_nm[ligand_heavy_abs] * 10.0,
        )

        # ── Optional: interaction energy decomposition ─────────────────────────────
        interaction_energy: float | None = None
        if compute_interaction_energy:
            interaction_energy = _compute_interaction_energy(
                simulation=simulation,
                ligand_atom_indices=ligand_atom_indices,
                protein_atom_indices=protein_atom_indices,
                final_energy_kj_per_mol=final_energy,
            )

        # ── Optional: induced-fit MD phase ───────────────────────────────────
        induced_fit_ligand_rmsd: float | None = None
        induced_fit_final_energy: float | None = None
        induced_fit_complex_pdb: str | None = None

        if run_induced_fit_md:
            # Release ligand; soften protein backbone restraints 5×
            simulation.context.setParameter("k_ligand_posres", 0.0)
            simulation.context.setParameter(
                "k_protein_posres",
                (protein_restraint_k_kcal_per_mol_A2 / 5.0) * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2,
            )
            simulation.step(induced_fit_steps)
            simulation.minimizeEnergy(maxIterations=500)

            ifd_state = simulation.context.getState(getEnergy=True, getPositions=True)
            induced_fit_final_energy = float(
                ifd_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )
            ifd_positions_nm = np.asarray(
                ifd_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            # RMSD of ligand heavy atoms relative to the post-minimization pose
            induced_fit_ligand_rmsd = _rmsd(
                final_positions_nm[ligand_heavy_abs] * 10.0,
                ifd_positions_nm[ligand_heavy_abs] * 10.0,
            )
            ifd_out = io.StringIO()
            PDBFile.writeFile(modeller.topology, ifd_state.getPositions(), ifd_out, keepIds=True)
            induced_fit_complex_pdb = ifd_out.getvalue()

        final_positions = final_state.getPositions()
        final_out = io.StringIO()
        PDBFile.writeFile(modeller.topology, final_positions, final_out, keepIds=True)
        minimized_pdb = final_out.getvalue()

        if output_path:
            Path(output_path).write_text(minimized_pdb)

        return LocalMinimizationResult(
            initial_energy_kj_per_mol=initial_energy,
            final_energy_kj_per_mol=final_energy,
            ligand_heavy_atom_rmsd_angstrom=ligand_rmsd,
            protein_atoms_flexible=len(flexible_protein_indices),
            protein_atoms_restrained=n_protein_restrained,
            ligand_atoms_restrained=n_ligand_restrained,
            minimized_positions_angstrom=final_positions_nm * 10.0,
            minimized_complex_pdb=minimized_pdb,
            interaction_energy_kj_per_mol=interaction_energy,
            induced_fit_ligand_rmsd_angstrom=induced_fit_ligand_rmsd,
            induced_fit_final_energy_kj_per_mol=induced_fit_final_energy,
            induced_fit_complex_pdb=induced_fit_complex_pdb,
        )
    except Exception:
        if output_path and write_pdb_on_error:
            out_path = Path(output_path)
            error_path = out_path.with_name(out_path.stem + error_suffix + out_path.suffix)
            error_path.write_text(initial_complex_pdb)
        raise


def _compute_interaction_energy(
    *,
    simulation: object,
    ligand_atom_indices: list[int],
    protein_atom_indices: list[int],
    final_energy_kj_per_mol: float,
) -> float | None:
    """Approximate protein–ligand interaction energy via NonbondedForce decomposition.

    Strategy
    --------
    1. Record E(complex) — already computed as *final_energy_kj_per_mol*.
    2. Zero the charge and epsilon of all **ligand** atoms → get E(protein_alone).
    3. Restore ligand params; zero charge and epsilon of all **protein** atoms →
       get E(ligand_alone).
    4. Restore protein params.
    5. Return  ΔE = E(complex) − E(protein_alone) − E(ligand_alone).

    This isolates the nonbonded cross-interaction terms (van der Waals + electrostatics
    between protein and ligand).  Intramolecular bonded contributions (bonds, angles,
    torsions) cancel exactly in the subtraction because they are unchanged by zeroing
    nonbonded parameters.

    Returns
    -------
    float or None
        Interaction energy in kJ/mol.  More negative = more favourable.
        Returns ``None`` if the NonbondedForce cannot be found (e.g. custom forces only).
    """
    # Locate the standard NonbondedForce (charge + LJ)
    nb_force = None
    for force in simulation.system.getForces():  # type: ignore[attr-defined]
        if type(force).__name__ == "NonbondedForce":
            nb_force = force
            break
    if nb_force is None:
        return None

    try:
        # ── Save and zero ligand params ──────────────────────────────────────
        lig_saved: list[tuple] = []
        for idx in ligand_atom_indices:
            params = nb_force.getParticleParameters(idx)
            lig_saved.append(params)
            # Zero charge (index 0) and epsilon (index 2); keep sigma (index 1)
            nb_force.setParticleParameters(idx, 0.0 * params[0].unit, params[1], 0.0 * params[2].unit)
        nb_force.updateParametersInContext(simulation.context)  # type: ignore[attr-defined]

        state = simulation.context.getState(getEnergy=True)  # type: ignore[attr-defined]
        e_protein = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        # ── Restore ligand, save and zero protein params ─────────────────────
        for i, idx in enumerate(ligand_atom_indices):
            nb_force.setParticleParameters(idx, *lig_saved[i])

        prot_saved: list[tuple] = []
        for idx in protein_atom_indices:
            params = nb_force.getParticleParameters(idx)
            prot_saved.append(params)
            nb_force.setParticleParameters(idx, 0.0 * params[0].unit, params[1], 0.0 * params[2].unit)
        nb_force.updateParametersInContext(simulation.context)  # type: ignore[attr-defined]

        state = simulation.context.getState(getEnergy=True)  # type: ignore[attr-defined]
        e_ligand = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        # ── Restore protein params ───────────────────────────────────────────
        for i, idx in enumerate(protein_atom_indices):
            nb_force.setParticleParameters(idx, *prot_saved[i])
        nb_force.updateParametersInContext(simulation.context)  # type: ignore[attr-defined]

        return final_energy_kj_per_mol - e_protein - e_ligand

    except Exception:
        # Restore everything on failure so simulation context is not corrupted
        try:
            for i, idx in enumerate(ligand_atom_indices):
                if i < len(lig_saved):
                    nb_force.setParticleParameters(idx, *lig_saved[i])
            for i, idx in enumerate(protein_atom_indices):
                if i < len(prot_saved):
                    nb_force.setParticleParameters(idx, *prot_saved[i])
            nb_force.updateParametersInContext(simulation.context)  # type: ignore[attr-defined]
        except Exception:
            pass
        return None


def _register_ligand_template(
    forcefield: ForceField,
    overlay_result: OverlayResult,
    *,
    openff_forcefield: str = "openff-2.3.0",
) -> None:
    """Register SMIRNOFF ligand parameters with *forcefield*.

    Uses OpenFF Sage (openff-2.3.0 by default) via SMIRNOFFTemplateGenerator —
    the recommended ligand force field for protein-ligand systems alongside
    AMBER ff14SB.  Falls back to GAFF-2.11 if openff-toolkit is unavailable.

    """
    cache_key = openff_forcefield
    if cache_key in _TEMPLATE_GENERATOR_CACHE:
        forcefield.registerTemplateGenerator(_TEMPLATE_GENERATOR_CACHE[cache_key].generator)
        return

    ligand_mol = Chem.MolFromMolBlock(overlay_result.conformer.mol_block, removeHs=False)
    if ligand_mol is None:
        raise ValueError("Failed to build RDKit molecule from ligand mol_block")

    # The Generator only cares about the topology + charges of the molecule to generate the template XML.
    # Therefore, we only need to construct it once per ligand, not once for every conformer's 3D coordinates.

    try:
        from openff.toolkit.topology import Molecule  # type: ignore
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        off_mol = Molecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)
        generator = SMIRNOFFTemplateGenerator(
            molecules=[off_mol],
            forcefield=openff_forcefield,
        )
        _TEMPLATE_GENERATOR_CACHE[cache_key] = generator
        forcefield.registerTemplateGenerator(generator.generator)
        return
    except Exception as exc:
        smirnoff_error = str(exc)

    # Fallback: GAFF-2.11 (requires ambertools)
    try:
        from openff.toolkit.topology import Molecule  # type: ignore
        from openmmforcefields.generators import GAFFTemplateGenerator

        off_mol = Molecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)
        generator = GAFFTemplateGenerator(molecules=[off_mol], forcefield="gaff-2.11")
        _TEMPLATE_GENERATOR_CACHE["gaff-2.11"] = generator
        forcefield.registerTemplateGenerator(generator.generator)
        return
    except Exception as exc:
        gaff_error = str(exc)

    raise ImportError(
        f"Could not register ligand parameters with {openff_forcefield!r} "
        f"(SMIRNOFFTemplateGenerator: {smirnoff_error}) "
        f"or GAFF-2.11 (GAFFTemplateGenerator: {gaff_error}). "
        "Ensure openmmforcefields and openff-toolkit are installed."
    )


def _partition_protein_atoms_by_shell(
    *,
    positions_nm: npt.NDArray[np.float64],
    protein_atom_indices: list[int],
    ligand_atom_indices: list[int],
    shell_radius_angstrom: float,
) -> tuple[list[int], list[int]]:
    protein_coords = positions_nm[np.asarray(protein_atom_indices, dtype=int)]
    ligand_coords = positions_nm[np.asarray(ligand_atom_indices, dtype=int)]
    shell_nm = shell_radius_angstrom / 10.0

    deltas = protein_coords[:, None, :] - ligand_coords[None, :, :]
    min_distances_nm = np.linalg.norm(deltas, axis=2).min(axis=1)

    flexible_mask = min_distances_nm <= shell_nm
    flexible = [protein_atom_indices[i] for i in np.where(flexible_mask)[0]]
    restrained = [protein_atom_indices[i] for i in np.where(~flexible_mask)[0]]
    return restrained, flexible


def _add_positional_restraints(
    *,
    system: object,
    positions_nm: npt.NDArray[np.float64],
    atom_indices: list[int],
    k_kcal_per_mol_a2: float,
    k_param_name: str,
) -> int:
    if k_kcal_per_mol_a2 <= 0.0 or not atom_indices:
        return 0

    force = CustomExternalForce(
        f"0.5*{k_param_name}*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    force.addGlobalParameter(
        k_param_name,
        k_kcal_per_mol_a2 * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2,
    )
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for atom_idx in atom_indices:
        x0, y0, z0 = positions_nm[int(atom_idx)]
        force.addParticle(int(atom_idx), [float(x0), float(y0), float(z0)])

    system.addForce(force)
    return len(atom_indices)


def _rmsd(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compute RMSD for shapes {a.shape} and {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

