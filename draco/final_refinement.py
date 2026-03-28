"""final_refinement.py – Local protein-ligand relaxation for GNINA-docked poses.

Role in the Pipeline
--------------------
    gnina_docking → sar_scoring → final_refinement

Applied only to the top-K (conformation, pose) pairs selected after SAR
scoring. Because GNINA-docked poses are already sterically valid (Monte
Carlo sampling avoids ring-threading/atom overlap), this step is gentle
refinement — not clash resolution.

Key difference from the deprecated local_minimization.py:
  - Accepts a GNINA docked pose in SDF-block format (not an OverlayResult)
  - No ligand positional restraints (the docked pose is already reasonable)
  - Protein restraints outside the binding shell are still applied
  - Simpler API matching the new pipeline structure
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rdkit import Chem

from constants import (
    DEFAULT_FORCEFIELD_FILES,
    DEFAULT_OPENFF_FORCEFIELD,
    DEFAULT_REFINEMENT_SHELL_RADIUS,
    DEFAULT_REFINEMENT_PROTEIN_RESTRAINT_K,
    DEFAULT_REFINEMENT_MAX_ITERATIONS,
    KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2,
)

_log = logging.getLogger(__name__)

# Cache for OpenMM template generator to avoid repeating AM1-BCC parameterization on the same ligand.
_TEMPLATE_GENERATOR_CACHE = {}

try:
    from openmm import CustomExternalForce, LangevinIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
except ImportError:  # pragma: no cover
    from simtk.openmm import CustomExternalForce, LangevinIntegrator, Platform, unit  # type: ignore
    from simtk.openmm.app import (  # type: ignore
        ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RefinementResult:
    """Result of local OpenMM minimization of a GNINA-docked pose."""

    initial_energy_kj_per_mol: float
    final_energy_kj_per_mol: float

    ligand_rmsd_from_dock_angstrom: float
    """How much the ligand heavy atoms moved during minimization (Å).
    A large drift (> 2 Å) may indicate the docked pose was not stable."""

    protein_atoms_flexible: int
    """Number of protein atoms in the flexible shell (no restraints)."""

    protein_atoms_restrained: int
    """Number of protein atoms outside the shell (restrained)."""

    interaction_energy_kj_per_mol: float | None
    """Approximate protein–ligand interaction energy (kJ/mol), or None."""

    refined_complex_pdb: str
    """PDB text of the refined protein+ligand complex."""

    status: str = "ok"
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def refine_docked_pose(
    protein_pdb_path: str | Path,
    docked_ligand_sdf_block: str,
    *,
    protein_forcefield_files: tuple[str, ...] = DEFAULT_FORCEFIELD_FILES,
    openff_forcefield: str = DEFAULT_OPENFF_FORCEFIELD,
    shell_radius_angstrom: float = DEFAULT_REFINEMENT_SHELL_RADIUS,
    protein_restraint_k_kcal_per_mol_A2: float = DEFAULT_REFINEMENT_PROTEIN_RESTRAINT_K,
    max_iterations: int = DEFAULT_REFINEMENT_MAX_ITERATIONS,
    platform_name: str = "CPU",
    compute_interaction_energy: bool = True,
    ligand_residue_name: str = "LIG",
    ligand_chain_id: str = "L",
    ligand_residue_id: int = 1,
) -> RefinementResult:
    """Locally minimize a GNINA-docked protein–ligand complex.

    Parameters
    ----------
    protein_pdb_path:
        Path to the protein-only PDB for this conformation (no water, no ions).
        Must match the protein used for docking.
    docked_ligand_sdf_block:
        SDF block string of the docked ligand pose (from ``GninaDockResult.pose_sdf_block``).
    protein_forcefield_files:
        OpenMM ForceField XML files for the protein (default: AMBER14 + TIP3P-FB).
    openff_forcefield:
        OpenFF force field for ligand parameterization (default: ``'openff-2.3.0'``).
    shell_radius_angstrom:
        Protein atoms within this radius of the ligand are freely flexible;
        atoms outside are harmonically restrained (default 8 Å).
    protein_restraint_k_kcal_per_mol_A2:
        Restraint force constant for protein atoms outside the shell (default 10 kcal/mol/Å²).
    max_iterations:
        Maximum energy minimization iterations (default 500; much less than the
        deprecated overlay approach's 2000, since poses are already clash-free).
    platform_name:
        OpenMM platform: ``'CPU'`` (default), ``'CUDA'``, or ``'OpenCL'``.
    compute_interaction_energy:
        If True, compute approximate non-bonded interaction energy via
        NonbondedForce decomposition (default True).
    ligand_residue_name / ligand_chain_id / ligand_residue_id:
        PDB naming for the ligand when building the complex topology.

    Returns
    -------
    RefinementResult
    """
    try:
        return _refine_impl(
            protein_pdb_path=Path(protein_pdb_path),
            docked_ligand_sdf_block=docked_ligand_sdf_block,
            protein_forcefield_files=protein_forcefield_files,
            openff_forcefield=openff_forcefield,
            shell_radius_angstrom=shell_radius_angstrom,
            protein_restraint_k_kcal_per_mol_A2=protein_restraint_k_kcal_per_mol_A2,
            max_iterations=max_iterations,
            platform_name=platform_name,
            compute_interaction_energy=compute_interaction_energy,
            ligand_residue_name=ligand_residue_name,
            ligand_chain_id=ligand_chain_id,
            ligand_residue_id=ligand_residue_id,
        )
    except Exception as exc:
        import traceback
        return RefinementResult(
            initial_energy_kj_per_mol=0.0,
            final_energy_kj_per_mol=0.0,
            ligand_rmsd_from_dock_angstrom=0.0,
            protein_atoms_flexible=0,
            protein_atoms_restrained=0,
            interaction_energy_kj_per_mol=None,
            refined_complex_pdb="",
            status="error",
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────────────────────────────────────

def _refine_impl(
    *,
    protein_pdb_path: Path,
    docked_ligand_sdf_block: str,
    protein_forcefield_files: tuple[str, ...],
    openff_forcefield: str,
    shell_radius_angstrom: float,
    protein_restraint_k_kcal_per_mol_A2: float,
    max_iterations: int,
    platform_name: str,
    compute_interaction_energy: bool,
    ligand_residue_name: str,
    ligand_chain_id: str,
    ligand_residue_id: int,
) -> RefinementResult:
    # ── 1. Load protein ────────────────────────────────────────────────────────
    protein_pdb = PDBFile(str(protein_pdb_path))

    # ── 2. Parse ligand from SDF block ────────────────────────────────────────
    ligand_mol, ligand_pdb = _sdf_block_to_openmm(
        docked_ligand_sdf_block,
        residue_name=ligand_residue_name,
        chain_id=ligand_chain_id,
        residue_id=ligand_residue_id,
    )

    # ── 3. Build combined topology ─────────────────────────────────────────────
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.positions)

    # ── 4. Parametrize ─────────────────────────────────────────────────────────
    forcefield = ForceField(*protein_forcefield_files)
    _register_ligand_template_from_mol(forcefield, ligand_mol, openff_forcefield)

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
        atom.index for atom in topology_atoms
        if atom.residue.name == ligand_residue_name
    ]
    protein_atom_indices = [
        atom.index for atom in topology_atoms
        if atom.residue.name != ligand_residue_name and atom.element is not None
    ]

    # ── 5. Shell-based protein restraints (no ligand restraints) ──────────────
    restrained, flexible = _partition_protein_atoms_by_shell(
        positions_nm=positions_nm,
        protein_atom_indices=protein_atom_indices,
        ligand_atom_indices=ligand_atom_indices,
        shell_radius_angstrom=shell_radius_angstrom,
    )
    n_restrained = _add_positional_restraints(
        system=system,
        positions_nm=positions_nm,
        atom_indices=restrained,
        k_kcal_per_mol_a2=protein_restraint_k_kcal_per_mol_A2,
        k_param_name="k_protein_posres",
    )

    # ── 6. Build simulation ────────────────────────────────────────────────────
    integrator = LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
        simulation = Simulation(modeller.topology, system, integrator, platform)
    except Exception:
        _log.warning("Platform %s unavailable, falling back to default.", platform_name)
        simulation = Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)

    # ── 7. Record initial state ────────────────────────────────────────────────
    initial_state = simulation.context.getState(getEnergy=True, getPositions=True)
    initial_energy = float(
        initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    initial_lig_pos_nm = np.asarray(
        initial_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )[ligand_atom_indices]

    # ── 8. Minimize ────────────────────────────────────────────────────────────
    simulation.minimizeEnergy(maxIterations=max_iterations)

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = float(
        final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    final_positions_nm = np.asarray(
        final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    final_lig_pos_nm = final_positions_nm[ligand_atom_indices]

    ligand_rmsd = _rmsd(initial_lig_pos_nm * 10.0, final_lig_pos_nm * 10.0)  # nm→Å

    # ── 9. Interaction energy ──────────────────────────────────────────────────
    ie: float | None = None
    if compute_interaction_energy:
        ie = _compute_interaction_energy(
            simulation=simulation,
            ligand_atom_indices=ligand_atom_indices,
            protein_atom_indices=protein_atom_indices,
            final_energy_kj_per_mol=final_energy,
        )

    # ── 10. Write refined complex PDB ─────────────────────────────────────────
    out = io.StringIO()
    PDBFile.writeFile(modeller.topology, final_state.getPositions(), out, keepIds=True)
    refined_pdb = out.getvalue()

    return RefinementResult(
        initial_energy_kj_per_mol=initial_energy,
        final_energy_kj_per_mol=final_energy,
        ligand_rmsd_from_dock_angstrom=ligand_rmsd,
        protein_atoms_flexible=len(flexible),
        protein_atoms_restrained=n_restrained,
        interaction_energy_kj_per_mol=ie,
        refined_complex_pdb=refined_pdb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SDF → OpenMM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sdf_block_to_openmm(
    sdf_block: str,
    *,
    residue_name: str,
    chain_id: str,
    residue_id: int,
) -> tuple[Chem.Mol, PDBFile]:
    """Convert an SDF block to an RDKit Mol and an OpenMM PDBFile.

    Writes a temporary PDB via RDKit so OpenMM can read the topology.
    """
    # Strip the trailing $$$$ if present
    clean = sdf_block.split("$$$$")[0].strip()
    mol = Chem.MolFromMolBlock(clean, removeHs=False, sanitize=True)
    if mol is None:
        # Try without removeHs=False as a fallback
        mol = Chem.MolFromMolBlock(clean, removeHs=True, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse the GNINA docked pose SDF block")

    # Build a PDB string with correct residue/chain labels
    pdb_lines = _mol_to_pdb_block(mol, residue_name=residue_name,
                                   chain_id=chain_id, residue_id=residue_id)
    ligand_pdb = PDBFile(io.StringIO(pdb_lines))
    return mol, ligand_pdb


def _mol_to_pdb_block(
    mol: Chem.Mol,
    *,
    residue_name: str,
    chain_id: str,
    residue_id: int,
) -> str:
    """Write an RDKit molecule to a PDB-format string."""
    conf = mol.GetConformer(0)
    lines: list[str] = []
    for idx, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        atom_name = f"{symbol}{idx % 1000:03d}"[:4]
        lines.append(
            f"HETATM{idx:5d} {atom_name:<4} {residue_name:>3} "
            f"{chain_id:1}{residue_id:4d}    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {symbol:>2}"
        )
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1
        lines.append(f"CONECT{i:5d}{j:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _register_ligand_template_from_mol(
    forcefield: ForceField,
    mol: Chem.Mol,
    openff_forcefield: str,
) -> None:
    """Register OpenFF/GAFF ligand parameters from an RDKit Mol."""
    # Deduplicate by SMILES to avoid re-parametrizing the same compound
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    cache_key = f"{openff_forcefield}::{smiles}"

    if cache_key in _TEMPLATE_GENERATOR_CACHE:
        forcefield.registerTemplateGenerator(
            _TEMPLATE_GENERATOR_CACHE[cache_key].generator
        )
        return

    try:
        from openff.toolkit.topology import Molecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        generator = SMIRNOFFTemplateGenerator(
            molecules=[off_mol],
            forcefield=openff_forcefield,
        )
        _TEMPLATE_GENERATOR_CACHE[cache_key] = generator
        forcefield.registerTemplateGenerator(generator.generator)
        return
    except Exception as exc:
        smirnoff_err = str(exc)

    try:
        from openff.toolkit.topology import Molecule
        from openmmforcefields.generators import GAFFTemplateGenerator

        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        generator = GAFFTemplateGenerator(molecules=[off_mol], forcefield="gaff-2.11")
        _TEMPLATE_GENERATOR_CACHE[f"gaff-2.11::{smiles}"] = generator
        forcefield.registerTemplateGenerator(generator.generator)
        return
    except Exception as exc:
        gaff_err = str(exc)

    raise ImportError(
        f"Could not parametrize ligand with {openff_forcefield!r} "
        f"({smirnoff_err}) or GAFF-2.11 ({gaff_err}). "
        "Ensure openmmforcefields and openff-toolkit are installed."
    )

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
