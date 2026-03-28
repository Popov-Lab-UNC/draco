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
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rdkit import Chem

_log = logging.getLogger(__name__)

# Reuse the shared helpers from local_minimization to avoid code duplication
from local_minimization import (
    KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2,
    _TEMPLATE_GENERATOR_CACHE,
    _add_positional_restraints,
    _compute_interaction_energy,
    _partition_protein_atoms_by_shell,
    _rmsd,
)

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
    protein_forcefield_files: tuple[str, ...] = ("amber14-all.xml", "amber14/tip3pfb.xml"),
    openff_forcefield: str = "openff-2.3.0",
    shell_radius_angstrom: float = 8.0,
    protein_restraint_k_kcal_per_mol_A2: float = 10.0,
    max_iterations: int = 500,
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
