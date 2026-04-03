"""refinement.py – Local protein-ligand relaxation for GNINA-docked poses.

Role in the Pipeline
--------------------
    docking → sar_scoring → refinement

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
from rdkit.Chem import AllChem

_log = logging.getLogger(__name__)

# Shared helpers now live in draco.utils (extracted from local_minimization)
from draco.constants import KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2
from draco.utils import (
    _TEMPLATE_GENERATOR_CACHE,
    add_positional_restraints as _add_positional_restraints,
    compute_interaction_energy as _compute_interaction_energy,
    partition_protein_atoms_by_shell as _partition_protein_atoms_by_shell,
    rmsd as _rmsd,
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
    ligand_mol, ligand_top, ligand_pos = _sdf_block_to_openmm_native(
        docked_ligand_sdf_block,
        residue_name=ligand_residue_name,
        chain_id=ligand_chain_id,
        residue_id=ligand_residue_id,
    )

    # ── 3. Build combined topology ─────────────────────────────────────────────
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_top, ligand_pos)

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

def _ensure_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of *mol* with explicit H atoms and 3D coordinates.

    GNINA docked SDF files typically contain only heavy atoms.  OpenFF's
    SMIRNOFFTemplateGenerator requires the OpenMM residue topology to match the
    registered molecule atom-for-atom (including Hs), so we must add Hs before
    building the topology.

    Strategy
    --------
    1. If *mol* already has any explicit H atoms, return it unchanged.
    2. Otherwise add Hs and embed using ETKDGv3 with a coordMap that pins the
       heavy-atom positions, allowing only the new H atoms to be placed.
    3. Fall back to ``Chem.AddHs(addCoords=True)`` (simple geometric placement)
       if the constrained embed fails.
    """
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return mol

    mol_h = Chem.AddHs(mol)

    if mol.GetNumConformers() > 0:
        try:
            from rdkit.Geometry import rdGeometry
            conf = mol.GetConformer(0)
            coord_map = {
                i: rdGeometry.Point3D(
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                )
                for i in range(mol.GetNumAtoms())
            }
            ps = AllChem.ETKDGv3()
            ps.coordMap = coord_map
            ps.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, ps) != -1:
                return mol_h
        except Exception:
            pass

    # Geometric fallback: places Hs at idealized bond angles without moving
    # heavy atoms.
    return Chem.AddHs(mol, addCoords=True)


def _assign_partial_charges_with_fallback(off_mol) -> None:
    """Assign partial charges in-place on an ``openff.toolkit.Molecule``.

    Tries charge methods in order of quality/accuracy:
    1. ``am1bcc``   – semi-empirical, best for drug-like molecules (requires
                      ambertools or an OpenFF-native AM1-BCC implementation)
    2. ``gasteiger`` – empirical, fast, works for virtually any organic molecule

    Silently returns without assigning charges only if every method fails
    (the SMIRNOFF generator will then attempt its own charge assignment at
    ``createSystem()`` time, which may still succeed or raise a descriptive
    error).
    """
    for method in ("am1bcc", "gasteiger"):
        try:
            off_mol.assign_partial_charges(method)
            _log.debug("Assigned partial charges using %r.", method)
            return
        except Exception as exc:
            _log.debug("Charge method %r failed: %s", method, exc)


def _sdf_block_to_openmm_native(
    sdf_block: str,
    *,
    residue_name: str,
    chain_id: str,
    residue_id: int,
):
    """Convert an SDF block directly to an RDKit Mol and native OpenMM Topology/Positions.
    
    Bypasses PDB string conversion to ensure no bonds are lost or misinterpreted
    by PDB format column alignments or CONECT limitations.
    """
    try:
        from openmm.app import Topology, Element
        from openmm import Vec3
    except ImportError:
        from simtk.openmm.app import Topology, Element
        from simtk.openmm import Vec3
        
    import openmm.unit as unit

    # Strip the trailing $$$$ if present
    clean = sdf_block.split("$$$$")[0].strip()
    mol = Chem.MolFromMolBlock(clean, removeHs=False, sanitize=True)
    if mol is None:
        # Try without removeHs=False as a fallback
        mol = Chem.MolFromMolBlock(clean, removeHs=True, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse the GNINA docked pose SDF block")

    # GNINA sometimes writes only polar Hs (e.g. the secondary-amine NH) rather
    # than all Hs. _ensure_explicit_hydrogens short-circuits if it sees *any*
    # H atom, so strip whatever partial Hs GNINA wrote and rebuild the full set.
    # SMIRNOFFTemplateGenerator matches the OpenMM residue topology atom-for-atom
    # against the registered OpenFF molecule, so the topology must have the
    # complete H count.
    mol = _ensure_explicit_hydrogens(Chem.RemoveHs(mol))

    top = Topology()
    chain = top.addChain(id=chain_id)
    res = top.addResidue(residue_name, chain, id=str(residue_id))
    
    atoms = []
    positions = []
    
    conf = mol.GetConformer(0)
    for idx, r_atom in enumerate(mol.GetAtoms(), start=1):
        sym = r_atom.GetSymbol()
        elem = Element.getBySymbol(sym)
        # Construct a 4-character atom name
        atom_name = f"{sym}{idx % 1000:03d}"[:4]
        
        omm_atom = top.addAtom(atom_name, elem, res, id=str(idx))
        atoms.append(omm_atom)
        
        pos = conf.GetAtomPosition(r_atom.GetIdx())
        positions.append(Vec3(pos.x, pos.y, pos.z))
        
    for bond in mol.GetBonds():
        a1 = atoms[bond.GetBeginAtomIdx()]
        a2 = atoms[bond.GetEndAtomIdx()]
        # Pass bond order so SMIRNOFFTemplateGenerator can reconstruct the
        # correct SMILES when matching the residue topology to the molecule.
        # Without this, all bonds default to single-bond and the SMILES lookup
        # fails, producing "Did not recognize residue LIG".
        top.addBond(a1, a2, order=bond.GetBondTypeAsDouble())
        
    return mol, top, positions * unit.angstrom


def _register_ligand_template_from_mol(
    forcefield: ForceField,
    mol: Chem.Mol,
    openff_forcefield: str,
) -> None:
    """Register OpenFF/GAFF ligand parameters from an RDKit Mol.

    Why ``Molecule.from_smiles`` instead of ``Molecule.from_rdkit``
    ---------------------------------------------------------------
    GNINA docked SDF blocks are often tagged as ``2D`` in their mol-file header
    even though the docked coordinates are 3D.  ``openff.toolkit`` warns about
    this and may produce a subtly malformed ``Molecule`` object (wrong
    conformer or incorrect bond perception).  Using the canonical heavy-atom
    SMILES side-steps the issue completely: the SMILES carries only topology
    (atoms + bonds), which is all that ``SMIRNOFFTemplateGenerator._match_residue``
    needs.  That method uses a NetworkX graph-isomorphism check on element and
    connectivity only — no bond orders, no coordinates, no stereochemistry.
    """
    # Canonical heavy-atom SMILES for deduplication and molecule construction.
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    cache_key = f"{openff_forcefield}::{smiles}"

    if cache_key in _TEMPLATE_GENERATOR_CACHE:
        forcefield.registerTemplateGenerator(
            _TEMPLATE_GENERATOR_CACHE[cache_key].generator
        )
        return

    try:
        from openff.toolkit.topology import Molecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        # Build from SMILES to avoid 2D-tag artefacts from GNINA SDF blocks.
        # allow_undefined_stereo=True suppresses errors for molecules where
        # stereochemistry cannot be fully inferred from the docked pose.
        off_mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        # Pre-assign partial charges so createSystem() does not try to run
        # AM1-BCC lazily (AM1-BCC requires ambertools/OpenEye and can fail for
        # halogenated or otherwise complex molecules).  Gasteiger charges are
        # fast and work for any organic molecule; they are accurate enough for
        # the gentle local minimisation performed here.
        if off_mol.partial_charges is None:
            _assign_partial_charges_with_fallback(off_mol)

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

        off_mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        if off_mol.partial_charges is None:
            _assign_partial_charges_with_fallback(off_mol)

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
