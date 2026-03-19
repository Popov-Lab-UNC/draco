from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from ligand_preparation import conformer_to_pdb_block
from overlay import OverlayResult
from protein_preparation import PreparedProtein

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
) -> LocalMinimizationResult:
    """Minimise one overlay pose against a pre-prepared protein.

    The caller is responsible for running ``prepare_protein()`` once and
    passing the resulting ``PreparedProtein`` here.  This avoids repeating
    the (slow) PDBFixer repair step for every pose.
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
        )
        n_ligand_restrained = _add_positional_restraints(
            system=system,
            positions_nm=positions_nm,
            atom_indices=ligand_atom_indices,
            k_kcal_per_mol_a2=ligand_restraint_k_kcal_per_mol_A2,
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
        )
    except Exception:
        if output_path and write_pdb_on_error:
            out_path = Path(output_path)
            error_path = out_path.with_name(out_path.stem + error_suffix + out_path.suffix)
            error_path.write_text(initial_complex_pdb)
        raise


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
    ligand_mol = Chem.MolFromMolBlock(overlay_result.conformer.mol_block, removeHs=False)
    if ligand_mol is None:
        raise ValueError("Failed to build RDKit molecule from ligand mol_block")

    conf = ligand_mol.GetConformer()
    for idx, xyz in enumerate(overlay_result.transformed_all_atom_coords):
        conf.SetAtomPosition(int(idx), tuple(float(v) for v in xyz))

    try:
        from openff.toolkit.topology import Molecule  # type: ignore
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        off_mol = Molecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)
        generator = SMIRNOFFTemplateGenerator(
            molecules=[off_mol],
            forcefield=openff_forcefield,
        )
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
) -> int:
    if k_kcal_per_mol_a2 <= 0.0 or not atom_indices:
        return 0

    force = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", k_kcal_per_mol_a2 * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2)
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

