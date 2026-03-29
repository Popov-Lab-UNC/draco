"""utils.py – Shared helpers used by multiple Draco pipeline stages.

Extracted from local_minimization.py so that final_refinement.py and
other modules can import them without depending on the deprecated
overlay-based local_minimization module.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from draco.constants import KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2

try:
    from openmm import CustomExternalForce, unit
except ImportError:  # pragma: no cover
    from simtk.openmm import CustomExternalForce, unit  # type: ignore


# Cache for OpenMM template generators to avoid repeating
# AM1-BCC / SMIRNOFF parameterization on the same ligand.
_TEMPLATE_GENERATOR_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def rmsd(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> float:
    """RMSD between two coordinate arrays of shape (N, 3)."""
    if a.shape != b.shape:
        raise ValueError(f"Cannot compute RMSD for shapes {a.shape} and {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# ─────────────────────────────────────────────────────────────────────────────
# Protein shell partitioning
# ─────────────────────────────────────────────────────────────────────────────

def partition_protein_atoms_by_shell(
    *,
    positions_nm: npt.NDArray[np.float64],
    protein_atom_indices: list[int],
    ligand_atom_indices: list[int],
    shell_radius_angstrom: float,
) -> tuple[list[int], list[int]]:
    """Partition protein atoms into (restrained, flexible) based on distance to ligand.

    Protein atoms within ``shell_radius_angstrom`` of any ligand atom are
    considered "flexible" (no restraints); all others are "restrained".

    Returns
    -------
    tuple[list[int], list[int]]
        ``(restrained_indices, flexible_indices)``
    """
    protein_coords = positions_nm[np.asarray(protein_atom_indices, dtype=int)]
    ligand_coords = positions_nm[np.asarray(ligand_atom_indices, dtype=int)]
    shell_nm = shell_radius_angstrom / 10.0

    deltas = protein_coords[:, None, :] - ligand_coords[None, :, :]
    min_distances_nm = np.linalg.norm(deltas, axis=2).min(axis=1)

    flexible_mask = min_distances_nm <= shell_nm
    flexible = [protein_atom_indices[i] for i in np.where(flexible_mask)[0]]
    restrained = [protein_atom_indices[i] for i in np.where(~flexible_mask)[0]]
    return restrained, flexible


# ─────────────────────────────────────────────────────────────────────────────
# Positional restraints
# ─────────────────────────────────────────────────────────────────────────────

def add_positional_restraints(
    *,
    system: object,
    positions_nm: npt.NDArray[np.float64],
    atom_indices: list[int],
    k_kcal_per_mol_a2: float,
    k_param_name: str,
) -> int:
    """Add harmonic positional restraints to specified atoms.

    Returns the number of particles restrained (0 if k <= 0 or empty list).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Interaction energy decomposition
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction_energy(
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

    Returns
    -------
    float or None
        Interaction energy in kJ/mol, or ``None`` if the NonbondedForce
        cannot be found.
    """
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
            nb_force.setParticleParameters(
                idx, 0.0 * params[0].unit, params[1], 0.0 * params[2].unit,
            )
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
            nb_force.setParticleParameters(
                idx, 0.0 * params[0].unit, params[1], 0.0 * params[2].unit,
            )
        nb_force.updateParametersInContext(simulation.context)  # type: ignore[attr-defined]

        state = simulation.context.getState(getEnergy=True)  # type: ignore[attr-defined]
        e_ligand = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        # ── Restore protein params ───────────────────────────────────────────
        for i, idx in enumerate(protein_atom_indices):
            nb_force.setParticleParameters(idx, *prot_saved[i])
        nb_force.updateParametersInContext(simulation.context)  # type: ignore[attr-defined]

        return final_energy_kj_per_mol - e_protein - e_ligand

    except Exception:
        # Restore everything on failure
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
