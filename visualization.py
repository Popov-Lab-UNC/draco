from __future__ import annotations

import contextlib
from typing import Any

import biotite.structure as struc  # type: ignore

from ligand_preparation import conformer_to_pdb_block
from pocket_coloring import PocketColoring, FEATURE_COLORS
from local_minimization import LocalMinimizationResult
from overlay import OverlayResult


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import py3Dmol  # type: ignore

with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    try:
        from atomworks.io.utils.visualize import view as atomworks_view  # type: ignore
    except (ImportError, ModuleNotFoundError):
        atomworks_view = None


def view_pocket_overlay(
    atomarray: struc.AtomArray,
    pocket_coloring: PocketColoring,
    overlay_result: OverlayResult | None = None,
    *,
    sphere_opacity: float = 0.65,
    sphere_scale: float = 1.0,
    receptor_cartoon: bool = True,
    receptor_surface: bool = False,
    show_color_points: bool = True,
) -> Any:
    if atomworks_view is None:
        raise ImportError(
            "atomworks is required for receptor visualization. Install pocketeer[vis] or atomworks."
        )
    if "py3Dmol" not in globals():
        raise ImportError("py3Dmol is required for ligand overlay visualization.")

    viewer = atomworks_view(
        atomarray, show_cartoon=receptor_cartoon, show_surface=receptor_surface
    )
    if receptor_cartoon:
        viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
    else:
        viewer.setStyle({"model": 0}, {"stick": {"colorscheme": "element"}})

    for sphere in pocket_coloring.spheres:
        x, y, z = sphere.center.tolist()
        viewer.addSphere(
            {
                "center": {"x": x, "y": y, "z": z},
                "radius": float(sphere.radius) * sphere_scale,
                "color": sphere.color,
                "opacity": sphere_opacity,
            }
        )

    if overlay_result is not None:
        ligand_block = conformer_to_pdb_block(
            overlay_result.conformer, overlay_result.transformed_all_atom_coords
        )
        viewer.addModel(ligand_block, "pdb")
        viewer.setStyle(
            {"model": 1, "chain": "L", "resn": "LIG"},
            {"stick": {"colorscheme": "default"}, "sphere": {"scale": 0.22}},
        )
        if show_color_points:
            for point in overlay_result.transformed_color_points:
                x, y, z = point.coords.tolist()
                viewer.addSphere(
                    {
                        "center": {"x": x, "y": y, "z": z},
                        "radius": point.radius * 0.55,
                        "color": FEATURE_COLORS.get(point.label, FEATURE_COLORS["mixed"]),
                        "opacity": 0.35,
                    }
                )

    viewer.zoomTo()
    return viewer


def summarize_overlay(result: OverlayResult) -> dict[str, Any]:
    """Return a flat dict of all overlay scores for a single pose.

    Includes both the coarse integer ``ranking_score`` (useful as a
    filter: keep > 0) and the continuous ``gaussian_fit_score`` (use
    for sorting — higher = tighter, more complementary match).
    """
    return {
        "pocket_id": result.pocket_id,
        "ligand_name": result.ligand_name,
        "conformer_id": result.conformer_id,
        "gaussian_fit_score": round(result.gaussian_fit_score, 4),
        "n_compatible": result.n_compatible,
        "n_neutral": result.n_neutral,
        "n_incompatible": result.n_incompatible,
        "n_unused": result.n_unused,
        "sphere_results": [
            {
                "sphere_id": s.sphere_id,
                "sphere_labels": list(s.sphere_labels),
                "matched_ligand_label": s.matched_ligand_label,
                "ligand_point_index": s.ligand_point_index,
                "distance": round(s.distance, 3),
                "compatibility": s.compatibility,
            }
            for s in result.sphere_results
        ],
    }


def summarize_minimization(
    overlay: OverlayResult,
    minim: LocalMinimizationResult,
) -> dict[str, Any]:
    """Return a flat dict merging overlay scores with post-minimization physics.

    Designed for use in notebooks::

        import pandas as pd
        rows = [summarize_minimization(pose, minim) for pose, minim in results]
        df = pd.DataFrame(rows).sort_values("gaussian_fit_score", ascending=False)

    Interpretation guide
    --------------------
    gaussian_fit_score
        Higher = tighter pharmacophore complementarity.  Primary sort key.
    interaction_energy_kj_per_mol
        Approximate protein–ligand MM interaction energy.  More negative = more
        favourable.  ``None`` if decomposition was skipped or unavailable.
    delta_energy_kj_per_mol
        final − initial potential energy.  Dominated by clash relief; more
        negative = more starting strain resolved by minimization.
    ligand_heavy_atom_rmsd_angstrom
        Ligand drift from the overlay pose after minimization.  Large values
        (> ~2 Å) indicate the overlay geometry was too strained to hold.
    induced_fit_ligand_rmsd_angstrom
        Ligand drift from the minimized pose after 50 ps of free dynamics.
        ``None`` if induced-fit MD was not run.  Large values (> ~2–3 Å)
        suggest the binding mode is not stable.
    """
    summary = summarize_overlay(overlay)
    summary.update(
        {
            "initial_energy_kj_per_mol": round(minim.initial_energy_kj_per_mol, 2),
            "final_energy_kj_per_mol": round(minim.final_energy_kj_per_mol, 2),
            "delta_energy_kj_per_mol": round(
                minim.final_energy_kj_per_mol - minim.initial_energy_kj_per_mol, 2
            ),
            "interaction_energy_kj_per_mol": (
                round(minim.interaction_energy_kj_per_mol, 2)
                if minim.interaction_energy_kj_per_mol is not None
                else None
            ),
            "ligand_heavy_atom_rmsd_angstrom": round(
                minim.ligand_heavy_atom_rmsd_angstrom, 3
            ),
            "induced_fit_ligand_rmsd_angstrom": (
                round(minim.induced_fit_ligand_rmsd_angstrom, 3)
                if minim.induced_fit_ligand_rmsd_angstrom is not None
                else None
            ),
            "protein_atoms_flexible": minim.protein_atoms_flexible,
        }
    )
    # Move sphere_results to the end for readability
    sphere_results = summary.pop("sphere_results")
    summary["sphere_results"] = sphere_results
    return summary

