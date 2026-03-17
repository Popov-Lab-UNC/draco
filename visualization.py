from __future__ import annotations

import contextlib
from typing import Any

import biotite.structure as struc  # type: ignore

from ligand_preparation import conformer_to_pdb_block
from pocket_coloring import PocketColoring, FEATURE_COLORS
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
    return {
        "pocket_id": result.pocket_id,
        "ligand_name": result.ligand_name,
        "conformer_id": result.conformer_id,
        "ranking_score": result.ranking_score,
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
                "distance": s.distance,
                "compatibility": s.compatibility,
            }
            for s in result.sphere_results
        ],
    }

