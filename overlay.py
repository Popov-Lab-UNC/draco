from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from ligand_preparation import (
    LigandColorPoint,
    PreparedLigand,
    PreparedLigandConformer,
)
from pocket_coloring import PocketColoring, FEATURE_COMPAT


@dataclass(frozen=True)
class SphereOverlapResult:
    sphere_id: int
    sphere_labels: tuple[str, ...]
    matched_ligand_label: str
    ligand_point_index: int
    distance: float
    compatibility: int


@dataclass(frozen=True)
class OverlayResult:
    pocket_id: int
    ligand_name: str
    conformer_id: int
    ranking_score: int
    n_compatible: int
    n_neutral: int
    n_incompatible: int
    n_unused: int
    sphere_results: tuple[SphereOverlapResult, ...]
    conformer: PreparedLigandConformer
    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]
    transformed_all_atom_coords: npt.NDArray[np.float64]
    transformed_color_points: tuple[LigandColorPoint, ...]


@dataclass(frozen=True)
class _PocketAlignPoint:
    feature_id: str
    label: str
    coords: npt.NDArray[np.float64]
    sphere_id: int


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def overlay_prepared_ligand(
    prepared_ligand: PreparedLigand,
    pocket_coloring: PocketColoring,
    *,
    feature_cutoff: float = 3.0,
    overlap_cutoff: float = 1.5,
    max_iterations: int = 8,
) -> OverlayResult:
    best_result: OverlayResult | None = None
    for conformer in prepared_ligand.conformers:
        result = overlay_conformer(
            conformer,
            prepared_ligand.name,
            pocket_coloring,
            feature_cutoff=feature_cutoff,
            overlap_cutoff=overlap_cutoff,
            max_iterations=max_iterations,
        )
        if best_result is None or result.ranking_score > best_result.ranking_score:
            best_result = result
    if best_result is None:
        raise ValueError(f"No conformers available for ligand '{prepared_ligand.name}'")
    return best_result


def rank_ligand_over_pockets(
    prepared_ligand: PreparedLigand,
    pocket_colorings: list[PocketColoring],
    *,
    feature_cutoff: float = 3.0,
    overlap_cutoff: float = 2.5,
    max_iterations: int = 8,
) -> list[OverlayResult]:
    results = [
        overlay_prepared_ligand(
            prepared_ligand,
            pocket_coloring,
            feature_cutoff=feature_cutoff,
            overlap_cutoff=overlap_cutoff,
            max_iterations=max_iterations,
        )
        for pocket_coloring in pocket_colorings
    ]
    return sorted(results, key=lambda r: r.ranking_score, reverse=True)


def overlay_conformer(
    conformer: PreparedLigandConformer,
    ligand_name: str,
    pocket_coloring: PocketColoring,
    *,
    feature_cutoff: float = 3.0,
    overlap_cutoff: float = 2.5,
    max_iterations: int = 8,
) -> OverlayResult:
    align_points = _build_align_points(pocket_coloring)
    if not align_points:
        raise ValueError(f"Pocket {pocket_coloring.pocket_id} has no color-labeled spheres")

    seeds = _initial_alignment_seeds(conformer, pocket_coloring)

    BestPayload = tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int, int, int, int, int,
        tuple[SphereOverlapResult, ...],
    ]
    best_payload: BestPayload | None = None

    for seed_rotation, seed_translation in seeds:
        rotation, translation = seed_rotation.copy(), seed_translation.copy()
        for _ in range(max_iterations):
            matches = _match_for_kabsch(
                conformer.color_points, align_points, rotation, translation,
                feature_cutoff=feature_cutoff,
            )
            if not matches:
                break
            source = np.asarray([conformer.color_points[li].coords for li, _ in matches], dtype=np.float64)
            target = np.asarray([align_points[pi].coords for _, pi in matches], dtype=np.float64)
            rotation, translation = _rigid_transform(source, target)

        xpts = _apply_color_points(conformer.color_points, rotation, translation)
        ranking_score, n_comp, n_neut, n_incomp, n_unused, sphere_results = (
            _score_sphere_overlap(pocket_coloring, xpts, overlap_cutoff)
        )
        payload: BestPayload = (
            rotation, translation,
            ranking_score, n_comp, n_neut, n_incomp, n_unused,
            sphere_results,
        )
        if best_payload is None or ranking_score > best_payload[2]:
            best_payload = payload

    if best_payload is None:
        rotation = np.eye(3, dtype=np.float64)
        translation = pocket_coloring.centroid - conformer.centroid
        xpts = _apply_color_points(conformer.color_points, rotation, translation)
        ranking_score, n_comp, n_neut, n_incomp, n_unused, sphere_results = (
            _score_sphere_overlap(pocket_coloring, xpts, overlap_cutoff)
        )
        best_payload = (
            rotation, translation,
            ranking_score, n_comp, n_neut, n_incomp, n_unused,
            sphere_results,
        )

    rotation, translation, ranking_score, n_comp, n_neut, n_incomp, n_unused, sphere_results = best_payload
    transformed_all_atom_coords = _apply_transform(conformer.all_atom_coords, rotation, translation)
    transformed_color_points = tuple(
        LigandColorPoint(
            label=point.label,
            coords=_apply_transform(point.coords[None, :], rotation, translation)[0],
            atom_indices=point.atom_indices,
            radius=point.radius,
        )
        for point in conformer.color_points
    )
    return OverlayResult(
        pocket_id=pocket_coloring.pocket_id,
        ligand_name=ligand_name,
        conformer_id=conformer.conformer_id,
        ranking_score=ranking_score,
        n_compatible=n_comp,
        n_neutral=n_neut,
        n_incompatible=n_incomp,
        n_unused=n_unused,
        sphere_results=sphere_results,
        conformer=conformer,
        rotation=rotation,
        translation=translation,
        transformed_all_atom_coords=transformed_all_atom_coords,
        transformed_color_points=transformed_color_points,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scoring and alignment internals
# ─────────────────────────────────────────────────────────────────────────────

def _score_sphere_overlap(
    pocket_coloring: PocketColoring,
    transformed_color_points: tuple[LigandColorPoint, ...],
    overlap_cutoff: float,
) -> tuple[int, int, int, int, int, tuple[SphereOverlapResult, ...]]:

    if not transformed_color_points:
        n_labeled = sum(1 for s in pocket_coloring.spheres if s.sphere_labels)
        return 0, 0, 0, 0, n_labeled, ()

    lig_coords = np.asarray([p.coords for p in transformed_color_points], dtype=np.float64)
    n_compatible = n_neutral = n_incompatible = n_unused = 0
    sphere_results: list[SphereOverlapResult] = []

    # Track which ligand atoms have already been "claimed" by a sphere to prevent score inflation
    used_lig_indices: set[int] = set()

    for sphere in pocket_coloring.spheres:
        if not sphere.sphere_labels:
            n_unused += 1
            continue

        dists = np.linalg.norm(lig_coords - sphere.center, axis=1)
        nearby_mask = dists <= overlap_cutoff

        if not np.any(nearby_mask):
            n_unused += 1
            continue

        best_compat: int | None = None
        best_lig_label = ""
        best_lig_idx = -1
        best_dist = float("inf")

        for lig_idx in np.where(nearby_mask)[0]:
            if int(lig_idx) in used_lig_indices:
                continue

            point = transformed_color_points[int(lig_idx)]
            dist = float(dists[lig_idx])
            for sphere_label in sphere.sphere_labels:
                compat = FEATURE_COMPAT.get((point.label, sphere_label), 0)
                if best_compat is None or compat > best_compat or (
                    compat == best_compat and dist < best_dist
                ):
                    best_compat = compat
                    best_lig_label = point.label
                    best_lig_idx = int(lig_idx)
                    best_dist = dist

        if best_compat is None:
            n_unused += 1
            continue

        used_lig_indices.add(best_lig_idx)

        sphere_results.append(
            SphereOverlapResult(
                sphere_id=sphere.sphere_id,
                sphere_labels=sphere.sphere_labels,
                matched_ligand_label=best_lig_label,
                ligand_point_index=best_lig_idx,
                distance=best_dist,
                compatibility=best_compat,
            )
        )
        if best_compat > 0:
            n_compatible += 1
        elif best_compat == 0:
            n_neutral += 1
        else:
            n_incompatible += 1

    ranking_score = n_compatible - n_incompatible
    return ranking_score, n_compatible, n_neutral, n_incompatible, n_unused, tuple(sphere_results)


def _build_align_points(pocket_coloring: PocketColoring) -> list[_PocketAlignPoint]:
    points: list[_PocketAlignPoint] = []
    for sphere in pocket_coloring.spheres:
        if not sphere.sphere_labels:
            continue
        for label in sphere.sphere_labels:
            points.append(
                _PocketAlignPoint(
                    feature_id=f"{sphere.sphere_id}:{label}",
                    label=label,
                    coords=sphere.center,
                    sphere_id=sphere.sphere_id,
                )
            )
    return points


def _initial_alignment_seeds(
    conformer: PreparedLigandConformer,
    pocket_coloring: PocketColoring,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    seeds: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    identity = np.eye(3, dtype=np.float64)
    seeds.append((identity, pocket_coloring.centroid - conformer.centroid))

    ligand_by_label: dict[str, list[npt.NDArray[np.float64]]] = defaultdict(list)
    for point in conformer.color_points:
        ligand_by_label[point.label].append(point.coords)

    pocket_by_label: dict[str, list[npt.NDArray[np.float64]]] = defaultdict(list)
    for sphere in pocket_coloring.spheres:
        for label in sphere.sphere_labels:
            pocket_by_label[label].append(sphere.center)

    for ligand_label, ligand_coords in ligand_by_label.items():
        compatible_labels = [
            pocket_label for pocket_label in pocket_by_label
            if FEATURE_COMPAT.get((ligand_label, pocket_label), 0) > 0
        ]
        for pocket_label in compatible_labels:
            seeds.append((
                identity,
                np.mean(pocket_by_label[pocket_label], axis=0) - np.mean(ligand_coords, axis=0),
            ))
    return seeds


def _match_for_kabsch(
    ligand_points: tuple[LigandColorPoint, ...],
    align_points: list[_PocketAlignPoint],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
    *,
    feature_cutoff: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for lig_idx, point in enumerate(ligand_points):
        transformed = _apply_transform(point.coords[None, :], rotation, translation)[0]
        for ap_idx, ap in enumerate(align_points):
            if FEATURE_COMPAT.get((point.label, ap.label), 0) <= 0:
                continue
            dist = float(np.linalg.norm(transformed - ap.coords))
            if dist <= feature_cutoff:
                candidates.append((dist, lig_idx, ap_idx))

    candidates.sort()
    matches: list[tuple[int, int]] = []
    used_lig: set[int] = set()
    used_ap: set[int] = set()
    for dist, lig_idx, ap_idx in candidates:
        if lig_idx in used_lig or ap_idx in used_ap:
            continue
        used_lig.add(lig_idx)
        used_ap.add(ap_idx)
        matches.append((lig_idx, ap_idx))
    return matches


def _apply_color_points(
    color_points: tuple[LigandColorPoint, ...],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> tuple[LigandColorPoint, ...]:
    return tuple(
        LigandColorPoint(
            label=p.label,
            coords=_apply_transform(p.coords[None, :], rotation, translation)[0],
            atom_indices=p.atom_indices,
            radius=p.radius,
        )
        for p in color_points
    )


def _rigid_transform(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    covariance = (source - source_centroid).T @ (target - target_centroid)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - source_centroid @ rotation.T
    return rotation.astype(np.float64), translation.astype(np.float64)


def _apply_transform(
    coords: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(coords, dtype=np.float64) @ rotation.T + translation

