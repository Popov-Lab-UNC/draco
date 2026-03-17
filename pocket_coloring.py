from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import biotite.structure as struc  # type: ignore
import numpy as np
import numpy.typing as npt


# ─────────────────────────────────────────────────────────────────────────────
# Residue categories
# ─────────────────────────────────────────────────────────────────────────────

AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "PHE", "TYR", "TRP"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature display colors
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLORS: dict[str, str] = {
    "donor": "#2f5eff",
    "acceptor": "#ff4d4d",
    "hydrophobe": "#d4a017",
    "ring": "#7a3cff",
    "cation": "#12b5cb",
    "anion": "#f542d4",
    "mixed": "#8a8a8a",
    "unlabeled": "#cccccc",
}


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility matrix 
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COMPAT: dict[tuple[str, str], int] = {
    ("donor",    "acceptor"): 1,
    ("acceptor", "donor"):    1,
    ("donor",    "anion"):    1,   
    ("acceptor", "cation"):   1,   
    ("anion",    "donor"):    1,   
    ("cation",   "acceptor"): 1,   
    ("donor",    "donor"):    -1,
    ("acceptor", "acceptor"): -1,
    ("anion",    "acceptor"): -1,  
    ("hydrophobe", "hydrophobe"): 1,
    ("hydrophobe", "ring"):       1,   
    ("ring",       "hydrophobe"): 1,   
    ("ring",       "ring"):       1,   
    ("ring",       "cation"):     1,   
    ("cation",     "ring"):       1,   
    ("cation", "anion"): 1,
    ("anion",  "cation"): 1,
    ("cation", "cation"): -1,
    ("anion",  "anion"):  -1,
    ("donor",      "hydrophobe"): -1,
    ("acceptor",   "hydrophobe"): -1,
    ("cation",     "hydrophobe"): -1,
    ("anion",      "hydrophobe"): -1,
    ("hydrophobe", "donor"):    -1,
    ("hydrophobe", "acceptor"): -1,
    ("hydrophobe", "cation"):   -1,
    ("hydrophobe", "anion"):    -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SphereFeatureRecord:
    sphere_id: int
    pocket_id: int
    center: npt.NDArray[np.float64]
    radius: float
    mean_sasa: float
    source_atom_indices: tuple[int, ...]
    residues: tuple[tuple[str, int, str], ...]
    sphere_labels: tuple[str, ...]   
    dominant_label: str
    label_scores: dict[str, float]
    feature_anchors: dict[str, tuple[float, float, float]]
    feature_directions: dict[str, tuple[float, float, float]]
    color: str


@dataclass(frozen=True)
class PocketColoring:
    pocket_id: int
    centroid: npt.NDArray[np.float64]
    spheres: tuple[SphereFeatureRecord, ...]

    @property
    def coords(self) -> npt.NDArray[np.float64]:
        return np.asarray([sphere.center for sphere in self.spheres], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def color_pocket(
    atomarray: struc.AtomArray,
    pocket: Any,
    *,
    shell_radius: float = 4.0,
    mixed_tolerance: float = 0.15,
) -> PocketColoring:
    if not isinstance(atomarray, struc.AtomArray):
        raise TypeError(f"Expected AtomArray, got {type(atomarray).__name__}")

    sphere_records = tuple(
        _color_single_sphere(
            atomarray,
            pocket_id=int(pocket.pocket_id),
            sphere=sphere,
            shell_radius=shell_radius,
            mixed_tolerance=mixed_tolerance,
        )
        for sphere in pocket.spheres
    )
    return PocketColoring(
        pocket_id=int(pocket.pocket_id),
        centroid=np.asarray(pocket.centroid, dtype=np.float64),
        spheres=sphere_records,
    )


def color_pockets(
    atomarray: struc.AtomArray,
    pockets: list[Any],
    *,
    shell_radius: float = 4.0,
    mixed_tolerance: float = 0.15,
) -> list[PocketColoring]:
    return [
        color_pocket(atomarray, pocket, shell_radius=shell_radius, mixed_tolerance=mixed_tolerance)
        for pocket in pockets
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Sphere coloring
# ─────────────────────────────────────────────────────────────────────────────

def _color_single_sphere(
    atomarray: struc.AtomArray,
    *,
    pocket_id: int,
    sphere: Any,
    shell_radius: float,
    mixed_tolerance: float,
) -> SphereFeatureRecord:
    coords = np.asarray(atomarray.coord, dtype=np.float64)
    center = np.asarray(sphere.center, dtype=np.float64)
    distances = np.linalg.norm(coords - center, axis=1)
    nearby_indices = np.where(distances <= shell_radius)[0]

    label_scores: dict[str, float] = defaultdict(float)
    anchor_sums: dict[str, npt.NDArray[np.float64]] = {}
    direction_sums: dict[str, npt.NDArray[np.float64]] = {}
    residue_contacts: set[tuple[str, int, str]] = set()

    for atom_idx in nearby_indices:
        residue_contacts.add(_residue_tuple(atomarray, int(atom_idx)))
        distance = max(float(distances[atom_idx]), 0.1)
        weight = math.exp(-((distance / shell_radius) ** 2))
        atom_coord = np.asarray(atomarray.coord[int(atom_idx)], dtype=np.float64)
        direction = _safe_normalize(center - atom_coord)
        for label in _labels_for_atom(atomarray, int(atom_idx)):
            label_scores[label] += weight
            anchor_sums[label] = (
                anchor_sums.get(label, np.zeros(3, dtype=np.float64)) + atom_coord * weight
            )
            direction_sums[label] = (
                direction_sums.get(label, np.zeros(3, dtype=np.float64)) + direction * weight
            )

    if not label_scores:
        sphere_labels: tuple[str, ...] = ()
        dominant_label = "unlabeled"
    else:
        ordered = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ordered[0][1]
        sphere_labels = tuple(
            label for label, score in ordered
            if score >= top_score * 0.35 and score > 0.05
        )
        if len(sphere_labels) > 1:
            dominant_label = "mixed"  # Kept as mixed for visualization purposes
        else:
            dominant_label = sphere_labels[0]

    feature_anchors: dict[str, tuple[float, float, float]] = {}
    feature_directions: dict[str, tuple[float, float, float]] = {}
    for label, score in label_scores.items():
        if score <= 0.0:
            continue
        anchor = anchor_sums[label] / score
        direction = _safe_normalize(direction_sums[label])
        feature_anchors[label] = tuple(float(v) for v in anchor)  # type: ignore[assignment]
        feature_directions[label] = tuple(float(v) for v in direction)  # type: ignore[assignment]

    return SphereFeatureRecord(
        sphere_id=int(sphere.sphere_id),
        pocket_id=pocket_id,
        center=center,
        radius=float(sphere.radius),
        mean_sasa=float(getattr(sphere, "mean_sasa", 0.0)),
        source_atom_indices=tuple(int(idx) for idx in getattr(sphere, "atom_indices", [])),
        residues=tuple(sorted(residue_contacts)),
        sphere_labels=sphere_labels,
        dominant_label=dominant_label,
        label_scores=dict(label_scores),
        feature_anchors=feature_anchors,
        feature_directions=feature_directions,
        color=FEATURE_COLORS.get(dominant_label, FEATURE_COLORS["unlabeled"]),
    )


def _labels_for_atom(atomarray: struc.AtomArray, atom_idx: int) -> set[str]:
    """Labels describing what the PROTEIN has at this atom's location."""
    res_name = _atomarray_value(atomarray, "res_name", atom_idx).upper()
    atom_name = _atomarray_value(atomarray, "atom_name", atom_idx).upper()
    element = _atomarray_value(atomarray, "element", atom_idx).upper()
    labels: set[str] = set()

    # --- 1. Base H-Bond Rules ---
    if element == "O":
        labels.add("acceptor")
    if element == "N":
        # Proline backbone N lacks a hydrogen (cannot donate)
        if not (res_name == "PRO" and atom_name == "N"):
            labels.add("donor")

    # --- 2. Metal Coordination ---
    if element in {"ZN", "MG", "CA", "FE", "MN", "CU", "NI", "CO"}:
        labels.update({"cation", "acceptor"}) 

    # --- 3. Charges (Anions & Cations) ---
    if res_name in NEGATIVE_RESIDUES and atom_name.startswith(("OD", "OE")):
        labels.add("anion")
    if atom_name == "OXT":  # C-terminal oxygen is negatively charged
        labels.add("anion")
        
    if res_name == "LYS" and atom_name == "NZ":
        labels.add("cation")
    if res_name == "ARG" and atom_name in {"NE", "NH1", "NH2"}:
        labels.add("cation")

    # --- 4. Dual-Polarity Sidechains ---
    if res_name in {"SER", "THR"} and atom_name.startswith("OG"):
        labels.add("donor") 
    if res_name == "TYR" and atom_name == "OH":
        labels.add("donor") 
    if res_name == "CYS" and atom_name == "SG":
        labels.update({"donor", "acceptor"}) 
    if res_name == "HIS" and atom_name in {"ND1", "NE2"}:
        labels.add("acceptor") 

    # --- 5. Hydrophobes ---
    if res_name in HYDROPHOBIC_RESIDUES and element in {"C", "S"}:
        if atom_name not in {"C", "CA", "O", "N"}:
            labels.add("hydrophobe")

    # --- 6. Aromatics ---
    if res_name in AROMATIC_RESIDUES:
        if atom_name not in {"N", "CA", "C", "O", "CB", "OH"}:
            labels.add("ring")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Overlay scoring
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


# ─────────────────────────────────────────────────────────────────────────────
# Alignment internals
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_normalize(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


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


def _atomarray_value(atomarray: struc.AtomArray, attr: str, atom_idx: int) -> str:
    if not hasattr(atomarray, attr):
        return ""
    return str(getattr(atomarray, attr)[atom_idx])


def _residue_tuple(atomarray: struc.AtomArray, atom_idx: int) -> tuple[str, int, str]:
    chain_id = _atomarray_value(atomarray, "chain_id", atom_idx)
    res_id = int(_atomarray_value(atomarray, "res_id", atom_idx) or 0)
    res_name = _atomarray_value(atomarray, "res_name", atom_idx)
    return (chain_id, res_id, res_name)