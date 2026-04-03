"""pocket.py – Pocketeer integration: alpha-sphere pockets, GNINA boxes, JSON artifacts.

Provides:

  - ``get_pocket_sphere_centers`` / ``docking_box_from_pocket`` – convert Pocketeer
    pocket objects to a ``DockingBox`` for GNINA.
  - Artifact read/write under ``<output-dir>/pockets/`` so a later run can run
    ``docking`` / ``scoring`` without re-running Pocketeer (same boxes as the
    original ``pocket`` step).

When ``--steps`` includes ``pocket``, each frame's pockets are written to
``pockets/frame{index:04d}.json``.
A later run with ``docking`` / ``scoring`` but *without* ``pocket`` loads these files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from draco.docking import DockingBox

SCHEMA_V1 = "draco.pockets.v1"


def find_pockets_above_threshold(
    protein_pdb_string: str,
    pocket_score_threshold: float,
) -> list[Any]:
    """Parse a protein-only PDB string and return Pocketeer pockets above the score cutoff."""
    import io

    import biotite.structure.io.pdb as _biotite_pdb
    import pocketeer as pt

    frame_atomarray = _biotite_pdb.PDBFile.read(
        io.StringIO(protein_pdb_string)
    ).get_structure(model=1)
    pockets = pt.find_pockets(frame_atomarray)
    return [
        p for p in pockets
        if float(getattr(p, "score", 0.0)) > pocket_score_threshold
    ]


def get_pocket_sphere_centers(pocket) -> npt.NDArray[np.float64]:
    """Extract alpha-sphere center coordinates (Å) from a Pocketeer pocket.

    Pocketeer may expose these via different attribute names depending on
    version; we try a few known names.
    """
    # Pocketeer >= 0.3: pocket.sphere_centers (Å) or pocket.spheres (list of AlphaSphere)
    for attr in ("sphere_centers", "alpha_sphere_centers", "centers", "spheres"):
        val = getattr(pocket, attr, None)
        if val is not None:
            if attr == "spheres" and isinstance(val, list):
                return np.stack([s.center for s in val], axis=0)
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr

    atoms = getattr(pocket, "atoms", None)
    if atoms is not None:
        coords = np.stack([a.coord for a in atoms], axis=0)
        return coords.astype(np.float64)

    raise AttributeError(
        "Cannot extract alpha-sphere centers from pocket object. "
        f"Available attributes: {[a for a in dir(pocket) if not a.startswith('_')]}"
    )


def docking_box_from_pocket(
    pocket,
    *,
    padding_angstrom: float = 5.0,
    min_size_angstrom: float = 15.0,
) -> DockingBox:
    """Derive a GNINA docking box from a Pocketeer pocket object.

    The center is the centroid of all alpha-sphere centers. The size is
    the coordinate range (max - min) along each axis, plus ``padding_angstrom``
    on each side, with a minimum of ``min_size_angstrom`` in each dimension.

    Parameters
    ----------
    pocket:
        A Pocketeer ``Pocket`` object.
    padding_angstrom:
        Extra space added to each side of the bounding box (default 5 Å).
    min_size_angstrom:
        Minimum box dimension in any axis (default 15 Å).

    Returns
    -------
    DockingBox
    """
    centers = get_pocket_sphere_centers(pocket)  # (N, 3) in Å

    centroid = centers.mean(axis=0)
    extent = centers.max(axis=0) - centers.min(axis=0)
    size = np.maximum(extent + 2.0 * padding_angstrom, min_size_angstrom)

    return DockingBox(
        center_x=float(centroid[0]),
        center_y=float(centroid[1]),
        center_z=float(centroid[2]),
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(size[2]),
    )


def pocket_artifact_path(project_output_dir: str | Path, frame_index: int) -> Path:
    """Path to the JSON artifact for one MD frame."""
    root = Path(project_output_dir) / "pockets"
    return root / f"frame{frame_index:04d}.json"


def build_pocket_artifact(
    frame_index: int,
    pockets: list[Any],
) -> dict[str, Any]:
    """Build a JSON-serializable dict from live Pocketeer pocket objects."""
    entries: list[dict[str, Any]] = []
    for idx, pocket in enumerate(pockets):
        pocket_id = int(getattr(pocket, "pocket_id", idx))
        score = float(getattr(pocket, "score", 0.0))
        box = docking_box_from_pocket(pocket)
        centers = get_pocket_sphere_centers(pocket)
        entries.append(
            {
                "pocket_id": pocket_id,
                "score": score,
                "docking_box": {
                    "center_x": box.center_x,
                    "center_y": box.center_y,
                    "center_z": box.center_z,
                    "size_x": box.size_x,
                    "size_y": box.size_y,
                    "size_z": box.size_z,
                },
                "sphere_centers": np.asarray(centers, dtype=np.float64).tolist(),
            }
        )
    return {
        "schema": SCHEMA_V1,
        "frame_index": frame_index,
        "pockets": entries,
    }


def write_pocket_artifact_for_frame(
    project_output_dir: str | Path,
    frame_index: int,
    pockets: list[Any],
) -> Path:
    """Write pocket metadata for one frame (may be an empty ``pockets`` list)."""
    path = pocket_artifact_path(project_output_dir, frame_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_pocket_artifact(frame_index, pockets)
    path.write_text(json.dumps(data, indent=2))
    return path


def _box_from_entry(entry: dict[str, Any]) -> DockingBox:
    db = entry["docking_box"]
    return DockingBox(
        center_x=float(db["center_x"]),
        center_y=float(db["center_y"]),
        center_z=float(db["center_z"]),
        size_x=float(db["size_x"]),
        size_y=float(db["size_y"]),
        size_z=float(db["size_z"]),
    )


def load_pocket_entries(
    project_output_dir: str | Path,
    frame_index: int,
) -> list[tuple[int, float, DockingBox]]:
    """Load (pocket_id, score, DockingBox) tuples from a saved artifact.

    Raises
    ------
    FileNotFoundError
        If the artifact file is missing.
    ValueError
        If the schema is unsupported or data is invalid.
    """
    path = pocket_artifact_path(project_output_dir, frame_index)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pocket artifact: {path}. Run a pipeline with the 'pocket' step "
            f"first (same --output-dir), or place a valid artifact at this path."
        )
    data = json.loads(path.read_text())
    schema = data.get("schema")
    if schema != SCHEMA_V1:
        raise ValueError(
            f"Unsupported pocket artifact schema {schema!r} in {path} (expected {SCHEMA_V1!r})."
        )
    file_frame = data.get("frame_index")
    if file_frame is not None and int(file_frame) != int(frame_index):
        raise ValueError(
            f"frame_index mismatch: artifact {path} has frame_index={file_frame}, "
            f"expected {frame_index}."
        )
    out: list[tuple[int, float, DockingBox]] = []
    for entry in data.get("pockets", []):
        pid = int(entry["pocket_id"])
        score = float(entry.get("score", 0.0))
        out.append((pid, score, _box_from_entry(entry)))
    return out
