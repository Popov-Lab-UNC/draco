"""gnina_docking.py – GNINA docking wrapper for the Draco pipeline.

Role in the Pipeline
--------------------
Sits between Pocketeer pocket detection and SAR scoring:

    dynamics.py → pocketeer → gnina_docking → sar_scoring → final_refinement

This module provides:
  1. DockingBox – derived from a Pocketeer pocket's alpha-sphere cloud.
  2. GninaDockResult – parsed output from a single GNINA docking run.
  3. dock_ligand() – run GNINA on a single ligand SDF into a single pocket.
  4. dock_ligands_to_pocket() – dock multiple ligands into the same pocket.

GNINA is called as a subprocess. It must be available on PATH (or provide
the full binary path via ``gnina_binary``).

GNINA output SDF properties parsed:
  - minimizedAffinity  → vina_score (kcal/mol, Vina-like)
  - CNNscore           → cnn_score  (pose quality, 0–1)
  - CNNaffinity        → cnn_affinity (predicted binding affinity, kcal/mol)
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from constants import (
    DEFAULT_DOCKING_PADDING_ANGSTROM,
    DEFAULT_DOCKING_MIN_SIZE_ANGSTROM,
    DEFAULT_GNINA_BINARY,
    DEFAULT_GNINA_TIMEOUT_SECONDS,
)

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DockingBox:
    """Axis-aligned docking search box in Angstroms."""

    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def __repr__(self) -> str:
        return (
            f"DockingBox(center=({self.center_x:.2f}, {self.center_y:.2f}, "
            f"{self.center_z:.2f}), "
            f"size=({self.size_x:.2f}, {self.size_y:.2f}, {self.size_z:.2f}))"
        )


@dataclass(frozen=True)
class GninaDockResult:
    """A single docked pose returned by GNINA.

    Scores are parsed from the SDF property fields written by GNINA.
    """

    ligand_name: str
    """Name of the docked ligand (from the SDF ``> <_Name>`` field)."""

    pose_rank: int
    """1-based rank within the GNINA output (best pose = 1)."""

    vina_score: float
    """AutoDock Vina minimized affinity (kcal/mol). More negative = better."""

    cnn_score: float
    """GNINA CNN pose score (0–1). Higher = more pose-like."""

    cnn_affinity: float
    """GNINA CNN predicted binding affinity (kcal/mol). More negative = better."""

    pose_sdf_block: str
    """The full SDF block for this pose (can be written back to disk)."""


@dataclass
class PocketDockResult:
    """Aggregated docking results for all compounds into a single pocket."""

    pocket_id: int
    docking_box: DockingBox
    results: dict[str, list[GninaDockResult]] = field(default_factory=dict)
    """Maps ligand_name → list of GninaDockResult (ranked best→worst)."""

    def best_score(self, ligand_name: str, score_key: str = "cnn_affinity") -> float:
        """Return the best score for a ligand (most negative = best)."""
        poses = self.results.get(ligand_name)
        if not poses:
            return 0.0
        scores = [getattr(p, score_key) for p in poses]
        return min(scores)  # most negative = best binding


# ─────────────────────────────────────────────────────────────────────────────
# Docking box derivation from Pocketeer pockets
# ─────────────────────────────────────────────────────────────────────────────

def docking_box_from_pocket(
    pocket,
    *,
    padding_angstrom: float = DEFAULT_DOCKING_PADDING_ANGSTROM,
    min_size_angstrom: float = DEFAULT_DOCKING_MIN_SIZE_ANGSTROM,
) -> DockingBox:
    """Derive a GNINA docking box from a Pocketeer pocket object.

    The center is the centroid of all alpha-sphere centers. The size is
    the coordinate range (max - min) along each axis, plus ``padding_angstrom``
    on each side, with a minimum of ``min_size_angstrom`` in each dimension.

    Parameters
    ----------
    pocket:
        A Pocketeer ``Pocket`` object (has ``.atom_coords`` or equivalent).
    padding_angstrom:
        Extra space added to each side of the bounding box (default 4 Å).
    min_size_angstrom:
        Minimum box dimension in any axis (default 15 Å).

    Returns
    -------
    DockingBox
    """
    centers = _get_pocket_sphere_centers(pocket)  # (N, 3) in Å

    centroid = centers.mean(axis=0)  # shape (3,)
    extent = centers.max(axis=0) - centers.min(axis=0)  # shape (3,)
    size = np.maximum(extent + 2.0 * padding_angstrom, min_size_angstrom)

    return DockingBox(
        center_x=float(centroid[0]),
        center_y=float(centroid[1]),
        center_z=float(centroid[2]),
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(size[2]),
    )


def _get_pocket_sphere_centers(pocket) -> npt.NDArray[np.float64]:
    """Extract alpha-sphere center coordinates (Å) from a Pocketeer pocket.

    Pocketeer may expose these via different attribute names depending on
    version; we try a few known names.
    """
    # Pocketeer >= 0.3: pocket.sphere_centers (Å)
    for attr in ("sphere_centers", "alpha_sphere_centers", "centers"):
        val = getattr(pocket, attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr

    # Fallback: pocket.atoms gives the lining atoms; use their coordinates
    # (less accurate than sphere centers but always available)
    atoms = getattr(pocket, "atoms", None)
    if atoms is not None:
        coords = np.stack([a.coord for a in atoms], axis=0)
        return coords.astype(np.float64)

    raise AttributeError(
        "Cannot extract alpha-sphere centers from pocket object. "
        f"Available attributes: {[a for a in dir(pocket) if not a.startswith('_')]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GNINA invocation
# ─────────────────────────────────────────────────────────────────────────────

def dock_ligand(
    protein_pdb_path: str | Path,
    ligand_sdf_path: str | Path,
    box: DockingBox,
    *,
    ligand_name: str | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cnn_scoring: str = "rescore",
    gnina_binary: str = DEFAULT_GNINA_BINARY,
    seed: int = 0,
    cpu: int = 1,
    timeout_seconds: int = DEFAULT_GNINA_TIMEOUT_SECONDS,
) -> list[GninaDockResult]:
    """Dock a single ligand SDF into a pocket using GNINA.

    Parameters
    ----------
    protein_pdb_path:
        Path to the protein PDB file (no ligand, no water).
    ligand_sdf_path:
        Path to the ligand SDF file (one or more conformers).
    box:
        Docking search box (from ``docking_box_from_pocket``).
    ligand_name:
        Label for the ligand (used in result objects). Defaults to the
        SDF filename stem.
    exhaustiveness:
        GNINA exhaustiveness parameter (default 8; higher = slower + better).
    num_modes:
        Number of binding modes to generate (default 9).
    cnn_scoring:
        CNN scoring mode: ``'rescore'`` (default, fastest), ``'refinement'``
        (re-minimises with CNN gradient, more accurate), or ``'none'``
        (Vina only, no GPU required).
    gnina_binary:
        Name or full path of the gnina binary (default ``'gnina'``).
    seed:
        Random seed for reproducibility (default 0).
    cpu:
        Number of CPU threads for GNINA (default 1; GPU does the heavy work).
    timeout_seconds:
        Wall-clock timeout for the GNINA subprocess (default 300 s).

    Returns
    -------
    list[GninaDockResult]
        Docked poses, sorted best→worst by CNN affinity.
        Empty list if GNINA fails or produces no poses.
    """
    protein_pdb_path = Path(protein_pdb_path)
    ligand_sdf_path = Path(ligand_sdf_path)
    name = ligand_name or ligand_sdf_path.stem

    import shlex
    gnina_cmd = shlex.split(gnina_binary)
    _check_gnina(gnina_cmd[0])

    with tempfile.TemporaryDirectory(prefix="draco_gnina_") as tmpdir:
        out_sdf = Path(tmpdir) / "docked.sdf"

        cmd = gnina_cmd + [
            "--receptor", str(protein_pdb_path),
            "--ligand", str(ligand_sdf_path),
            "--center_x", f"{box.center_x:.4f}",
            "--center_y", f"{box.center_y:.4f}",
            "--center_z", f"{box.center_z:.4f}",
            "--size_x", f"{box.size_x:.4f}",
            "--size_y", f"{box.size_y:.4f}",
            "--size_z", f"{box.size_z:.4f}",
            "--out", str(out_sdf),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--cnn_scoring", cnn_scoring,
            "--seed", str(seed),
            "--cpu", str(cpu),
            "--quiet",
        ]

        _log.debug("GNINA cmd: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _log.warning("GNINA timed out after %d s for ligand '%s'", timeout_seconds, name)
            return []
        except FileNotFoundError:
            raise RuntimeError(
                f"GNINA binary not found: '{gnina_binary}'. "
                "Make sure gnina is installed and on PATH."
            )

        if proc.returncode != 0:
            err_str = proc.stderr[:500] if proc.stderr else ""
            _log.warning(
                "GNINA returned exit code %d for ligand '%s'.\nstderr: %s",
                proc.returncode, name, err_str,
            )
            return []

        if not out_sdf.exists() or out_sdf.stat().st_size == 0:
            _log.warning("GNINA produced no output SDF for ligand '%s'", name)
            return []

        return _parse_gnina_sdf(out_sdf.read_text(), ligand_name=name)


def dock_ligands_to_pocket(
    protein_pdb_path: str | Path,
    ligand_sdf_paths: dict[str, Path],
    box: DockingBox,
    pocket_id: int = 0,
    **dock_kwargs: Any,
) -> PocketDockResult:
    """Dock multiple ligands into the same pocket sequentially.

    Parameters
    ----------
    protein_pdb_path:
        Protein PDB file path.
    ligand_sdf_paths:
        Mapping of ``{ligand_name: sdf_path}``.
    box:
        Docking box for this pocket.
    pocket_id:
        Integer identifier for this pocket (for tracking).
    **dock_kwargs:
        Forwarded to ``dock_ligand``.

    Returns
    -------
    PocketDockResult
    """
    result = PocketDockResult(pocket_id=pocket_id, docking_box=box)
    for name, sdf_path in ligand_sdf_paths.items():
        poses = dock_ligand(
            protein_pdb_path,
            sdf_path,
            box,
            ligand_name=name,
            **dock_kwargs,
        )
        result.results[name] = poses
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SDF parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gnina_sdf(sdf_text: str, *, ligand_name: str) -> list[GninaDockResult]:
    """Parse GNINA's output SDF and extract per-pose scores.

    GNINA writes one SDF entry per pose. Each entry has SD properties:
      > <minimizedAffinity>   (Vina score, kcal/mol)
      > <CNNscore>            (CNN pose quality, 0–1)
      > <CNNaffinity>         (CNN affinity, kcal/mol)
    """
    results: list[GninaDockResult] = []
    # Split by the $$$$ record separator
    blocks = [b.strip() for b in sdf_text.split("$$$$") if b.strip()]

    for rank, block in enumerate(blocks, start=1):
        vina = _parse_sdf_property(block, "minimizedAffinity")
        cnn_score = _parse_sdf_property(block, "CNNscore")
        cnn_aff = _parse_sdf_property(block, "CNNaffinity")

        results.append(
            GninaDockResult(
                ligand_name=ligand_name,
                pose_rank=rank,
                vina_score=vina,
                cnn_score=cnn_score,
                cnn_affinity=cnn_aff,
                pose_sdf_block=block + "\n$$$$\n",
            )
        )

    # Sort best → worst by CNN affinity (most negative = best)
    results.sort(key=lambda r: r.cnn_affinity)
    return results


def _parse_sdf_property(block: str, prop_name: str) -> float:
    """Extract a numeric SD property from an SDF record block.

    Looks for lines of the form::

        > <prop_name>
        <float_value>

    Returns 0.0 if the property is missing or cannot be parsed.
    """
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if f"<{prop_name}>" in line and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except ValueError:
                pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _check_gnina(gnina_binary: str) -> None:
    """Raise RuntimeError if the gnina binary is not findable."""
    import shlex
    binary = shlex.split(gnina_binary)[0]
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise RuntimeError(
            f"GNINA binary not found: '{binary}'. "
            "Install gnina (e.g. conda install -c conda-forge gnina) "
            "and make sure it is on PATH."
        )
