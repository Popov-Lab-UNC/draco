"""docking.py – GNINA docking wrapper for the Draco pipeline.

Role in the Pipeline
--------------------
Sits between ``draco.pocket`` (Pocketeer) and SAR scoring:

    dynamics.py → pocket.py → docking → sar_scoring → refinement

This module provides:
  1. DockingBox – axis-aligned GNINA search box (often built via ``draco.pocket``).
  2. GninaDockResult – parsed output from a single GNINA docking run.
  3. dock_ligand() – run GNINA on a single ligand SDF into a single pocket.
  4. dock_ligands_to_pocket() – dock multiple ligands into the same pocket.

Pocketeer-specific box construction lives in ``draco.pocket``.

GNINA is called as a subprocess. It must be available on PATH (or provide
the full binary path via ``gnina_binary``).

GNINA output SDF properties parsed:
  - minimizedAffinity  → vina_score (kcal/mol, Vina-like)
  - CNNscore           → cnn_score  (pose quality, 0–1)
  - CNNaffinity        → cnn_affinity (predicted affinity in **pK** units; higher = tighter binding)

CNN affinity vs Vina (discrepancy to avoid)
--------------------------------------------
``CNNaffinity`` is **not** in the same units as ``minimizedAffinity``. The latter
is a Vina-like score in kcal/mol where more negative is better; the former is a
**pK-scale** CNN prediction where **higher is better** (GNINA authors:
https://github.com/gnina/gnina/issues/259). Draco ranks ``cnn_affinity`` with
that convention; do not sort it like an energy score.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
    """GNINA CNN predicted binding affinity in pK units (higher = tighter binding)."""

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
        """Return the best score for a ligand (direction depends on ``score_key``)."""
        poses = self.results.get(ligand_name)
        if not poses:
            return 0.0
        scores = [getattr(p, score_key) for p in poses]
        if score_key in ("cnn_affinity", "cnn_score"):
            return max(scores)
        return min(scores)  # vina_score: more negative = better


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
    gnina_binary: str = "gnina",
    seed: int = 0,
    cpu: int = 1,
    timeout_seconds: int | None = None,
    output_dir: str | Path | None = None,
    write_gnina_logs: bool = True,
) -> list[GninaDockResult]:
    """Dock a single ligand SDF into a pocket using GNINA.

    Parameters
    ----------
    protein_pdb_path:
        Path to the protein PDB file (no ligand, no water).
    ligand_sdf_path:
        Path to the ligand SDF file (one or more conformers).
    box:
        Docking search box (e.g. from ``draco.pocket.docking_box_from_pocket``).
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
        Wall-clock timeout for the GNINA subprocess in seconds, or ``None``
        (default) to wait until GNINA exits with no limit.
    output_dir:
        If provided, GNINA will write its output SDF into this directory and
        (optionally) persist stdout/stderr logs there. If not provided, GNINA
        is run in a temporary directory and outputs are discarded after parsing.
    write_gnina_logs:
        If True (default), write GNINA stdout/stderr to files when ``output_dir``
        is provided.

    Returns
    -------
    list[GninaDockResult]
        Docked poses, sorted best→worst by CNN affinity.
        Empty list if GNINA fails or produces no poses.
    """
    protein_pdb_path = Path(protein_pdb_path).resolve()
    ligand_sdf_path = Path(ligand_sdf_path).resolve()
    name = ligand_name or ligand_sdf_path.stem

    import shlex
    gnina_cmd = shlex.split(gnina_binary)
    _check_gnina(gnina_cmd[0])

    if output_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="draco_gnina_")
        workdir = Path(tmp_ctx.name).resolve()
        cleanup_ctx = tmp_ctx
    else:
        cleanup_ctx = None
        workdir = Path(output_dir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        out_sdf = workdir / f"{name}.gnina.sdf"
        stdout_path = workdir / f"{name}.gnina.stdout.log"
        stderr_path = workdir / f"{name}.gnina.stderr.log"

        # When running through Apptainer, we must bind-mount all directories
        # that GNINA needs to read from or write to.
        if _is_apptainer_cmd(gnina_cmd):
            bind_dirs = _collect_bind_dirs(
                protein_pdb_path, ligand_sdf_path, out_sdf,
            )
            # Insert --bind flags right after the Apptainer subcommand.
            # The typical pattern is: apptainer run [--nv] [--bind ...] image.sif
            gnina_cmd = _inject_apptainer_binds(gnina_cmd, bind_dirs)

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
            assert timeout_seconds is not None  # subprocess only times out if timeout is set
            _log.warning(
                "GNINA timed out after %d s for ligand '%s'", timeout_seconds, name
            )
            return []
        except FileNotFoundError:
            raise RuntimeError(
                f"GNINA binary not found: '{gnina_binary}'. "
                "Make sure gnina is installed and on PATH."
            )

        if output_dir is not None and write_gnina_logs:
            try:
                stdout_path.write_text(proc.stdout or "")
                stderr_path.write_text(proc.stderr or "")
            except OSError as e:
                _log.warning("Failed writing GNINA logs in '%s': %s", str(workdir), str(e))

        if proc.returncode != 0:
            err_str = proc.stderr[:500] if proc.stderr else ""
            out_str = proc.stdout[:500] if proc.stdout else ""
            _log.warning(
                "GNINA returned exit code %d for ligand '%s'.\nstdout: %s\nstderr: %s",
                proc.returncode, name, out_str, err_str,
            )
            return []

        if not out_sdf.exists() or out_sdf.stat().st_size == 0:
            _log.warning(
                "GNINA produced no output SDF for ligand '%s'.\n"
                "  cmd: %s\n  stdout: %s\n  stderr: %s",
                name, " ".join(cmd),
                (proc.stdout or "")[:300], (proc.stderr or "")[:300],
            )
            return []

        return _parse_gnina_sdf(out_sdf.read_text(), ligand_name=name)
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.cleanup()


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
      > <CNNaffinity>         (CNN affinity, pK units; higher = tighter binding)
    """
    results: list[GninaDockResult] = []
    # Split by the $$$$ record separator
    blocks = [b.strip() for b in sdf_text.split("$$$$") if b.strip()]

    for rank, block in enumerate(blocks, start=1):
        vina = _parse_sdf_property(block, "minimizedAffinity", required=True)
        cnn_score = _parse_sdf_property(block, "CNNscore", required=True)
        cnn_aff = _parse_sdf_property(block, "CNNaffinity", required=True)

        # GNINA output can occasionally contain a trailing/truncated SDF record
        # (e.g. interrupted write). Treat such records as invalid instead of
        # assigning 0.0 scores, which would incorrectly dominate ranking.
        if vina is None or cnn_score is None or cnn_aff is None:
            continue

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

    # Sort best → worst by CNN affinity (pK: higher = better)
    results.sort(key=lambda r: r.cnn_affinity, reverse=True)
    return results


def _parse_sdf_property(
    block: str,
    prop_name: str,
    *,
    required: bool = False,
) -> float | None:
    """Extract a numeric SD property from an SDF record block.

    Looks for lines of the form::

        > <prop_name>
        <float_value>

    Returns 0.0 if the property is missing or cannot be parsed.
    If ``required`` is True, returns ``None`` when the property is missing or
    unparsable.
    """
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if f"<{prop_name}>" in line and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except ValueError:
                return None if required else 0.0
    return None if required else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _check_gnina(gnina_binary: str) -> None:
    """Raise RuntimeError if the gnina binary is not findable."""
    import shlex
    binary = shlex.split(gnina_binary)[0]
    # If using Apptainer, check for apptainer instead of gnina
    if binary in ("apptainer", "singularity"):
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"Container runtime '{binary}' not found on PATH. "
                "Load it with 'module load apptainer' or install it."
            )
        return
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise RuntimeError(
            f"GNINA binary not found: '{binary}'. "
            "Install gnina (e.g. conda install -c conda-forge gnina) "
            "and make sure it is on PATH."
        )


def _is_apptainer_cmd(cmd: list[str]) -> bool:
    """Check if the command is an Apptainer/Singularity invocation."""
    if not cmd:
        return False
    base = Path(cmd[0]).name
    return base in ("apptainer", "singularity")


def _collect_bind_dirs(*paths: Path) -> set[str]:
    """Collect unique parent directories that need to be bind-mounted."""
    dirs: set[str] = set()
    for p in paths:
        resolved = p.resolve()
        parent = str(resolved.parent)
        dirs.add(parent)
    return dirs


def _inject_apptainer_binds(
    cmd: list[str],
    bind_dirs: set[str],
) -> list[str]:
    """Insert --bind flags into an Apptainer command.

    The command is expected to look like:
        apptainer run [--nv] [existing-flags...] image.sif
    We insert --bind flags before the .sif image path.
    """
    # Find the .sif file position
    sif_idx = None
    for i, tok in enumerate(cmd):
        if tok.endswith(".sif"):
            sif_idx = i
            break

    if sif_idx is None:
        # Can't find .sif — just prepend bind flags after cmd[1] (the subcommand)
        insert_at = 2 if len(cmd) > 1 else 1
    else:
        insert_at = sif_idx

    bind_args: list[str] = []
    for d in sorted(bind_dirs):
        bind_args.extend(["--bind", d])

    return cmd[:insert_at] + bind_args + cmd[insert_at:]
