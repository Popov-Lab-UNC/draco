"""docking.py – GNINA / AutoDock-GPU docking for the Draco pipeline.

Role in the Pipeline
--------------------
Sits between ``draco.pocket`` (Pocketeer) and SAR scoring:

    dynamics.py → pocket.py → docking → sar_scoring → refinement

This module provides:
  1. DockingBox – axis-aligned GNINA search box (often built via ``draco.pocket``).
  2. GninaDockResult – parsed output from a single GNINA docking or rescoring run.
  3. dock_ligand() – run GNINA on a single ligand SDF into a single pocket.
  4. dock_ligands_to_pocket() – dock multiple ligands into the same pocket (GNINA).
  5. dock_ligands_to_pocket_adgpu_gnina() – AutoDock-GPU pose search, then GNINA
     ``--score_only`` rescoring (same ``GninaDockResult`` objects downstream).

Pocketeer-specific box construction lives in ``draco.pocket``.

GNINA is called as a subprocess. It must be available on PATH (or provide
the full binary path via ``gnina_binary``).

The ``adgpu_gnina`` path additionally requires Meeko (``mk_prepare_receptor.py``,
``mk_prepare_ligand.py``), ``autogrid4``, and ``adgpu`` on ``PATH`` (typical HPC
``module load autodock-gpu autogrid`` layout).

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
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

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

    cnn_vs: float
    """GNINA CNN virtual screening score (probability, 0-1). Higher = better."""

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
# AutoDock-GPU docking + GNINA rescoring (single visible GPU per worker)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_exe(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    p = Path(name)
    if p.is_file() and os.access(p, os.X_OK):
        return str(p.resolve())
    return name


def _prepare_adgpu_grid(
    protein_pdb_path: Path,
    box: DockingBox,
    grid_dir: Path,
    *,
    mk_prepare_receptor: str,
    autogrid_binary: str,
    delete_bad_res: bool,
    timeout_seconds: int | None,
    grid_basename: str = "draco_rec",
) -> Path:
    """Run Meeko receptor prep + AutoGrid4; return path to ``*.maps.fld``."""
    grid_dir.mkdir(parents=True, exist_ok=True)
    rec_pdb = grid_dir / "receptor_in.pdb"
    shutil.copy2(protein_pdb_path, rec_pdb)

    mk_bin = _resolve_exe(mk_prepare_receptor)
    cmd = [
        mk_bin,
        "--read_pdb",
        rec_pdb.name,
        "-o",
        grid_basename,
        "-p",
        "-g",
        "--box_center",
        f"{box.center_x:.4f}",
        f"{box.center_y:.4f}",
        f"{box.center_z:.4f}",
        "--box_size",
        f"{box.size_x:.4f}",
        f"{box.size_y:.4f}",
        f"{box.size_z:.4f}",
    ]
    if delete_bad_res:
        cmd.append("--delete_bad_res")

    _log.debug("mk_prepare_receptor cmd: %s (cwd=%s)", " ".join(cmd), grid_dir)
    proc = subprocess.run(
        cmd,
        cwd=str(grid_dir),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    mk_stdout_log = grid_dir / f"{grid_basename}.mk_prepare_receptor.stdout.log"
    mk_stderr_log = grid_dir / f"{grid_basename}.mk_prepare_receptor.stderr.log"
    try:
        mk_stdout_log.write_text(proc.stdout or "")
        mk_stderr_log.write_text(proc.stderr or "")
    except OSError:
        pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"mk_prepare_receptor failed (exit {proc.returncode}):\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout_log: {mk_stdout_log}\nstderr_log: {mk_stderr_log}\n"
            f"stdout_tail: {(proc.stdout or '')[-4000:]}\n"
            f"stderr_tail: {(proc.stderr or '')[-4000:]}"
        )

    gpf = grid_dir / f"{grid_basename}.gpf"
    if not gpf.is_file():
        raise RuntimeError(f"mk_prepare_receptor did not write expected GPF: {gpf}")

    ag_bin = _resolve_exe(autogrid_binary)
    glg = grid_dir / f"{grid_basename}.glg"
    ag_cmd = [ag_bin, "-p", gpf.name, "-l", glg.name]
    _log.debug("autogrid4 cmd: %s (cwd=%s)", " ".join(ag_cmd), grid_dir)
    ag_proc = subprocess.run(
        ag_cmd,
        cwd=str(grid_dir),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    ag_stdout_log = grid_dir / f"{grid_basename}.autogrid.stdout.log"
    ag_stderr_log = grid_dir / f"{grid_basename}.autogrid.stderr.log"
    try:
        ag_stdout_log.write_text(ag_proc.stdout or "")
        ag_stderr_log.write_text(ag_proc.stderr or "")
    except OSError:
        pass
    if ag_proc.returncode != 0:
        raise RuntimeError(
            f"autogrid4 failed (exit {ag_proc.returncode}):\n"
            f"command: {' '.join(ag_cmd)}\n"
            f"stdout_log: {ag_stdout_log}\nstderr_log: {ag_stderr_log}\n"
            f"stdout_tail: {(ag_proc.stdout or '')[-4000:]}\n"
            f"stderr_tail: {(ag_proc.stderr or '')[-4000:]}"
        )

    fld = grid_dir / f"{grid_basename}.maps.fld"
    if not fld.is_file() or fld.stat().st_size == 0:
        raise RuntimeError(f"AutoGrid4 did not produce maps file: {fld}")
    return fld.resolve()


def _ligand_sdf_to_pdbqt_tasks(
    ligand_sdf: Path,
    lig_work: Path,
    *,
    mk_prepare_ligand: str,
    base_stem: str,
    timeout_seconds: int | None,
) -> list[tuple[Path, str]]:
    """Convert each SDF entry to a PDBQT; return (pdbqt_path, unique_stem) pairs."""
    lig_work.mkdir(parents=True, exist_ok=True)
    mk_bin = _resolve_exe(mk_prepare_ligand)
    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False, sanitize=False)
    tasks: list[tuple[Path, str]] = []
    idx = 0
    for mol in supplier:
        if mol is None:
            continue
        stem = f"{base_stem}_c{idx}"
        idx += 1
        conf_sdf = lig_work / f"{stem}.sdf"
        w = Chem.SDWriter(str(conf_sdf))
        w.write(mol)
        w.close()

        out_pdbqt = lig_work / f"{stem}.pdbqt"
        cmd = [mk_bin, "-i", conf_sdf.name, "-o", out_pdbqt.name]
        proc = subprocess.run(
            cmd,
            cwd=str(lig_work),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0 or not out_pdbqt.is_file():
            _log.warning(
                "mk_prepare_ligand failed for '%s' entry %s: exit=%s stderr=%s",
                ligand_sdf.name,
                stem,
                proc.returncode,
                (proc.stderr or "")[:400],
            )
            continue
        tasks.append((out_pdbqt.resolve(), stem))
    return tasks


def _run_adgpu_filelist(
    workdir: Path,
    maps_fld: Path,
    pdbqt_stems: list[tuple[Path, str]],
    *,
    adgpu_binary: str,
    nrun: int,
    timeout_seconds: int | None,
) -> None:
    """Run AutoDock-GPU with a ``filelist.txt`` batch (see AutoDockGPUOracle)."""
    filelist = workdir / "filelist.txt"
    lig_dir = workdir / "ligands"
    lig_dir.mkdir(exist_ok=True)

    lines = [maps_fld.name]
    for pdbqt_path, stem in pdbqt_stems:
        dest = lig_dir / pdbqt_path.name
        if pdbqt_path.resolve() != dest.resolve():
            shutil.copy2(pdbqt_path, dest)
        rel = f"ligands/{dest.name}"
        lines.append(rel)
        lines.append(stem)

    filelist.write_text("\n".join(lines) + "\n")

    # Symlink / copy map siblings into workdir (same pattern as benchmark).
    receptor_dir = maps_fld.parent
    allowed = {".map", ".fld", ".xyz", ".pdbqt"}
    for ref_file in receptor_dir.iterdir():
        if not ref_file.is_file():
            continue
        if ref_file.suffix in allowed or ref_file.name == maps_fld.name:
            target = workdir / ref_file.name
            if not target.exists():
                try:
                    os.symlink(ref_file.resolve(), target)
                except OSError:
                    shutil.copy2(ref_file, target)

    adgpu = _resolve_exe(adgpu_binary)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OMP_PROC_BIND", "true")
    cmd = [
        adgpu,
        "--filelist",
        "filelist.txt",
        "--nrun",
        str(nrun),
        "--xmloutput",
        "0",
        "--dlgoutput",
        "1",
    ]
    _log.debug("AutoDock-GPU cmd: %s (cwd=%s)", " ".join(cmd), workdir)
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    if proc.returncode != 0:
        _log.warning(
            "adgpu returned %s for workdir %s\nstdout tail: %s\nstderr tail: %s",
            proc.returncode,
            workdir,
            (proc.stdout or "")[-400:],
            (proc.stderr or "")[-400:],
        )


def _unroll_multiconf_rdkit_mols(rdkit_mols: list[Chem.Mol]) -> list[Chem.Mol]:
    if len(rdkit_mols) == 1 and rdkit_mols[0].GetNumConformers() > 1:
        base = rdkit_mols[0]
        out: list[Chem.Mol] = []
        for conf in base.GetConformers():
            new_mol = Chem.Mol(base)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(conf, assignId=True)
            out.append(new_mol)
        return out
    return rdkit_mols


def _collect_adgpu_poses_from_workdir(
    workdir: Path,
    pdbqt_stems: list[tuple[Path, str]],
    *,
    max_poses: int,
) -> list[tuple[Chem.Mol, float]]:
    """Parse DLGs from AutoDock-GPU; return (mol, AD_free_energy) sorted best-first."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'meeko' package is required for AutoDock-GPU DLG parsing in adgpu_gnina mode. "
            "Install it in your Draco environment (e.g. `pixi install` / conda-forge meeko)."
        ) from exc

    scored: list[tuple[Chem.Mol, float]] = []
    for _pq, stem in pdbqt_stems:
        dlg = workdir / f"{stem}.dlg"
        if not dlg.is_file():
            continue
        try:
            pdbqt_mol = PDBQTMolecule.from_file(str(dlg), is_dlg=True, skip_typing=True)
            rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        except Exception as exc:
            _log.warning("Failed parsing DLG %s: %s", dlg, exc)
            continue
        if not rdkit_mols:
            continue
        rdkit_mols = _unroll_multiconf_rdkit_mols(rdkit_mols)
        energies = getattr(pdbqt_mol, "_pose_data", {}).get("free_energies", [])
        for idx, mol in enumerate(rdkit_mols):
            e = energies[idx] if idx < len(energies) else float("nan")
            if mol is None or not math.isfinite(e):
                continue
            scored.append((mol, float(e)))

    scored.sort(key=lambda t: t[1])
    return scored[:max_poses]


def _write_mols_sdf(mols: list[Chem.Mol], path: Path, *, base_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(path))
    for i, mol in enumerate(mols):
        m = Chem.Mol(mol)
        title = f"{base_name}_pose{i}"
        m.SetProp("_Name", title)
        w.write(m)
    w.close()


def dock_ligand_gnina_score_only(
    protein_pdb_path: str | Path,
    ligand_sdf_path: str | Path,
    box: DockingBox,
    *,
    ligand_name: str | None = None,
    cnn_scoring: str = "rescore",
    gnina_binary: str = "gnina",
    seed: int = 0,
    cpu: int = 1,
    num_modes: int = 9,
    timeout_seconds: int | None = None,
    output_dir: str | Path | None = None,
    write_gnina_logs: bool = True,
) -> list[GninaDockResult]:
    """GNINA CNN/Vina rescoring of existing poses (``--score_only``)."""
    protein_pdb_path = Path(protein_pdb_path).resolve()
    ligand_sdf_path = Path(ligand_sdf_path).resolve()
    name = ligand_name or ligand_sdf_path.stem

    import shlex

    gnina_cmd = shlex.split(gnina_binary)
    _check_gnina(gnina_cmd[0])

    if output_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="draco_gnina_score_")
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

        if _is_apptainer_cmd(gnina_cmd):
            bind_dirs = _collect_bind_dirs(protein_pdb_path, ligand_sdf_path, out_sdf)
            gnina_cmd = _inject_apptainer_binds(gnina_cmd, bind_dirs)

        cmd = gnina_cmd + [
            "--receptor",
            str(protein_pdb_path),
            "--ligand",
            str(ligand_sdf_path),
            "--center_x",
            f"{box.center_x:.4f}",
            "--center_y",
            f"{box.center_y:.4f}",
            "--center_z",
            f"{box.center_z:.4f}",
            "--size_x",
            f"{box.size_x:.4f}",
            "--size_y",
            f"{box.size_y:.4f}",
            "--size_z",
            f"{box.size_z:.4f}",
            "--out",
            str(out_sdf),
            "--score_only",
            "--num_modes",
            str(max(1, num_modes)),
            "--cnn_scoring",
            cnn_scoring,
            "--seed",
            str(seed),
            "--cpu",
            str(cpu),
        ]
        _log.debug("GNINA score_only cmd: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _log.warning("GNINA score_only timed out after %s s for '%s'", timeout_seconds, name)
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
            _log.warning(
                "GNINA score_only exit %s for '%s': stdout=%s stderr=%s",
                proc.returncode,
                name,
                (proc.stdout or "")[:400],
                (proc.stderr or "")[:400],
            )
            return []

        if not out_sdf.exists() or out_sdf.stat().st_size == 0:
            return []

        return _parse_gnina_sdf(out_sdf.read_text(), ligand_name=name)
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.cleanup()


def dock_ligand_adgpu_gnina(
    protein_pdb_path: str | Path,
    ligand_sdf_path: str | Path,
    box: DockingBox,
    *,
    ligand_name: str | None = None,
    gnina_binary: str = "gnina",
    cnn_scoring: str = "rescore",
    seed: int = 0,
    cpu: int = 1,
    num_modes: int = 9,
    adgpu_nrun: int | None = None,
    gnina_timeout_seconds: int | None = None,
    adgpu_timeout_seconds: int | None = None,
    adgpu_grid_timeout_seconds: int | None = None,
    output_dir: str | Path | None = None,
    write_gnina_logs: bool = True,
    adgpu_binary: str = "adgpu",
    autogrid_binary: str = "autogrid4",
    mk_prepare_receptor: str = "mk_prepare_receptor.py",
    mk_prepare_ligand: str = "mk_prepare_ligand.py",
    adgpu_delete_bad_res: bool = False,
) -> list[GninaDockResult]:
    """Dock with AutoDock-GPU, then GNINA ``--score_only`` on the best AD poses."""
    protein_pdb_path = Path(protein_pdb_path).resolve()
    ligand_sdf_path = Path(ligand_sdf_path).resolve()
    name = ligand_name or ligand_sdf_path.stem
    nrun = adgpu_nrun if adgpu_nrun is not None else max(num_modes, 9)
    grid_t = adgpu_grid_timeout_seconds or adgpu_timeout_seconds or 3600
    adgpu_t = adgpu_timeout_seconds or 3600

    if output_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="draco_adgpu_gnina_")
        root = Path(tmp_ctx.name).resolve()
        cleanup_ctx = tmp_ctx
    else:
        cleanup_ctx = None
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        grid_dir = root / "adgpu_grid"
        fld_path = grid_dir / "draco_rec.maps.fld"
        if not fld_path.is_file():
            _prepare_adgpu_grid(
                protein_pdb_path,
                box,
                grid_dir,
                mk_prepare_receptor=mk_prepare_receptor,
                autogrid_binary=autogrid_binary,
                delete_bad_res=adgpu_delete_bad_res,
                timeout_seconds=grid_t,
            )

        safe = name.replace(" ", "_").replace("/", "_")
        lig_work = root / "ligand_pdbqt" / safe
        lig_work.mkdir(parents=True, exist_ok=True)
        tasks = _ligand_sdf_to_pdbqt_tasks(
            ligand_sdf_path,
            lig_work,
            mk_prepare_ligand=mk_prepare_ligand,
            base_stem=safe,
            timeout_seconds=grid_t,
        )
        if not tasks:
            _log.warning("No PDBQT ligands produced for '%s'", name)
            return []

        adgpu_work = root / "adgpu_run" / safe
        if adgpu_work.exists():
            shutil.rmtree(adgpu_work, ignore_errors=True)
        adgpu_work.mkdir(parents=True, exist_ok=True)

        _run_adgpu_filelist(
            adgpu_work,
            fld_path,
            tasks,
            adgpu_binary=adgpu_binary,
            nrun=nrun,
            timeout_seconds=adgpu_t,
        )

        pose_mols = _collect_adgpu_poses_from_workdir(
            adgpu_work, tasks, max_poses=max(num_modes, 1)
        )
        if not pose_mols:
            _log.warning("AutoDock-GPU produced no poses for '%s'", name)
            return []

        poses_sdf = root / f"{name}.adgpu_poses.sdf"
        _write_mols_sdf([m for m, _e in pose_mols], poses_sdf, base_name=name)

        return dock_ligand_gnina_score_only(
            protein_pdb_path,
            poses_sdf,
            box,
            ligand_name=name,
            cnn_scoring=cnn_scoring,
            gnina_binary=gnina_binary,
            seed=seed,
            cpu=cpu,
            num_modes=len(pose_mols),
            timeout_seconds=gnina_timeout_seconds,
            output_dir=str(root),
            write_gnina_logs=write_gnina_logs,
        )
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.cleanup()


def dock_ligands_to_pocket_adgpu_gnina(
    protein_pdb_path: str | Path,
    ligand_sdf_paths: dict[str, Path],
    box: DockingBox,
    pocket_id: int = 0,
    **dock_kwargs: Any,
) -> PocketDockResult:
    """Dock multiple ligands with AutoDock-GPU + GNINA rescoring."""
    result = PocketDockResult(pocket_id=pocket_id, docking_box=box)
    for name, sdf_path in ligand_sdf_paths.items():
        poses = dock_ligand_adgpu_gnina(
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
      > <CNN_VS>              (CNN virtual screening score, 0-1)
    """
    results: list[GninaDockResult] = []
    # Split by the $$$$ record separator
    # IMPORTANT: do NOT `.strip()` the block text.
    # GNINA writes SD properties with required blank-line terminators; stripping
    # can delete those and break downstream SDF parsers and/or drop properties.
    sdf_text = sdf_text.replace("\r\n", "\n")
    blocks: list[str] = []
    for b in sdf_text.split("$$$$"):
        if not b.strip():
            continue
        # Avoid leading blank lines which can confuse some readers.
        blocks.append(b.lstrip("\n"))

    for rank, block in enumerate(blocks, start=1):
        vina = _parse_sdf_property(block, "minimizedAffinity", required=True)
        cnn_score = _parse_sdf_property(block, "CNNscore", required=True)
        cnn_aff = _parse_sdf_property(block, "CNNaffinity", required=True)
        cnn_vs = _parse_sdf_property(block, "CNN_VS", required=False) # Make CNN_VS optional to be safe, defaulting to 0.0 if missing

        if cnn_vs is None:
            cnn_vs = 0.0

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
                cnn_vs=cnn_vs,
                pose_sdf_block=(block if block.endswith("\n") else (block + "\n")) + "$$$$\n",
            )
        )

    # Note: Sorting here is still by CNN affinity to maintain previous local pose ranking behavior,
    # or should we sort by cnn_vs? Let's keep it sorted by CNN affinity for GNINA's local ordering
    # but the pipeline ranking will depend on the requested scoring method.
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


# ─────────────────────────────────────────────────────────────────────────────
# Glide docking backend
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GlideDockResult:
    """A single docked pose returned by Glide.

    Scores are parsed from SD properties. Glide may write ``*_lib.sdf`` or
    (newer releases) ``*_lib.maegz`` for ``POSE_OUTTYPE ligandlib``; Draco
    converts the latter to SDF via ``structconvert`` before parsing.
    """

    ligand_name: str
    """Name of the docked ligand."""

    pose_rank: int
    """1-based rank within the Glide output (best pose = 1)."""

    glide_score: float
    """GlideScore (kcal/mol). Lower (more negative) = better binding."""

    glide_emodel: float
    """Glide Emodel score (internal energy model). Lower = better."""

    glide_docking_score: float
    """Glide composite docking score (r_i_docking_score). Lower = better."""

    pose_sdf_block: str
    """The full SDF block for this pose (can be written back to disk)."""


def _schrodinger_binary(name: str) -> str:
    """Resolve a Schrödinger binary path.

    Uses the ``SCHRODINGER`` environment variable if set, otherwise returns
    the bare name (assumes it is on ``PATH``).

    Parameters
    ----------
    name:
        Binary name, e.g. ``'glide'``, ``'ligprep'``,
        ``'utilities/structconvert'``.
    """
    schrodinger = os.environ.get("SCHRODINGER", "").strip()
    if schrodinger:
        return str(Path(schrodinger) / name)
    return name


def _pdb_to_mae(
    pdb_path: Path,
    mae_path: Path,
    *,
    timeout_seconds: int | None = None,
) -> Path:
    """Convert a PDB file to Maestro (``.mae``) format using structconvert.

    Glide requires the receptor in Maestro format for grid generation.

    Parameters
    ----------
    pdb_path:
        Input PDB file path.
    mae_path:
        Output ``.mae`` file path.
    timeout_seconds:
        Optional wall-clock timeout in seconds.

    Returns
    -------
    Path
        Resolved path to the generated ``.mae`` file.

    Raises
    ------
    RuntimeError
        If structconvert fails or produces no output.
    """
    structconvert = _schrodinger_binary("utilities/structconvert")
    cmd = [structconvert, "-ipdb", str(pdb_path), "-omae", str(mae_path)]
    _log.debug("structconvert cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"structconvert not found at '{structconvert}'. "
            "Ensure $SCHRODINGER is set (module load schrodinger)."
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"structconvert failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[:400]}\nstderr: {proc.stderr[:400]}"
        )
    if not mae_path.exists() or mae_path.stat().st_size == 0:
        raise RuntimeError(f"structconvert produced no output MAE at '{mae_path}'.")
    return mae_path.resolve()


def _structconvert_to_sdf(
    input_path: Path,
    sdf_path: Path,
    *,
    timeout_seconds: int | None = None,
) -> Path:
    """Convert Maestro ``.mae``/``.maegz`` (or other structconvert input) to ``.sdf``.

    Glide 2025-2+ with ``POSE_OUTTYPE ligandlib`` often writes ``*_lib.maegz``
    instead of ``*_lib.sdf``. Draco parses SD format, so we normalize via
    ``$SCHRODINGER/utilities/structconvert`` (positional ``inputfile outputfile``).
    """
    structconvert = _schrodinger_binary("utilities/structconvert")
    in_p = input_path.resolve()
    out_p = sdf_path.resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [structconvert, str(in_p), str(out_p)]
    _log.debug("structconvert (→SDF) cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"structconvert not found at '{structconvert}'. "
            "Ensure $SCHRODINGER is set (module load schrodinger)."
        ) from None
    if proc.returncode != 0:
        raise RuntimeError(
            f"structconvert failed (exit {proc.returncode}):\n"
            f"stdout: {(proc.stdout or '')[:600]}\nstderr: {(proc.stderr or '')[:600]}"
        )
    if not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError(
            f"structconvert produced no output SDF at '{out_p}' "
            f"(input '{in_p}')."
        )
    return out_p


def generate_glide_grid(
    protein_pdb_path: str | Path,
    box: "DockingBox",
    output_dir: str | Path,
    *,
    job_name: str = "glide_grid",
    inner_box_scale: float = 0.7,
    glide_binary: str | None = None,
    timeout_seconds: int | None = 600,
    n_cpus: int | None = None,
) -> Path:
    """Generate a Glide receptor grid from a protein PDB and a docking box.

    This function:
    1. Converts the PDB receptor to Maestro format (required by Glide).
    2. Writes a ``glide-grid.in`` input file.
    3. Runs ``glide glide-grid.in -WAIT -LOCAL`` to produce the grid ``.zip``.

    Parameters
    ----------
    protein_pdb_path:
        Path to the receptor PDB file (no ligand, no water).
    box:
        Docking search box (from ``draco.pocket.docking_box_from_pocket``).
    output_dir:
        Directory in which to write intermediate files and the grid zip.
    job_name:
        Base name for Glide job files (default ``'glide_grid'``).
    inner_box_scale:
        Fraction of the outer box used as the inner (core sampling) box.
        Glide recommends inner box ≈ ligand diameter; default 0.7 of outer.
    glide_binary:
        Full path or name of the glide binary. Defaults to
        ``$SCHRODINGER/glide``.
    timeout_seconds:
        Wall-clock timeout for the grid generation subprocess (default 600 s).
    n_cpus:
        Number of CPU cores to request for the Glide job via
        ``-HOST localhost:N``. If None, Glide uses its own default (typically 1).
        On SLURM nodes pass the value of ``--n-cpus`` to make use of all
        allocated cores.

    Returns
    -------
    Path
        Resolved path to the generated ``.zip`` grid file.

    Raises
    ------
    RuntimeError
        If Glide fails or produces no grid.
    """
    protein_pdb_path = Path(protein_pdb_path).resolve()
    workdir = Path(output_dir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Convert PDB → MAE
    mae_path = workdir / f"{job_name}_receptor.mae"
    _pdb_to_mae(protein_pdb_path, mae_path, timeout_seconds=timeout_seconds)

    # 2. Compute box sizes
    outer_x, outer_y, outer_z = box.size_x, box.size_y, box.size_z
    # Glide expects INNERBOX as int_list; ceil avoids undersizing the box.
    inner_x = max(1, int(math.ceil(outer_x * inner_box_scale)))
    inner_y = max(1, int(math.ceil(outer_y * inner_box_scale)))
    inner_z = max(1, int(math.ceil(outer_z * inner_box_scale)))

    # 3. Write Glide grid input file
    grid_in_path = workdir / f"{job_name}.in"
    grid_zip_path = workdir / f"{job_name}.zip"
    grid_in_content = (
        f"GRIDFILE   {grid_zip_path}\n"
        f"RECEP_FILE   {mae_path}\n"
        f"GRID_CENTER   {box.center_x:.4f}, {box.center_y:.4f}, {box.center_z:.4f}\n"
        f"OUTERBOX   {outer_x:.4f}, {outer_y:.4f}, {outer_z:.4f}\n"
        f"INNERBOX   {inner_x}, {inner_y}, {inner_z}\n"
    )
    grid_in_path.write_text(grid_in_content)
    _log.debug("Glide grid input:\n%s", grid_in_content)

    # 4. Run Glide grid generation
    glide_bin = glide_binary or _schrodinger_binary("glide")
    cmd = [glide_bin, str(grid_in_path), "-WAIT", "-LOCAL"]
    if n_cpus and n_cpus > 1:
        cmd += ["-HOST", f"localhost:{n_cpus}"]
    _log.debug("Glide grid cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(workdir),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Glide grid generation timed out after {timeout_seconds} s."
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Glide binary not found at '{glide_bin}'. "
            "Ensure $SCHRODINGER is set (module load schrodinger)."
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Glide grid generation failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[:600]}\nstderr: {proc.stderr[:600]}"
        )

    if not grid_zip_path.exists() or grid_zip_path.stat().st_size == 0:
        raise RuntimeError(
            f"Glide produced no grid zip at '{grid_zip_path}'. "
            f"stdout: {proc.stdout[:400]}"
        )

    _log.info("Glide grid generated: %s", grid_zip_path)
    return grid_zip_path.resolve()


def dock_ligand_glide(
    protein_pdb_path: str | Path,
    ligand_maegz_path: str | Path,
    box: "DockingBox",
    *,
    ligand_name: str | None = None,
    precision: str = "SP",
    num_poses: int = 5,
    glide_binary: str | None = None,
    timeout_seconds: int | None = 600,
    output_dir: str | Path | None = None,
    grid_zip_path: str | Path | None = None,
    write_glide_logs: bool = True,
    n_cpus: int | None = None,
) -> list[GlideDockResult]:
    """Dock a single ligand into a pocket using Glide.

    Parameters
    ----------
    protein_pdb_path:
        Path to the receptor PDB file (no ligand, no water).
    ligand_maegz_path:
        Path to the LigPrep-prepared ligand ``.maegz`` file.
    box:
        Docking search box (e.g. from ``draco.pocket.docking_box_from_pocket``).
    ligand_name:
        Label for the ligand (used in result objects). Defaults to the
        ``.maegz`` file stem.
    precision:
        Glide docking precision: ``'HTVS'``, ``'SP'`` (default), or ``'XP'``.
    num_poses:
        Number of output poses per ligand (default 5).
    glide_binary:
        Full path or name of the glide binary.
    timeout_seconds:
        Wall-clock timeout for the Glide subprocess (default 600 s).
    output_dir:
        Directory for Glide output files. Uses a temporary directory if not
        provided.
    grid_zip_path:
        Pre-generated Glide receptor grid ``.zip``. If not provided, the grid
        is generated automatically via :func:`generate_glide_grid`.
    write_glide_logs:
        If True (default), preserve Glide stdout/stderr log files in
        ``output_dir``.
    n_cpus:
        Number of CPU cores to request for this Glide job via
        ``-HOST localhost:N``. If None, Glide uses its own default (typically 1).
        On SLURM nodes pass the value of ``--n-cpus`` to use all allocated cores.

    Returns
    -------
    list[GlideDockResult]
        Docked poses sorted best→worst by GlideScore (most negative first).
        Empty list if Glide fails or produces no poses.
    """
    protein_pdb_path = Path(protein_pdb_path).resolve()
    ligand_maegz_path = Path(ligand_maegz_path).resolve()
    name = ligand_name or ligand_maegz_path.stem

    if output_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="draco_glide_")
        workdir = Path(tmp_ctx.name).resolve()
        cleanup_ctx = tmp_ctx
    else:
        cleanup_ctx = None
        workdir = Path(output_dir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        job_name = f"{name}.glide"

        # 1. Ensure we have a receptor grid
        if grid_zip_path is not None:
            grid_zip = Path(grid_zip_path).resolve()
        else:
            grid_zip = generate_glide_grid(
                protein_pdb_path,
                box,
                output_dir=workdir,
                job_name=f"{job_name}_grid",
                glide_binary=glide_binary,
                timeout_seconds=timeout_seconds,
                n_cpus=n_cpus,
            )

        # 2. Write Glide docking input file
        out_sdf = workdir / f"{job_name}_lib.sdf"
        dock_in_path = workdir / f"{job_name}.in"
        dock_in_content = (
            f"GRIDFILE   {grid_zip}\n"
            f"LIGANDFILE   {ligand_maegz_path}\n"
            f"PRECISION   {precision}\n"
            f"POSES_PER_LIG   {num_poses}\n"
            f"POSE_OUTTYPE   ligandlib\n"
            f"WRITE_CSV   TRUE\n"
        )
        dock_in_path.write_text(dock_in_content)
        _log.debug("Glide dock input for '%s':\n%s", name, dock_in_content)

        # 3. Run Glide docking
        glide_bin = glide_binary or _schrodinger_binary("glide")
        stdout_path = workdir / f"{job_name}.stdout.log"
        stderr_path = workdir / f"{job_name}.stderr.log"

        cmd = [glide_bin, str(dock_in_path), "-WAIT", "-LOCAL"]
        if n_cpus and n_cpus > 1:
            cmd += ["-HOST", f"localhost:{n_cpus}"]
        _log.debug("Glide dock cmd: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(workdir),
            )
        except subprocess.TimeoutExpired:
            assert timeout_seconds is not None
            _log.warning("Glide timed out after %d s for ligand '%s'", timeout_seconds, name)
            return []
        except FileNotFoundError:
            raise RuntimeError(
                f"Glide binary not found at '{glide_bin}'. "
                "Ensure $SCHRODINGER is set (module load schrodinger)."
            )

        if write_glide_logs and output_dir is not None:
            try:
                stdout_path.write_text(proc.stdout or "")
                stderr_path.write_text(proc.stderr or "")
            except OSError as e:
                _log.warning("Failed writing Glide logs in '%s': %s", str(workdir), str(e))

        if proc.returncode != 0:
            err_str = proc.stderr[:500] if proc.stderr else ""
            out_str = proc.stdout[:500] if proc.stdout else ""
            _log.warning(
                "Glide returned exit code %d for ligand '%s'.\nstdout: %s\nstderr: %s",
                proc.returncode, name, out_str, err_str,
            )
            return []

        # 4. Locate pose output: Glide historically wrote {jobname}_lib.sdf; newer
        # releases (e.g. 2025-2) write {jobname}_lib.maegz for ligandlib jobs.
        in_stem = dock_in_path.stem  # e.g. "LIG.glide" — matches Glide's job basename
        out_maegz = workdir / f"{job_name}_lib.maegz"
        candidate_sdfs = list(workdir.glob(f"{in_stem}*.sdf")) + list(
            workdir.glob("*_lib.sdf")
        )
        candidate_maegz = list(workdir.glob(f"{in_stem}*_lib.maegz")) + list(
            workdir.glob("*_lib.maegz")
        )

        found_sdf: Path | None = None
        if out_sdf.exists() and out_sdf.stat().st_size > 0:
            found_sdf = out_sdf
        elif candidate_sdfs:
            found_sdf = max(candidate_sdfs, key=lambda p: p.stat().st_size)
        elif out_maegz.exists() and out_maegz.stat().st_size > 0:
            try:
                _structconvert_to_sdf(
                    out_maegz, out_sdf, timeout_seconds=timeout_seconds
                )
                if out_sdf.exists() and out_sdf.stat().st_size > 0:
                    found_sdf = out_sdf
            except Exception as exc:
                _log.warning(
                    "Glide wrote '%s' but maegz→sdf conversion failed for '%s': %s",
                    out_maegz,
                    name,
                    exc,
                )
        elif candidate_maegz:
            src_mae = max(candidate_maegz, key=lambda p: p.stat().st_size)
            try:
                _structconvert_to_sdf(
                    src_mae, out_sdf, timeout_seconds=timeout_seconds
                )
                if out_sdf.exists() and out_sdf.stat().st_size > 0:
                    found_sdf = out_sdf
            except Exception as exc:
                _log.warning(
                    "Found Glide library '%s' but maegz→sdf conversion failed for '%s': %s",
                    src_mae,
                    name,
                    exc,
                )

        if found_sdf is None:
            _log.warning(
                "Glide produced no usable pose library (no *_lib.sdf / *_lib.maegz) "
                "for ligand '%s'.\n  stdout: %s\n  stderr: %s",
                name,
                (proc.stdout or "")[:300],
                (proc.stderr or "")[:300],
            )
            return []

        # 5. Parse and return results
        results = _parse_glide_sdf(found_sdf.read_text(), ligand_name=name)

        # Also copy to the canonical output path if different
        if found_sdf != out_sdf and output_dir is not None:
            try:
                out_sdf.write_text(found_sdf.read_text())
            except OSError:
                pass

        return results
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.cleanup()


def dock_ligands_to_pocket_glide(
    protein_pdb_path: str | Path,
    ligand_maegz_paths: dict[str, Path],
    box: "DockingBox",
    pocket_id: int = 0,
    grid_zip_path: str | Path | None = None,
    **dock_kwargs: Any,
) -> "PocketDockResult":
    """Dock multiple ligands into the same pocket using Glide sequentially.

    Parameters
    ----------
    protein_pdb_path:
        Receptor PDB file path.
    ligand_maegz_paths:
        Mapping of ``{ligand_name: maegz_path}``.
    box:
        Docking box for this pocket.
    pocket_id:
        Integer identifier for this pocket (for tracking).
    grid_zip_path:
        Pre-generated Glide receptor grid. If not provided, generated
        automatically (once, shared across all ligands in this call).
    **dock_kwargs:
        Forwarded to :func:`dock_ligand_glide`. Includes ``n_cpus`` for
        multi-core Glide jobs and ``glide_binary``, ``timeout_seconds``, etc.

    Returns
    -------
    PocketDockResult
        Results dict maps ``ligand_name → list[GlideDockResult]``.
    """
    workdir_arg: str | Path | None = dock_kwargs.get("output_dir")

    # Generate the grid once for the whole pocket, then reuse.
    if grid_zip_path is None and workdir_arg is not None:
        workdir = Path(workdir_arg).resolve()
        try:
            grid_zip_path = generate_glide_grid(
                protein_pdb_path,
                box,
                output_dir=workdir,
                job_name="pocket_grid",
                glide_binary=dock_kwargs.get("glide_binary"),
                timeout_seconds=dock_kwargs.get("timeout_seconds", 600),
                n_cpus=dock_kwargs.get("n_cpus"),
            )
        except Exception as exc:
            _log.warning(
                "Failed to generate Glide grid for pocket %d: %s. "
                "Returning empty results.",
                pocket_id, exc,
            )
            result = PocketDockResult(pocket_id=pocket_id, docking_box=box)
            for name in ligand_maegz_paths:
                result.results[name] = []
            return result

    result = PocketDockResult(pocket_id=pocket_id, docking_box=box)

    # If all ligands share a single LigPrep library file, dock once and split
    # results by SDF title line to avoid N repeated Glide runs.
    unique_maegz = {str(Path(p).resolve()) for p in ligand_maegz_paths.values()}
    if len(unique_maegz) == 1 and ligand_maegz_paths:
        shared_maegz = next(iter(unique_maegz))
        batch_poses = dock_ligand_glide(
            protein_pdb_path,
            shared_maegz,
            box,
            ligand_name="__ligprep_library__",
            grid_zip_path=grid_zip_path,
            **dock_kwargs,
        )
        for lig_name in ligand_maegz_paths:
            result.results[lig_name] = []
        for pose in batch_poses:
            title = pose.pose_sdf_block.splitlines()[0].strip() if pose.pose_sdf_block else ""
            if title in result.results:
                result.results[title].append(
                    GlideDockResult(
                        ligand_name=title,
                        pose_rank=0,  # reassigned below after per-ligand sorting
                        glide_score=pose.glide_score,
                        glide_emodel=pose.glide_emodel,
                        glide_docking_score=pose.glide_docking_score,
                        pose_sdf_block=pose.pose_sdf_block,
                    )
                )
        for lig_name, poses in result.results.items():
            poses.sort(key=lambda r: r.glide_score)
            result.results[lig_name] = [
                GlideDockResult(
                    ligand_name=lig_name,
                    pose_rank=i,
                    glide_score=p.glide_score,
                    glide_emodel=p.glide_emodel,
                    glide_docking_score=p.glide_docking_score,
                    pose_sdf_block=p.pose_sdf_block,
                )
                for i, p in enumerate(poses, start=1)
            ]
        return result

    for name, maegz_path in ligand_maegz_paths.items():
        poses = dock_ligand_glide(
            protein_pdb_path,
            maegz_path,
            box,
            ligand_name=name,
            grid_zip_path=grid_zip_path,
            **dock_kwargs,
        )
        result.results[name] = poses  # type: ignore[assignment]
    return result


def _parse_glide_sdf(sdf_text: str, *, ligand_name: str) -> list[GlideDockResult]:
    """Parse Glide's ``_lib.sdf`` output and extract per-pose scores.

    Glide writes one SDF entry per pose. Relevant SD properties:

    .. code-block:: text

        > <r_i_glide_gscore>     (GlideScore, kcal/mol; lower = better)
        > <r_i_glide_emodel>     (Emodel; lower = better)
        > <r_i_docking_score>    (composite docking score; lower = better)

    Parameters
    ----------
    sdf_text:
        Full content of the Glide `_lib.sdf` file.
    ligand_name:
        Name to assign to all poses in this file.

    Returns
    -------
    list[GlideDockResult]
        Poses sorted best→worst by GlideScore (most negative first).
    """
    results: list[GlideDockResult] = []
    sdf_text = sdf_text.replace("\r\n", "\n")
    blocks: list[str] = []
    for b in sdf_text.split("$$$$"):
        if not b.strip():
            continue
        blocks.append(b.lstrip("\n"))

    for rank, block in enumerate(blocks, start=1):
        gscore = _parse_sdf_property(block, "r_i_glide_gscore", required=True)
        emodel = _parse_sdf_property(block, "r_i_glide_emodel", required=False)
        dscore = _parse_sdf_property(block, "r_i_docking_score", required=False)

        if gscore is None:
            # Truncated or invalid record — skip
            continue

        results.append(
            GlideDockResult(
                ligand_name=ligand_name,
                pose_rank=rank,
                glide_score=gscore,
                glide_emodel=emodel if emodel is not None else 0.0,
                glide_docking_score=dscore if dscore is not None else gscore,
                pose_sdf_block=(block if block.endswith("\n") else (block + "\n")) + "$$$$\n",
            )
        )

    # Sort best→worst (lowest GlideScore first)
    results.sort(key=lambda r: r.glide_score)
    return results
