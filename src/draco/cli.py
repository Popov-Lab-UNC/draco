"""cli.py – Entry point for the Draco pipeline: conformation sampling + GNINA docking.

Pipeline
--------
                  ┌──────────────────────────────────────────────────────────┐
  protein-pdb +   │               dynamics.py (this script calls it)          │
  compounds  ──►  │  OpenMM: prepare → solvate → MD → frame extraction         │
  .csv or         │  or BioEmu: sequence from PDB → samples → frames from XTC  │
  --ligand-smiles └─────────────────────────┬────────────────────────────────┘
                                            │  conformational frames
                                            ▼  (protein-only snapshots)
                              ┌─────────────────────────┐
                              │   per-frame pipeline     │
                              │                          │
                              │  1. pocketeer.find_pockets
                              │  2. GNINA docking        │
                              │     (actives+inactives)  │
                              │  3. SAR scoring (AUC-ROC)│
                              │     [only if CSV mode]   │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  top-k heap (rolling)   │
                              │  ranked by AUC / score  │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  refinement (OpeNMM)
                              │  (top-K poses only)     │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Output                │
                              │  summary.csv            │
                              │  top5_poses.pdb         │
                              │  trajectory (DCD or XTC)│
                              │  topology PDB           │
                              └─────────────────────────┘

Usage (SAR series mode):
    draco --mode sar \
        --steps dynamics pocket docking scoring refinement \
        --protein-pdb 6pbc-prepared.pdb \
        --ligand-csv compounds.csv \\
        --output-dir dynamics_docking_output

Usage (single-compound mode):
    draco --mode single \
        --steps dynamics pocket docking refinement \
        --protein-pdb 6pbc-prepared.pdb \
        --ligand-smiles "CN(CC1(CC1)c1ccc(F)cc1)C(=O)..." \\
        --output-dir dynamics_gnina_single

Usage (virtual screening mode):
    draco --mode screening \
        --steps dynamics pocket docking refinement \
        --protein-pdb 6pbc-prepared.pdb \
        --ligand-csv compounds.csv \\
        --output-dir dynamics_screening
"""
from __future__ import annotations

import os
import sys
import threading
import time

# Set thread limits BEFORE importing rdkit, numpy, openmm, py3dmol, pytorch etc.
# Without this, 15 concurrent workers on 16 cores will spin up 15*16 = 240 threads,
# causing severe CPU oversubscription, thrashing, and effectively freezing the pipeline.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# Resource detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_gpus() -> tuple[int, list[int], str]:
    """Detect the number and IDs of available CUDA GPUs.

    Returns
    -------
    (num_gpus, gpu_ids, source)
        ``num_gpus``  – integer count of visible CUDA GPUs (0 if none).
        ``gpu_ids``   – list of integer GPU indices (e.g. [0, 1]).
        ``source``    – human-readable string naming the detection method.
    """
    import subprocess

    # 1. Prefer Slurm's explicit allocation: SLURM_GPUS_ON_NODE=2
    slurm_gpus_on_node = os.environ.get("SLURM_GPUS_ON_NODE", "").strip()
    if slurm_gpus_on_node.isdigit() and int(slurm_gpus_on_node) > 0:
        n = int(slurm_gpus_on_node)
        return n, list(range(n)), "SLURM_GPUS_ON_NODE"

    # 2. SLURM_JOB_GPUS / SLURM_GPUS: can be comma-separated GPU IDs like "0,1"
    for env_var in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_STEP_GPUS"):
        val = os.environ.get(env_var, "").strip()
        if val:
            ids = [int(x) for x in val.split(",") if x.strip().isdigit()]
            if ids:
                return len(ids), ids, env_var

    # 3. CUDA_VISIBLE_DEVICES: e.g. "0,1" or "NoDevFiles" (none)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd.upper() not in ("", "NODEVFILES", "-1"):
        ids = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        if ids:
            return len(ids), ids, "CUDA_VISIBLE_DEVICES"

    # 4. nvidia-smi subprocess fallback
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
            if ids:
                return len(ids), ids, "nvidia-smi"
    except Exception:
        pass

    # 5. No GPUs found
    return 0, [], "none"


def _detect_cpus() -> tuple[int, str]:
    """Detect the number of CPUs allocated to this job/process.

    Returns
    -------
    (num_cpus, source)
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus.isdigit() and int(slurm_cpus) > 0:
        return int(slurm_cpus), "SLURM_CPUS_PER_TASK"
    return int(os.cpu_count() or 1), "os.cpu_count()"

import argparse
import concurrent.futures
import csv
import heapq
import io
import threading
import time
import multiprocessing
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from draco.dynamics import DynamicsFrame, DynamicsResult, run_bioemu_sampling, run_dynamics
from draco.docking import (
    DockingBox, GninaDockResult, PocketDockResult,
    dock_ligands_to_pocket,
)
from draco.ligand_preparation import (
    PreparedLigand, prepare_ligand_from_smiles,
    load_compound_csv, load_screening_csv, write_ligand_sdf, write_ligands_for_docking,
)
from draco.refinement import RefinementResult, refine_docked_pose
from draco.sar_scoring import SARScoreResult, compute_sar_discrimination
from draco.protein_preparation import PreparedProtein, prepare_protein

try:
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, PDBxFile, Simulation
except ImportError:  # pragma: no cover
    from simtk.openmm import LangevinMiddleIntegrator, Platform, unit  # type: ignore
    from simtk.openmm.app import (  # type: ignore
        ForceField, HBonds, Modeller, NoCutoff, PDBFile, PDBxFile, Simulation,
    )

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.distributed\.reduce_op.*",
)


# ─────────────────────────────────────────────────────────────────────────────
# Progress / timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    """Return current wall-clock time as a compact string: HH:MM:SS."""
    return time.strftime("%H:%M:%S")


def _elapsed(start: float) -> str:
    """Return human-readable elapsed time since *start* (time.monotonic())."""
    secs = int(time.monotonic() - start)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draco: MD → Pocketeer → GNINA docking → SAR scoring → top-k refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Primary Execution Parameters ──────────────────────────────────────────
    primary_grp = parser.add_argument_group("Primary Execution Parameters")
    primary_grp.add_argument(
        "--mode",
        required=True,
        choices=["single", "sar", "screening"],
        help="[REQUIRED] Mode of operation: 'single' for one compound, 'sar' for structure-activity discrimination, or 'screening' for unlabeled virtual screening."
    )
    primary_grp.add_argument(
        "--steps",
        nargs="+",
        default=["dynamics", "pocket", "docking", "scoring", "refinement"],
        choices=["dynamics", "pocket", "docking", "scoring", "refinement"],
        help="[OPTIONAL] Pipeline steps. With 'pocket', artifacts are written to "
        "<output-dir>/pockets/. Docking/scoring without 'pocket' reloads those artifacts."
    )
    primary_grp.add_argument(
        "--protein-pdb",
        required=True,
        help="[REQUIRED] Input protein PDB path."
    )
    primary_grp.add_argument(
        "--ligand-csv",
        help=(
            "[REQUIRED for --mode sar/screening] CSV for ligand input. "
            "Default columns: SAR uses name,smiles,active (1/0); "
            "screening uses name,smiles. Override via --csv-*-column flags."
        ),
    )
    primary_grp.add_argument(
        "--csv-name-column",
        default="name",
        help="Column name to use as ligand identifier in --ligand-csv.",
    )
    primary_grp.add_argument(
        "--csv-smiles-column",
        default="smiles",
        help="Column name to use as SMILES in --ligand-csv.",
    )
    primary_grp.add_argument(
        "--csv-activity-column",
        default="active",
        help="Column name to use as activity label (SAR mode only).",
    )
    primary_grp.add_argument(
        "--ligand-smiles",
        help="[REQUIRED for --mode single] Single ligand SMILES. Poses ranked by CNN affinity (no SAR scoring)."
    )
    primary_grp.add_argument(
        "--output-dir",
        default="dynamics_docking_output",
        help="[OPTIONAL] Directory for output files."
    )

    # ── Output & Global Parameters ────────────────────────────────────────────
    global_grp = parser.add_argument_group("Global & Output Parameters")
    global_grp.add_argument("--ph", type=float, default=7.4, help="Target pH for protonation states.")
    global_grp.add_argument("--top-k", type=int, default=5, help="Number of top poses to keep.")
    global_grp.add_argument("--top-output", default="top5_poses.cif", help="Filename for the top poses (CIF format).")
    global_grp.add_argument("--dry-run", action="store_true", default=False, help="Run without computationally intensive steps.")

    # ── Dynamics (MD) Parameters ──────────────────────────────────────────────
    dyn_grp = parser.add_argument_group("Dynamics (MD) Parameters")
    dyn_grp.add_argument("--platform-name", default="CUDA", help="OpenMM platform name.")
    dyn_grp.add_argument("--cuda-precision", default="mixed", help="CUDA precision.")
    dyn_grp.add_argument("--box-padding-nm", type=float, default=1.0, help="Solvent box padding in nm.")
    dyn_grp.add_argument("--ionic-strength", type=float, default=0.15, help="Ionic strength in Molar.")
    dyn_grp.add_argument("--nvt-steps", type=int, default=50_000, help="Number of NVT equilibration steps.")
    dyn_grp.add_argument("--npt-steps", type=int, default=50_000, help="Number of NPT equilibration steps.")
    dyn_grp.add_argument("--production-steps", type=int, default=2_500_000, help="Number of production MD steps.")
    dyn_grp.add_argument("--timestep-fs", type=float, default=2.0, help="MD timestep in femtoseconds.")
    dyn_grp.add_argument("--friction-per-ps", type=float, default=1.0, help="Langevin friction coefficient (1/ps).")
    dyn_grp.add_argument("--temperature-kelvin", type=float, default=300.0, help="Simulation temperature in Kelvin.")
    dyn_grp.add_argument("--water-model", default="tip3pfb", help="Water model for solvation.")
    dyn_grp.add_argument("--report-interval-steps", type=int, default=5_000, help="Steps between frame reports.")
    dyn_grp.add_argument("--rmsd-threshold-angstrom", type=float, default=1.5, help="C-alpha RMSD threshold for keeping a frame.")
    dyn_grp.add_argument("--no-trajectory", action="store_true", default=False, help="Disable saving the MD trajectory (DCD).")
    dyn_grp.add_argument(
        "--dynamics-backend",
        default="md",
        choices=["md", "bioemu"],
        help="Conformation source: explicit-solvent OpenMM MD (default) or BioEmu sampling.",
    )
    dyn_grp.add_argument(
        "--bioemu-num-samples",
        type=int,
        default=10,
        metavar="N",
        help="Number of BioEmu samples to generate (only used with --dynamics-backend bioemu).",
    )

    # ── Pocket & Ligand Prep Parameters ───────────────────────────────────────
    pocket_grp = parser.add_argument_group("Pocket & Ligand Prep Parameters")
    pocket_grp.add_argument("--pocket-score-threshold", type=float, default=5.0, help="Pocketeer score threshold.")
    pocket_grp.add_argument("--num-conformers", type=int, default=10, help="Number of conformers to generate for ligands.")
    pocket_grp.add_argument("--prune-rms-threshold", type=float, default=1.0, help="RMSD threshold for conformer pruning.")
    pocket_grp.add_argument("--energy-cutoff", type=float, default=5.0, help="Energy cutoff (kcal/mol) for conformer pruning.")
    pocket_grp.add_argument("--ligand-name", default="LIG", help="Name for --ligand-smiles compound.")

    # ── GNINA Docking Parameters ──────────────────────────────────────────────
    docking_grp = parser.add_argument_group("GNINA Docking Parameters")
    docking_grp.add_argument("--gnina-binary", default="gnina", help="Path or name of the gnina binary.")
    docking_grp.add_argument("--exhaustiveness", type=int, default=8, help="GNINA exhaustiveness.")
    docking_grp.add_argument("--num-modes", type=int, default=9, help="Number of docked poses per ligand.")
    docking_grp.add_argument("--cnn-scoring", default="rescore", choices=["rescore", "refinement", "none"], help="GNINA CNN scoring mode.")
    docking_grp.add_argument("--gnina-seed", type=int, default=0, help="Random seed for GNINA.")
    docking_grp.add_argument("--gnina-timeout-seconds", type=int, default=None, metavar="N", help="Wall-clock limit per GNINA subprocess (seconds).")
    docking_grp.add_argument(
        "--dock-filter",
        nargs="+",
        metavar="SPEC",
        default=None,
        help=(
            "Restrict docking to specific frame:pocket pairs. "
            "Each SPEC is FRAME_IDX:POCKET_ID; multiple pairs can be comma- or space-separated. "
            "Omit the pocket (just FRAME_IDX) to dock all pockets for that frame. "
            "Examples: '1:2 3:0 4:0' or '1:2, 3:0, 4:0' (frame 1 pocket 2, frame 3 pocket 0, etc.). "
            "If omitted entirely, all frames and pockets are docked."
        ),
    )
    docking_grp.add_argument("--scoring-method", default="cnn_vs", choices=["cnn_vs", "cnn_affinity", "cnn_score", "vina_score"], help="Which GNINA metric to use for ranking poses and computing SAR.")
    docking_grp.add_argument(
        "--sar-metric",
        default="auc_roc",
        choices=["auc_roc", "auc_pr", "enrichment_1pct", "enrichment_5pct", "enrichment_10pct"],
        help="Metric to rank poses by in 'sar' mode. Defaults to auc_roc.",
    )
    docking_grp.add_argument("--max-docking-workers", type=int, default=None, help="Maximum concurrent GNINA instances. Defaults to min(4, CPU_COUNT//4).")
    docking_grp.add_argument("--gnina-cpu", type=int, default=None, metavar="N", help="CPU threads per GNINA call for Vina sampling. Auto-computed as CPU_COUNT // docking_workers when omitted (recommended).")

    # ── Refinement Parameters ───────────────────────────────────────────────────
    refine_grp = parser.add_argument_group("Refinement Parameters")
    refine_grp.add_argument("--shell-radius-angstrom", type=float, default=8.0, help="Radius around ligand to keep flexible.")
    refine_grp.add_argument("--protein-restraint-k", type=float, default=10.0, help="Restraint force constant for non-flexible protein atoms.")
    refine_grp.add_argument("--refine-iterations", type=int, default=500, help="Maximum number of refinement iterations.")
    refine_grp.add_argument("--no-interaction-energy", action="store_true", default=False, help="Disable computing the interaction energy.")

    args = parser.parse_args()

    # Post-parsing validation
    if args.mode == "sar":
        if not args.ligand_csv:
            parser.error("--ligand-csv is REQUIRED when --mode is 'sar'.")
        if args.ligand_smiles:
            parser.error("--ligand-smiles cannot be used when --mode is 'sar'. Use --ligand-csv instead.")
    elif args.mode == "screening":
        if not args.ligand_csv:
            parser.error("--ligand-csv is REQUIRED when --mode is 'screening'.")
        if args.ligand_smiles:
            parser.error("--ligand-smiles cannot be used when --mode is 'screening'. Use --ligand-csv instead.")
    elif args.mode == "single":
        if not args.ligand_smiles:
            parser.error("--ligand-smiles is REQUIRED when --mode is 'single'.")
        if args.ligand_csv:
            parser.error("--ligand-csv cannot be used when --mode is 'single'. Use --ligand-smiles instead.")

    if args.bioemu_num_samples < 1:
        parser.error("--bioemu-num-samples must be >= 1.")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Top-k tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FramePoseResult:
    """Result for one (conformation, pocket, ligand) docking + optional refinement."""
    frame_index: int
    frame_time_ps: float
    pocket_id: int
    pocket_score: float          # pocketeer score
    ligand_name: str             # name of the docked compound
    cnn_affinity: float          # best GNINA CNN affinity (pK; higher = better)
    vina_score: float            # best GNINA Vina score
    cnn_score: float             # best GNINA CNN pose score (0-1)
    cnn_vs: float                # best GNINA CNN VS score (0-1)
    auc_roc: float | None        # SAR discrimination metric (None in single-compound mode)
    docked_sdf_block: str = ""   # The SDF block of the docked pose
    # Post-refinement fields (filled only for top-K poses)
    initial_energy_kj_per_mol: float = 0.0
    final_energy_kj_per_mol: float = 0.0
    interaction_energy_kj_per_mol: float | None = None
    ligand_rmsd_from_dock_angstrom: float = 0.0
    protein_atoms_flexible: int = 0
    protein_atoms_restrained: int = 0
    refined_complex_pdb: str = ""
    refined_complex_cif: str = ""
    status: str = "ok"
    error: str = ""
    smiles: str = ""             # canonical smiles of the docked compound (if available)


@dataclass
class _TopKHeap:
    k: int
    ranking_metric: str = "cnn_vs"  # depends on args.scoring_method or args.sar_metric
    _heap: list[tuple[float, int, _FramePoseResult]] = field(default_factory=list)
    _counter: int = 0

    def _rank_score(self, result: _FramePoseResult) -> float:
        # If in SAR mode, the ranking metric is a SAR metric (e.g. auc_roc)
        if self.ranking_metric in ("auc_roc", "auc_pr", "enrichment_1pct", "enrichment_5pct", "enrichment_10pct"):
            sar_val = getattr(result, self.ranking_metric, None)
            # If SAR metric is missing, fall back to cnn_vs
            return sar_val if sar_val is not None else result.cnn_vs

        # In single mode, the ranking metric is the primary scoring method.
        # Note: vina_score is lower-is-better, but we want to maximize the rank score.
        # So if ranking_metric is vina_score, we must negate it so that the heap works correctly (it's a min-heap tracking the worst retained entry)
        # Actually, higher=better is assumed by _TopKHeap logic.
        val = getattr(result, self.ranking_metric)
        if self.ranking_metric == "vina_score":
            return -val
        return val

    def push(self, result: _FramePoseResult) -> bool:
        """Push by selected ranking metric (higher = better)."""
        # Keep a size-k min-heap keyed by ranking score.
        # Heap root is the current worst among retained entries.
        rank_score = self._rank_score(result)
        entry = (rank_score, self._counter, result)
        self._counter += 1
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            return True
        elif rank_score > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            return True
        return False

    def sorted_best(self) -> list[_FramePoseResult]:
        return [r for _, _, r in sorted(self._heap, key=lambda t: t[0], reverse=True)]

    def current_best(self) -> _FramePoseResult | None:
        if not self._heap:
            return None
        return self.sorted_best()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Parallel Worker Logic
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dock_filter(tokens: list[str]) -> dict[int, set[int] | None]:
    """Parse ``--dock-filter`` tokens into ``{frame_idx: pocket_set | None}``.

    ``None`` as the value means "all pockets for this frame".

    Tokens are comma- and/or space-separated ``FRAME:POCKET`` pairs.
    A bare ``FRAME`` (no colon) means all pockets for that frame.
    Specifying the same frame multiple times merges the pocket sets.

    Examples::

        ["1:2"]              → {1: {2}}
        ["1:2", "3:0", "4:0"] → {1: {2}, 3: {0}, 4: {0}}
        ["1:2, 3:0, 4:0"]   → {1: {2}, 3: {0}, 4: {0}}   # comma-separated in one string
        ["1:2,3:0"]         → {1: {2}, 3: {0}}             # no spaces needed
        ["1:2", "1:3"]      → {1: {2, 3}}                  # same frame, merged
        ["1"]               → {1: None}                     # frame 1, all pockets
    """
    result: dict[int, set[int] | None] = {}
    # Normalize: join all tokens, split on whitespace and commas
    raw = " ".join(tokens)
    items = [s.strip() for s in raw.replace(",", " ").split() if s.strip()]
    for item in items:
        if ":" in item:
            frame_str, pocket_str = item.split(":", 1)
            frame_idx = int(frame_str.strip())
            pocket_id = int(pocket_str.strip())
            if frame_idx in result and result[frame_idx] is None:
                pass  # already "all pockets" — don't narrow it back down
            elif frame_idx in result:
                result[frame_idx].add(pocket_id)  # type: ignore[union-attr]
            else:
                result[frame_idx] = {pocket_id}
        else:
            frame_idx = int(item.strip())
            result[frame_idx] = None  # all pockets for this frame
    return result


def _worker_init(gpu_id: int | None = None) -> None:
    """Initializer for ProcessPoolExecutor workers.

    Parameters
    ----------
    gpu_id:
        If provided, pin this worker to the specified CUDA device by setting
        ``CUDA_VISIBLE_DEVICES`` before GNINA is invoked. Set to ``None`` for
        CPU-only workers (pocket detection, refinement).
    """
    os.environ["OPENMM_CPU_THREADS"] = "1"
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import openmm
        import rdkit
        import draco.pocket
        import draco.protein_preparation
    except ImportError:
        pass


def _dock_frame_worker(
    frame_index: int,
    frame_time_ps: float,
    protein_pdb_path: str,       # path to temp PDB file for this frame
    protein_pdb_string: str,     # PDB text (kept for passing forward)
    ligand_sdf_paths: dict[str, str],  # {state_name: sdf_path_str}
    name_map: dict[str, str],          # {state_name: parent_ligand_name}
    parent_smiles_map: dict[str, str], # {parent_ligand_name: canonical_smiles}
    active_names: list[str],           # parent-level active names
    inactive_names: list[str],         # parent-level inactive names
    pocket_score_threshold: float,
    gnina_binary: str,
    exhaustiveness: int,
    num_modes: int,
    cnn_scoring: str,
    gnina_seed: int,
    dry_run: bool,
    scoring_method: str = "cnn_vs",
    sar_metric: str = "auc_roc",
    docking_output_dir: str | None = None,
    gnina_timeout_seconds: int | None = None,
    steps: list[str] | None = None,
    project_output_dir: str | None = None,
    pocket_filter: set[int] | None = None,
    gnina_cpu_threads: int = 1,
    gpu_id: int | None = None,
) -> list[_FramePoseResult]:
    """Worker: Detect pockets, dock all ligands with GNINA, compute SAR score.

    All protonation states of each parent ligand are docked. Scores are
    aggregated to report the **best score per parent ligand** for SAR
    discrimination and single-compound ranking.

    If *gpu_id* is provided, the worker sets ``CUDA_VISIBLE_DEVICES`` so that
    GNINA uses only that specific GPU (useful for multi-GPU nodes).

    Returns one _FramePoseResult per (pocket, parent_ligand) pair in `single`
    and `screening` modes.
    In `sar` mode it returns one result per (frame, pocket) with SAR discrimination
    metrics (and a representative active ligand used for pose export).
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from draco.docking import DockingBox, dock_ligands_to_pocket
    from draco.pocket import (
        docking_box_from_pocket,
        find_pockets_above_threshold,
        load_pocket_entries,
        write_pocket_artifact_for_frame,
    )
    from draco.sar_scoring import compute_sar_discrimination
    from draco.docking import PocketDockResult

    if dry_run:
        return []

    if steps is None:
        steps = ["dynamics", "pocket", "docking", "scoring", "refinement"]

    work_items: list[tuple[int, float, DockingBox]] = []

    if "pocket" in steps:
        try:
            pockets = find_pockets_above_threshold(
                protein_pdb_string, pocket_score_threshold
            )
        except Exception as exc:
            import traceback
            return [_FramePoseResult(
                frame_index=frame_index, frame_time_ps=frame_time_ps,
                pocket_id=-1, pocket_score=0.0, ligand_name="",
                cnn_affinity=0.0, vina_score=0.0, auc_roc=None,
                status="error", error=f"Pocketeer failed: {exc}\n{traceback.format_exc()}"
            )]
        if project_output_dir:
            write_pocket_artifact_for_frame(project_output_dir, frame_index, pockets)
        if not pockets:
            return []
        for idx, pocket in enumerate(pockets):
            pid = int(getattr(pocket, "pocket_id", idx))
            score = float(getattr(pocket, "score", 0.0))
            box = docking_box_from_pocket(pocket)
            work_items.append((pid, score, box))
    elif "docking" in steps or "scoring" in steps:
        if not project_output_dir:
            raise ValueError(
                "project_output_dir is required when running docking or scoring without the "
                "'pocket' step (expected pocket artifacts under <output-dir>/pockets/)."
            )
        try:
            work_items = load_pocket_entries(project_output_dir, frame_index)
        except FileNotFoundError as exc:
            raise ValueError(
                "Docking/scoring without the 'pocket' step requires precomputed pocket "
                f"artifacts. {exc}"
            ) from exc
        if not work_items:
            return []

    if not work_items:
        return []

    results: list[_FramePoseResult] = []
    sar_mode = bool(active_names or inactive_names)

    # Build the set of state-level active/inactive names from the name_map
    active_state_names = {sn for sn, pn in name_map.items() if pn in set(active_names)}
    inactive_state_names = {sn for sn, pn in name_map.items() if pn in set(inactive_names)}

    for pocket_id, pocket_score, box in work_items:
        if pocket_filter is not None and pocket_id not in pocket_filter:
            continue

        # Create per-pocket output dir for GNINA
        if docking_output_dir:
            pocket_out = __import__('pathlib').Path(docking_output_dir) / f"frame{frame_index:04d}_pocket{pocket_id}"
            pocket_out.mkdir(parents=True, exist_ok=True)
            out_dir_str = str(pocket_out)
        else:
            out_dir_str = None

        try:
            if "docking" not in steps and out_dir_str:
                import draco.docking as docking
                pocket_dock = docking.PocketDockResult(pocket_id=pocket_id, docking_box=box)
                for name, sdf_path in ligand_sdf_paths.items():
                    out_sdf = Path(out_dir_str) / f"{name}.gnina.sdf"
                    if out_sdf.exists():
                        pocket_dock.results[name] = docking._parse_gnina_sdf(out_sdf.read_text(), ligand_name=name)
                    else:
                        pocket_dock.results[name] = []
            elif "docking" in steps:
                pocket_dock = dock_ligands_to_pocket(
                    protein_pdb_path,
                    {n: __import__('pathlib').Path(p) for n, p in ligand_sdf_paths.items()},
                    box,
                    pocket_id=pocket_id,
                    gnina_binary=gnina_binary,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    cnn_scoring=cnn_scoring,
                    seed=gnina_seed,
                    cpu=gnina_cpu_threads,
                    timeout_seconds=gnina_timeout_seconds,
                    output_dir=out_dir_str,
                    write_gnina_logs=True,
                )
            else:
                pocket_dock = PocketDockResult(pocket_id=pocket_id, docking_box=box)
        except Exception as exc:
            import traceback
            results.append(_FramePoseResult(
                frame_index=frame_index, frame_time_ps=frame_time_ps,
                pocket_id=pocket_id, pocket_score=pocket_score, ligand_name="",
                cnn_affinity=0.0, vina_score=0.0, auc_roc=None,
                status="error", error=f"GNINA failed pocket {pocket_id}: {exc}\n{traceback.format_exc()}"
            ))
            continue

        # ── Aggregate best score per parent ligand ─────────────────────────
        # pocket_dock.results keys are state-level names (e.g. "LIG_s0").
        # We group by parent name and pick the best pose (by scoring_method) per parent.
        parent_best: dict[str, tuple[float, float, float, float, float, str]] = {}  # parent -> (best_scoring_metric, cnn_affinity, vina_score, cnn_score, cnn_vs, best_sdf)
        for state_name, poses in pocket_dock.results.items():
            if not poses:
                continue
            parent = name_map.get(state_name, state_name)

            # Find the best pose for this state by the requested scoring method
            if scoring_method == "vina_score":
                best_pose = min(poses, key=lambda p: p.vina_score)
            else:
                best_pose = max(poses, key=lambda p: getattr(p, scoring_method))

            p_val = getattr(best_pose, scoring_method)

            # Determine if this state's best pose is better than the parent's current best
            is_better = False
            if parent not in parent_best:
                is_better = True
            else:
                current_best_val = parent_best[parent][0]
                if scoring_method == "vina_score":
                    is_better = p_val < current_best_val
                else:
                    is_better = p_val > current_best_val

            if is_better:
                parent_best[parent] = (p_val, best_pose.cnn_affinity, best_pose.vina_score, best_pose.cnn_score, best_pose.cnn_vs, best_pose.pose_sdf_block)

        # In SAR mode, also write a ligand-only SDF for this pocket containing the
        # best pose per parent ligand, sorted by the selected scoring method.
        if sar_mode and out_dir_str and parent_best:
            try:
                # Sort parents by "best pose" scoring_method.
                # For vina_score: lower is better. For others: higher is better.
                def _parent_sort_key(item: tuple[str, tuple[float, float, float, float, float, str]]) -> float:
                    score_val = item[1][0]
                    return score_val if scoring_method != "vina_score" else -score_val

                ranked = sorted(parent_best.items(), key=_parent_sort_key, reverse=True)
                ranked_sdf_blocks = [t[-1] for _, t in ranked if (t[-1] or "").strip()]
                ranked_sdf_path = Path(out_dir_str) / f"ligands_ranked_by_{scoring_method}.sdf"
                _write_multimol_sdf(ranked_sdf_path, ranked_sdf_blocks)
            except Exception:
                # Don't fail docking/scoring just because an optional export failed.
                pass

        if sar_mode:
            # Build a virtual PocketDockResult with best-per-parent scores
            # for SAR discrimination. We create a synthetic results dict
            # mapping parent names to the best poses from any state.
            parent_results: dict[str, list] = {}
            for state_name, poses in pocket_dock.results.items():
                parent = name_map.get(state_name, state_name)
                if parent not in parent_results:
                    parent_results[parent] = []
                parent_results[parent].extend(poses)

            # Build a synthetic PocketDockResult for SAR scoring at parent level
            parent_dock = PocketDockResult(
                pocket_id=pocket_id,
                docking_box=box,
                results=parent_results,
            )
            auc_roc = None
            auc_pr = None
            enrichment_1pct = None
            enrichment_5pct = None
            enrichment_10pct = None
            n_actives = None
            n_inactives = None
            active_mean_score = None
            inactive_mean_score = None
            active_std_score = None
            inactive_std_score = None
            active_min_score = None
            active_max_score = None
            inactive_min_score = None
            inactive_max_score = None
            active_best_score = None
            inactive_best_score = None
            overall_min_score = None
            overall_max_score = None
            mean_rank_active_minus_inactive = None
            if "scoring" in steps:
                sar = compute_sar_discrimination(
                    frame_index=frame_index,
                    pocket_result=parent_dock,
                    active_names=set(active_names),
                    inactive_names=set(inactive_names),
                    score_key=scoring_method,
                )
                auc_roc = sar.auc_roc
                auc_pr = sar.auc_pr
                enrichment_1pct = sar.enrichment_1pct
                enrichment_5pct = sar.enrichment_5pct
                enrichment_10pct = sar.enrichment_10pct
                n_actives = sar.n_actives
                n_inactives = sar.n_inactives
                active_mean_score = sar.active_mean_score
                inactive_mean_score = sar.inactive_mean_score
                active_std_score = sar.active_std_score
                inactive_std_score = sar.inactive_std_score
                active_min_score = sar.active_min_score
                active_max_score = sar.active_max_score
                inactive_min_score = sar.inactive_min_score
                inactive_max_score = sar.inactive_max_score
                active_best_score = sar.active_best_score
                inactive_best_score = sar.inactive_best_score
                overall_min_score = sar.overall_min_score
                overall_max_score = sar.overall_max_score
                mean_rank_active_minus_inactive = sar.mean_rank_active_minus_inactive

                # Fetch whichever sar metric the user requested to set correctly for the summary if we need
                # but currently we only track auc_roc explicitly in _FramePoseResult, so let's set auc_roc to the sar metric
                # actually _FramePoseResult is requested to only have auc_roc, or should we add the others?
                # Let's map auc_roc to the requested sar_metric so it can be sorted properly in the summary.csv
                sar_val = getattr(sar, sar_metric)

            # Best-scoring active as the representative pose
            best_active = ""
            best_active_val = None
            for n in active_names:
                if n in parent_best:
                    val = parent_best[n][0]
                    if best_active_val is None:
                        best_active = n
                        best_active_val = val
                    else:
                        if scoring_method == "vina_score":
                            if val < best_active_val:
                                best_active = n
                                best_active_val = val
                        else:
                            if val > best_active_val:
                                best_active = n
                                best_active_val = val

            if best_active:
                _, best_cnn_aff, best_vina, best_cnn_score, best_cnn_vs, best_sdf = parent_best[best_active]
            else:
                best_cnn_aff, best_vina, best_cnn_score, best_cnn_vs, best_sdf = 0.0, 0.0, 0.0, 0.0, ""

            r = _FramePoseResult(
                frame_index=frame_index, frame_time_ps=frame_time_ps,
                pocket_id=pocket_id, pocket_score=pocket_score,
                ligand_name=best_active,
                smiles=parent_smiles_map.get(best_active, ""),
                cnn_affinity=best_cnn_aff, vina_score=best_vina,
                cnn_score=best_cnn_score, cnn_vs=best_cnn_vs,
                auc_roc=sar_val if "scoring" in steps else None,
                docked_sdf_block=best_sdf,
                status="ok",
            )
            # monkey-patch other metrics in case TopKHeap wants them
            if "scoring" in steps:
                r.auc_roc = auc_roc
                setattr(r, "auc_pr", auc_pr)
                setattr(r, "n_actives", n_actives)
                setattr(r, "n_inactives", n_inactives)
                setattr(r, "active_mean_score", active_mean_score)
                setattr(r, "inactive_mean_score", inactive_mean_score)
                setattr(r, "active_std_score", active_std_score)
                setattr(r, "inactive_std_score", inactive_std_score)
                setattr(r, "active_min_score", active_min_score)
                setattr(r, "active_max_score", active_max_score)
                setattr(r, "inactive_min_score", inactive_min_score)
                setattr(r, "inactive_max_score", inactive_max_score)
                setattr(r, "active_best_score", active_best_score)
                setattr(r, "inactive_best_score", inactive_best_score)
                setattr(r, "overall_min_score", overall_min_score)
                setattr(r, "overall_max_score", overall_max_score)
                setattr(r, "mean_rank_active_minus_inactive", mean_rank_active_minus_inactive)
                setattr(r, "enrichment_1pct", enrichment_1pct)
                setattr(r, "enrichment_5pct", enrichment_5pct)
                setattr(r, "enrichment_10pct", enrichment_10pct)
                # Re-assign the correct one to auc_roc for backward compatibility
                # in the summary CSV, or leave it. Actually the user wants auc_roc in summary csv, but might want to sort by something else.
                # Actually _FramePoseResult now has auc_roc. We will set the chosen sar_metric value to the field matching it if we had it, but we only have auc_roc in _FramePoseResult.
                # Let's just set the `auc_roc` field to auc_roc, and monkeypatch the requested `sar_metric` for TopKHeap.
                setattr(r, sar_metric, sar_val)

            results.append(r)
        else:
            # Single compound mode: one result per pocket per parent ligand
            for parent, (_, best_cnn_aff, best_vina, best_cnn_score, best_cnn_vs, best_sdf) in parent_best.items():
                results.append(_FramePoseResult(
                    frame_index=frame_index, frame_time_ps=frame_time_ps,
                    pocket_id=pocket_id, pocket_score=pocket_score, ligand_name=parent,
                    smiles=parent_smiles_map.get(parent, ""),
                    cnn_affinity=best_cnn_aff,
                    vina_score=best_vina,
                    cnn_score=best_cnn_score,
                    cnn_vs=best_cnn_vs,
                    auc_roc=None, 
                    docked_sdf_block=best_sdf,
                    status="ok",
                ))

    return results


def _refine_pose_worker(
    result: _FramePoseResult,
    protein_pdb_path: str,
    docked_sdf_block: str,
    shell_radius_angstrom: float,
    protein_restraint_k: float,
    refine_iterations: int,
    compute_ie: bool,
) -> _FramePoseResult:
    """Worker: Locally minimise one GNINA-docked pose with OpenMM."""
    from draco.refinement import refine_docked_pose
    ref = refine_docked_pose(
        protein_pdb_path,
        docked_sdf_block,
        shell_radius_angstrom=shell_radius_angstrom,
        protein_restraint_k_kcal_per_mol_A2=protein_restraint_k,
        max_iterations=refine_iterations,
        platform_name="CPU",
        compute_interaction_energy=compute_ie,
    )
    import dataclasses
    return dataclasses.replace(
        result,
        initial_energy_kj_per_mol=ref.initial_energy_kj_per_mol,
        final_energy_kj_per_mol=ref.final_energy_kj_per_mol,
        interaction_energy_kj_per_mol=ref.interaction_energy_kj_per_mol,
        ligand_rmsd_from_dock_angstrom=ref.ligand_rmsd_from_dock_angstrom,
        protein_atoms_flexible=ref.protein_atoms_flexible,
        protein_atoms_restrained=ref.protein_atoms_restrained,
        refined_complex_pdb=ref.refined_complex_pdb,
        status=ref.status if ref.status != "ok" else "refined",
        error=ref.error,
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.monotonic()

    # ── Resource Detection ───────────────────────────────────────────────────
    num_gpus, gpu_ids, gpu_source = _detect_gpus()
    num_cpus, cpu_source = _detect_cpus()

    # ── Resource Allocation ──────────────────────────────────────────────────
    # Phase 1: Pocket detection – CPU-only, concurrent with MD.
    # Reserve 1 CPU for the MD orchestrator process.
    num_pocket_workers = max(1, num_cpus - 1)

    # Phase 2: Docking – GNINA is the throughput bottleneck.
    # GNINA has two sequential phases per ligand:
    #   1. Vina sampling  (CPU)  – scales with --cpu threads, up to --exhaustiveness
    #   2. CNN scoring    (GPU)  – brief GPU burst (rescore mode)
    #
    # Optimal strategy: 2 workers per GPU so that one worker can Vina-sample on
    # CPU while the other CNN-scores on GPU — overlapping CPU and GPU work.
    # Each worker gets num_cpus / num_workers CPU threads for Vina sampling,
    # which is far more productive than 1 thread per worker.
    #
    # Derivation:
    #   workers_per_gpu = 2   (pipeline overlap; more adds diminishing returns)
    #   max_docking_workers   = max(1, num_gpus) * workers_per_gpu
    #   gnina_cpu_threads     = max(1, num_cpus // max_docking_workers)
    #
    # --max-docking-workers and --gnina-cpu CLI overrides take full precedence.
    _eff_gpus = max(1, num_gpus)           # treat 0-GPU nodes as "1 slot"
    _default_workers = _eff_gpus * 2       # 2 workers pipelined per GPU
    max_docking_workers = (
        args.max_docking_workers
        if args.max_docking_workers is not None
        else _default_workers
    )
    gnina_cpu_threads = (
        args.gnina_cpu
        if args.gnina_cpu is not None
        else max(1, num_cpus // max_docking_workers)
    )

    # Build the GPU assignment list for docking workers (round-robin).
    # If no GPUs, gpu_assignment is all None → workers run GNINA without a
    # CUDA_VISIBLE_DEVICES override (GNINA will try to use GPU 0 if present,
    # or fall back to CPU).
    if num_gpus > 0:
        gpu_assignment = [gpu_ids[i % num_gpus] for i in range(max_docking_workers)]
    else:
        gpu_assignment = [None] * max_docking_workers

    # GPU 0 is used for MD (OpenMM CUDA). If multi-GPU and docking workers are
    # being distributed, note this so users are aware MD and some docking share GPU 0.
    md_gpu_desc = f"GPU {gpu_ids[0]} (CUDA)" if num_gpus > 0 else f"{args.platform_name} (no GPU detected)"

    print("\n" + "═" * 70)
    print("RESOURCE DETECTION")
    print(f"  GPUs detected : {num_gpus}  (via {gpu_source})"
          + (f"  →  IDs: {gpu_ids}" if gpu_ids else ""))
    print(f"  CPUs detected : {num_cpus}  (via {cpu_source})")
    print()
    print("RESOURCE ALLOCATION")
    print(f"  Dynamics (MD)      : 1 process · {md_gpu_desc}")
    print(f"  Pocket detection   : {num_pocket_workers} CPU workers (concurrent with MD)")
    print(f"  Docking (GNINA)    : {max_docking_workers} workers · {gnina_cpu_threads} thread(s) each"
          + (f"  ·  GPUs: {sorted(set(g for g in gpu_assignment if g is not None))}"
             if num_gpus > 0 else "  ·  CPU-only"))
    if num_gpus > 1:
        from collections import Counter
        gpu_counts = Counter(g for g in gpu_assignment if g is not None)
        print(f"    GPU round-robin  : "
              + "  ".join(f"GPU {g} → {n} workers" for g, n in sorted(gpu_counts.items())))
    print("═" * 70)

    # ── [1/4] Explicit-solvent MD ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"[1/4]  Explicit-solvent MD  (dynamics engine)  [{_ts()}]")
    print("═" * 70)

    # Shared state for analysis
    results_lock = threading.Lock()
    ranking_metric = args.sar_metric if args.mode == "sar" else args.scoring_method
    top_heap = _TopKHeap(k=args.top_k, ranking_metric=ranking_metric)
    all_rows: list[dict[str, Any]] = []
    summary_fieldnames = _FIELDNAMES_SAR if args.mode == "sar" else _FIELDNAMES_SINGLE

    # Progress counters for docking (updated inside on_dock_done under results_lock)
    _dock_progress: dict[str, int] = {"frames_done": 0, "poses_ok": 0, "poses_err": 0}
    _dock_total_frames: list[int] = [0]   # set just before Phase 2 loop
    _dock_phase_start: list[float] = [0.0]
    
    summary_csv_path = outdir / "summary.csv"
    csv_fh = summary_csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=summary_fieldnames)
    csv_writer.writeheader()
    csv_fh.flush()

    # Ligand Preparation goes here (we can preserve from the old logic)
    # ── Ligand preparation ───────────────────────────────────────────────────
    ligands_dir = outdir / "ligands"
    ligands_dir.mkdir(exist_ok=True)
    docking_output_dir = outdir / "docking_output"
    docking_output_dir.mkdir(exist_ok=True)

    # name_map: state_name → parent_ligand_name
    name_map: dict[str, str] = {}
    parent_smiles_map: dict[str, str] = {}
    ligand_sdf_paths: dict[str, str] = {}

    ligand_manifest_path = outdir / "ligand_prep_manifest.json"

    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # NOTE: ligand_preparation currently uses these defaults when Draco doesn't
    # explicitly pass protonation parameters.
    _DEFAULT_PROTONATION = {
        "ph_min": 6.4,
        "ph_max": 8.4,
        "max_variants": 8,
        "precision": 1.0,
    }
    _DEFAULT_PREP = {
        "num_conformers": args.num_conformers,
        "prune_rms_threshold": args.prune_rms_threshold,
        "random_seed": 0xF00D,
        "optimize": True,
        "max_iterations": 200,
        "energy_cutoff": args.energy_cutoff,
    }

    cache_hit = False
    cached_manifest: dict[str, Any] | None = None
    if ligand_manifest_path.is_file():
        try:
            cached_manifest = json.loads(ligand_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cached_manifest = None

    if args.ligand_csv:
        csv_columns: dict[str, str] = {
            "name": args.csv_name_column,
            "smiles": args.csv_smiles_column,
        }
        if args.mode == "sar":
            csv_columns["active"] = args.csv_activity_column
        ligand_signature = {
            "mode": args.mode,
            "ligand_csv_sha256": _sha256_file(Path(args.ligand_csv)),
            "csv_columns": csv_columns,
            "protonation": _DEFAULT_PROTONATION,
            "prep": _DEFAULT_PREP,
        }
    else:
        ligand_signature = {
            "mode": "single",
            "ligand_smiles_sha256": hashlib.sha256(args.ligand_smiles.encode("utf-8")).hexdigest(),
            "ligand_name": args.ligand_name,
            "protonation": _DEFAULT_PROTONATION,
            "prep": _DEFAULT_PREP,
        }

    if cached_manifest and cached_manifest.get("signature") == ligand_signature:
        expected_state_names: list[str] = []
        expected_state_names.extend(cached_manifest.get("active_state_names", []))
        expected_state_names.extend(cached_manifest.get("inactive_state_names", []))
        expected_state_names.extend(cached_manifest.get("screening_state_names", []))
        cache_hit = all((ligands_dir / f"{sn}.sdf").is_file() for sn in expected_state_names)

    if cache_hit and cached_manifest:
        name_map = dict(cached_manifest.get("name_map", {}))
        parent_smiles_map = dict(cached_manifest.get("parent_smiles_map", {}))
        active_names = list(cached_manifest.get("active_parent_names", []))
        inactive_names = list(cached_manifest.get("inactive_parent_names", []))
        active_state_names = list(cached_manifest.get("active_state_names", []))
        inactive_state_names = list(cached_manifest.get("inactive_state_names", []))
        screening_state_names = list(cached_manifest.get("screening_state_names", []))
        for sn in active_state_names + inactive_state_names + screening_state_names:
            ligand_sdf_paths[sn] = str((ligands_dir / f"{sn}.sdf").resolve())
        print(
            f"\n[2/4]  Preparing ligands …  [{_ts()}]  (cache hit)  "
            f"{len(ligand_sdf_paths)} SDFs reused from {ligands_dir}/"
        )
    else:
        _ligand_prep_start = time.monotonic()
        print(f"\n[2/4]  Preparing ligands …  [{_ts()}]")
        streamed_sdf_count = 0

        def _on_prepared_ligand(prep: PreparedLigand, _parent_name: str) -> None:
            nonlocal streamed_sdf_count
            sdf_path = write_ligand_sdf(prep, ligands_dir / f"{prep.name}.sdf")
            ligand_sdf_paths[prep.name] = str(sdf_path.resolve())
            streamed_sdf_count += 1
            # Emit occasional heartbeat logs for long runs.
            if streamed_sdf_count == 1 or streamed_sdf_count % 100 == 0:
                print(f"  Prepared+written {streamed_sdf_count} ligands so far …")
                sys.stdout.flush()

        if args.mode == "sar":
            actives, inactives, name_map = load_compound_csv(
                args.ligand_csv, num_conformers=args.num_conformers,
                prune_rms_threshold=args.prune_rms_threshold,
                energy_cutoff=args.energy_cutoff,
                name_column=args.csv_name_column,
                smiles_column=args.csv_smiles_column,
                activity_column=args.csv_activity_column,
                on_prepared=_on_prepared_ligand,
            )
            all_compounds = actives + inactives
            # active_names / inactive_names are PARENT-level names
            active_names = sorted({name_map[l.name] for l in actives})
            inactive_names = sorted({name_map[l.name] for l in inactives})
            print(
                f"  Loaded {len(active_names)} actives, "
                f"{len(inactive_names)} inactives from {args.ligand_csv}"
            )
        elif args.mode == "screening":
            all_compounds, name_map = load_screening_csv(
                args.ligand_csv,
                num_conformers=args.num_conformers,
                prune_rms_threshold=args.prune_rms_threshold,
                energy_cutoff=args.energy_cutoff,
                name_column=args.csv_name_column,
                smiles_column=args.csv_smiles_column,
                on_prepared=_on_prepared_ligand,
            )
            active_names = []
            inactive_names = []
            print(f"  Loaded {len(all_compounds)} compounds from {args.ligand_csv}")
        else:
            prep = prepare_ligand_from_smiles(
                args.ligand_smiles, name=args.ligand_name,
                num_conformers=args.num_conformers,
                prune_rms_threshold=args.prune_rms_threshold,
                energy_cutoff=args.energy_cutoff,
            )
            all_compounds = [prep]
            name_map[prep.name] = args.ligand_name
            active_names = []
            inactive_names = []
            n_confs = len(prep.conformers)
            print(
                f"  Single compound mode: {args.ligand_name} "
                f"({n_confs} conformers within {args.energy_cutoff} kcal/mol cutoff)"
            )

        # parent_ligand_name -> canonical_smiles map for reporting.
        parent_smiles_map = {
            name_map.get(compound.name, compound.name): compound.canonical_smiles
            for compound in all_compounds
        }

        # For CSV modes, SDFs are stream-written via on_prepared callback.
        # Single-ligand mode still writes here.
        if args.mode == "single":
            ligand_sdf_paths = write_ligands_for_docking(all_compounds, ligands_dir)
        print(f"  Written {len(ligand_sdf_paths)} SDF files to {ligands_dir}/  (elapsed {_elapsed(_ligand_prep_start)})")
        sys.stdout.flush()

        # Write cache manifest so subsequent runs can skip ligand preparation.
        manifest: dict[str, Any] = {
            "signature": ligand_signature,
            "name_map": name_map,
            "parent_smiles_map": parent_smiles_map,
            "active_parent_names": active_names,
            "inactive_parent_names": inactive_names,
        }
        if args.mode == "sar":
            manifest["active_state_names"] = [l.name for l in actives]
            manifest["inactive_state_names"] = [l.name for l in inactives]
            manifest["screening_state_names"] = []
        elif args.mode == "screening":
            manifest["active_state_names"] = []
            manifest["inactive_state_names"] = []
            manifest["screening_state_names"] = [c.name for c in all_compounds]
        else:
            manifest["active_state_names"] = [c.name for c in all_compounds]
            manifest["inactive_state_names"] = []
            manifest["screening_state_names"] = []
        ligand_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Prepare initial protein structure
    _prot_prep_start = time.monotonic()
    print(f"  Preparing initial protein (PDBFixer + H) …  [{_ts()}]")
    initial_prepared = prepare_protein(args.protein_pdb, ph=args.ph, output_dir=str(outdir))
    print(f"  Protein ready  (elapsed {_elapsed(_prot_prep_start)})", flush=True)
    _initial_buf = io.StringIO()
    PDBFile.writeFile(initial_prepared.topology, initial_prepared.positions, _initial_buf)
    initial_pdb_string = _initial_buf.getvalue()

    print("\n" + "═" * 70)
    print(f"[3/4]  Execution Pipeline  [{_ts()}]")
    print("═" * 70)

    rolling_best_energy: list[float | None] = [None]
    
    def on_dock_done(future):
        try:
            frame_results = future.result()
            if not frame_results:
                # Still count the frame as done so the progress line updates
                with results_lock:
                    _dock_progress["frames_done"] += 1
                    _print_dock_progress(_dock_progress, _dock_total_frames[0], _dock_phase_start[0])
                return
            for r in frame_results:
                with results_lock:
                    if r.status in ("ok", "refined"):
                        _dock_progress["poses_ok"] += 1
                        heap_updated = top_heap.push(r)
                        row = _to_row_sar(r) if args.mode == "sar" else _to_row_single(r)
                        if args.mode == "sar":
                            metric_val = getattr(r, ranking_metric, None)
                            score_str = f"{ranking_metric}={metric_val:.3f}" if metric_val is not None else "sar=NA"
                        else:
                            metric_val = getattr(r, ranking_metric, None)
                            score_str = (
                                f"{ranking_metric}={metric_val:.3f}"
                                if metric_val is not None
                                else f"{ranking_metric}=NA"
                            )
                        print(f"  [dock] frame={r.frame_index}  pocket={r.pocket_id}  {score_str}")
                        best = top_heap.current_best()
                        if best is not None and args.top_k > 0:
                            prev = rolling_best_energy[0]
                            rank_score = top_heap._rank_score(best)
                            if prev is None or rank_score > prev:
                                rolling_best_energy[0] = rank_score
                                best_str = f"{ranking_metric}={rank_score:.3f}"
                                print(f"  [top-{args.top_k}] NEW BEST {best_str} | frame={best.frame_index} pocket={best.pocket_id} ligand={best.ligand_name}")
                        if heap_updated and args.top_k > 0:
                            top_now = top_heap.sorted_best()
                            # In SAR mode, export pose models sorted by CNN_VS (more intuitive for pose inspection),
                            # even if the *selection* criterion is a SAR metric like AUC/enrichment.
                            if args.mode == "sar":
                                top_now = sorted(top_now, key=lambda rr: rr.cnn_vs, reverse=True)
                            _write_multimodel_pdb(outdir / args.top_output, top_now)
                    else:
                        _dock_progress["poses_err"] += 1
                        print(f"  [dock] FAILED frame={r.frame_index} pocket={r.pocket_id}: {r.error[:120]}")
                        if args.mode == "sar":
                            row = {
                                "status": "error",
                                "frame_index": r.frame_index,
                                "frame_time_ps": f"{r.frame_time_ps:.3f}",
                                "pocket_id": r.pocket_id,
                                "pocket_score": "",
                                "best_active_ligand_name": r.ligand_name,
                                "auc_roc": "",
                                "auc_pr": "",
                                "n_actives": "",
                                "n_inactives": "",
                                "active_mean_score": "",
                                "inactive_mean_score": "",
                                "active_std_score": "",
                                "inactive_std_score": "",
                                "active_min_score": "",
                                "active_max_score": "",
                                "inactive_min_score": "",
                                "inactive_max_score": "",
                                "active_best_score": "",
                                "inactive_best_score": "",
                                "overall_min_score": "",
                                "overall_max_score": "",
                                "mean_rank_active_minus_inactive": "",
                                "enrichment_1pct": "",
                                "enrichment_5pct": "",
                                "enrichment_10pct": "",
                                "initial_energy_kj_per_mol": "",
                                "final_energy_kj_per_mol": "",
                                "interaction_energy_kj_per_mol": "",
                                "ligand_rmsd_from_dock_angstrom": "",
                                "protein_atoms_flexible": "",
                                "protein_atoms_restrained": "",
                                "error": r.error,
                            }
                        else:
                            row = {
                                "status": "error",
                                "frame_index": r.frame_index,
                                "frame_time_ps": f"{r.frame_time_ps:.3f}",
                                "pocket_id": r.pocket_id,
                                "pocket_score": "",
                                "ligand_name": r.ligand_name,
                                "smiles": r.smiles,
                                "cnn_vs": "",
                                "cnn_affinity": "",
                                "cnn_score": "",
                                "vina_score": "",
                                "auc_roc": "",
                                "initial_energy_kj_per_mol": "",
                                "final_energy_kj_per_mol": "",
                                "interaction_energy_kj_per_mol": "",
                                "ligand_rmsd_from_dock_angstrom": "",
                                "protein_atoms_flexible": "",
                                "protein_atoms_restrained": "",
                                "error": r.error,
                            }
                    all_rows.append(row)
                    csv_writer.writerow(row)
                    csv_fh.flush()
                # One progress line per frame (not per pocket) — update after last result of this future batch
            with results_lock:
                _dock_progress["frames_done"] += 1
                _print_dock_progress(_dock_progress, _dock_total_frames[0], _dock_phase_start[0])
            sys.stdout.flush()
        except Exception as exc:
            import traceback
            print(f"  [ERROR] dock worker crashed: {exc}\n{traceback.format_exc()}")
            with results_lock:
                _dock_progress["frames_done"] += 1
                _print_dock_progress(_dock_progress, _dock_total_frames[0], _dock_phase_start[0])
            sys.stdout.flush()
            
    # PHASE 1: MD + CPU Pocketeer Streaming
    frame_dir = outdir / "frames"
    frame_dir.mkdir(exist_ok=True)
    dynamics_frames = []
    
    if "dynamics" in args.steps:
        _phase1_start = time.monotonic()
        phase_tag = "BioEmu" if args.dynamics_backend == "bioemu" else "MD"
        phase_title = (
            "BioEmu sampling + Pocket Detection"
            if args.dynamics_backend == "bioemu"
            else "MD + Pocket Detection"
        )
        print(f"  [Phase 1] {phase_title}  [{_ts()}]  —  {num_pocket_workers} CPU pocket workers")
        sys.stdout.flush()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_pocket_workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as executor:
            active_pocket_futures = []
            
            def submit_pocket_job(pdb_str, pdb_path, f_idx, f_time):
                if "pocket" not in args.steps:
                    return
                future = executor.submit(
                    _dock_frame_worker,
                    frame_index=f_idx, frame_time_ps=f_time, protein_pdb_path=pdb_path, protein_pdb_string=pdb_str,
                    ligand_sdf_paths={}, name_map={}, parent_smiles_map={}, active_names=[], inactive_names=[],
                    pocket_score_threshold=args.pocket_score_threshold, gnina_binary=args.gnina_binary,
                    exhaustiveness=args.exhaustiveness, num_modes=args.num_modes, cnn_scoring="none",
                    gnina_seed=args.gnina_seed, dry_run=args.dry_run, scoring_method=args.scoring_method,
                    sar_metric=args.sar_metric, docking_output_dir=None, gnina_timeout_seconds=args.gnina_timeout_seconds,
                    steps=["pocket"], project_output_dir=str(outdir)
                )
                active_pocket_futures.append(future)

            initial_pdb_path = str(frame_dir / "frame_0000.pdb")
            (frame_dir / "frame_0000.pdb").write_text(initial_pdb_string)
            submit_pocket_job(initial_pdb_string, initial_pdb_path, 0, 0.0)
            dynamics_frames.append((0, 0.0, initial_pdb_path, initial_pdb_string))
            
            def frame_callback(frame):
                f_path = frame_dir / f"frame_{frame.frame_index:04d}.pdb"
                f_path.write_text(frame.protein_pdb_string)
                submit_pocket_job(frame.protein_pdb_string, str(f_path), frame.frame_index, frame.simulation_time_ps)
                dynamics_frames.append((frame.frame_index, frame.simulation_time_ps, str(f_path), frame.protein_pdb_string))
                print(
                    f"  [{phase_tag}] frame={frame.frame_index}  t={frame.simulation_time_ps:.1f} ps  "
                    f"frames_so_far={len(dynamics_frames)}  [{_ts()}]  elapsed {_elapsed(_phase1_start)}",
                    flush=True,
                )

            if args.dynamics_backend == "bioemu":
                dynamics_result = run_bioemu_sampling(
                    initial_prepared,
                    ph=args.ph,
                    num_samples=args.bioemu_num_samples,
                    output_dir=outdir,
                    save_trajectory=not args.no_trajectory,
                    frame_callback=frame_callback,
                    verbose=True,
                )
            else:
                dynamics_result = run_dynamics(
                    initial_prepared, ph=args.ph, box_padding_nm=args.box_padding_nm,
                    ionic_strength_molar=args.ionic_strength, nvt_steps=args.nvt_steps,
                    npt_steps=args.npt_steps, production_steps=args.production_steps,
                    temperature_kelvin=args.temperature_kelvin, friction_per_ps=args.friction_per_ps,
                    timestep_fs=args.timestep_fs, report_interval_steps=args.report_interval_steps,
                    rmsd_threshold_angstrom=args.rmsd_threshold_angstrom, platform_name=args.platform_name,
                    cuda_precision=args.cuda_precision, output_dir=outdir,
                    save_trajectory=not args.no_trajectory, water_model=args.water_model,
                    frame_callback=frame_callback, verbose=True
                )

            dynamics_sim_time = dynamics_result.simulation_time_ps
            dynamics_frames_len = len(dynamics_result.frames)
            dynamics_trajectory = dynamics_result.trajectory_dcd
            dynamics_topology_pdb = dynamics_result.topology_pdb

            if args.dynamics_backend == "bioemu":
                print(
                    f"\n  [Phase 1] BioEmu complete  [{_ts()}]  elapsed {_elapsed(_phase1_start)}  —  "
                    f"{len(dynamics_result.frames)} samples,  {len(dynamics_frames)} frames (including initial)"
                    f"\n  Waiting for any remaining pocket analysis workers…"
                )
            else:
                print(
                    f"\n  [Phase 1] MD complete  [{_ts()}]  elapsed {_elapsed(_phase1_start)}  —  "
                    f"{dynamics_result.simulation_time_ps:.1f} ps,  {len(dynamics_frames)} frames"
                    f"\n  Waiting for any remaining pocket analysis workers…"
                )
            sys.stdout.flush()
            concurrent.futures.wait(active_pocket_futures)
            for f in active_pocket_futures:
                f.result() # Surface any exceptions
    else:
        dynamics_sim_time = 0.0
        dynamics_frames_len = 0
        dynamics_trajectory = None
        dynamics_topology_pdb = None
        
        # Ensure initial frame is present in frames directory even if dynamics is not run
        initial_pdb_path = frame_dir / "frame_0000.pdb"
        if not initial_pdb_path.exists():
            initial_pdb_path.write_text(initial_pdb_string)
            
        pdb_files = sorted(frame_dir.glob("*.pdb"))
        if not pdb_files:
            print(f"  Error: No PDB files found in {frame_dir}. Cannot run docking without frames.")
            return
        print(f"  [MD Skipped]  [{_ts()}]  Found {len(pdb_files)} frame PDB files for downstream steps.")
        sys.stdout.flush()
        for pdb_path in pdb_files:
            pdb_str = pdb_path.read_text()
            import re
            m = re.fullmatch(r"frame_(\d+)\.pdb", pdb_path.name)
            if not m:
                # Ignore helper/auxiliary PDBs (e.g., frame_initial_input.pdb) so
                # --dock-filter frame indices map one-to-one to actual frame files.
                continue
            f_idx = int(m.group(1))
            dynamics_frames.append((f_idx, 0.0, str(pdb_path), pdb_str))
        if not dynamics_frames:
            print(
                f"  Error: No frame_####.pdb files found in {frame_dir}. "
                "Cannot run docking without valid frame snapshots."
            )
            return
        
        # We still need to run pocket detection if it was passed without dynamics
        if "pocket" in args.steps:
            _pocket_only_start = time.monotonic()
            print(f"  [Phase 1] Standalone Pocket Detection  [{_ts()}]  —  {num_pocket_workers} CPU workers, {len(dynamics_frames)} frames")
            sys.stdout.flush()
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_pocket_workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as executor:
                pocket_futures = []
                for f_idx, f_time, p_path, p_str in dynamics_frames:
                    future = executor.submit(
                        _dock_frame_worker,
                        frame_index=f_idx, frame_time_ps=f_time, protein_pdb_path=p_path, protein_pdb_string=p_str,
                        ligand_sdf_paths={}, name_map={}, parent_smiles_map={}, active_names=[], inactive_names=[],
                        pocket_score_threshold=args.pocket_score_threshold, gnina_binary=args.gnina_binary,
                        exhaustiveness=args.exhaustiveness, num_modes=args.num_modes, cnn_scoring="none",
                        gnina_seed=args.gnina_seed, dry_run=args.dry_run, scoring_method=args.scoring_method,
                        sar_metric=args.sar_metric, docking_output_dir=None, gnina_timeout_seconds=args.gnina_timeout_seconds,
                        steps=["pocket"], project_output_dir=str(outdir)
                    )
                    pocket_futures.append(future)
                concurrent.futures.wait(pocket_futures)
            print(f"  [Phase 1] Pocket detection complete  [{_ts()}]  elapsed {_elapsed(_pocket_only_start)}", flush=True)


    # PHASE 2A: Docking (GPU heavy; --dock-filter applies only here)
    if "docking" in args.steps:
        dock_filter: dict[int, set[int] | None] | None = None
        if args.dock_filter:
            dock_filter = _parse_dock_filter(args.dock_filter)
            filter_desc = ", ".join(
                f"frame {fi} (all pockets)" if pi is None else f"frame {fi} pockets {sorted(pi)}"
                for fi, pi in sorted(dock_filter.items())
            )
            print(f"  [dock-filter] Restricting to: {filter_desc}")

        frames_to_dock = [
            (f_idx, f_time, p_path, p_str)
            for f_idx, f_time, p_path, p_str in dynamics_frames
            if dock_filter is None or f_idx in dock_filter
        ]
        _dock_progress.update({"frames_done": 0, "poses_ok": 0, "poses_err": 0})
        _dock_total_frames[0] = len(frames_to_dock)
        _dock_phase_start[0] = time.monotonic()

        print(f"\n  [Phase 2A] Docking  [{_ts()}]  —  "
              f"{_dock_total_frames[0]} frames queued  |"
              f"  {max_docking_workers} workers × {gnina_cpu_threads} thread(s)  |"
              f"  GPU(s): {sorted(set(g for g in gpu_assignment if g is not None)) if num_gpus > 0 else 'CPU-only'}")
        print(f"  {'─' * 66}")

        def on_docking_only_done(future):
            try:
                frame_results = future.result()
                with results_lock:
                    _dock_progress["frames_done"] += 1
                    if frame_results:
                        for r in frame_results:
                            if r.status in ("ok", "refined"):
                                _dock_progress["poses_ok"] += 1
                            else:
                                _dock_progress["poses_err"] += 1
                                print(f"  [dock] FAILED frame={r.frame_index} pocket={r.pocket_id}: {r.error[:120]}")
                    _print_dock_progress(_dock_progress, _dock_total_frames[0], _dock_phase_start[0])
                sys.stdout.flush()
            except Exception as exc:
                import traceback
                print(f"  [ERROR] docking worker crashed: {exc}\n{traceback.format_exc()}")
                with results_lock:
                    _dock_progress["frames_done"] += 1
                    _print_dock_progress(_dock_progress, _dock_total_frames[0], _dock_phase_start[0])
                sys.stdout.flush()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_docking_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
        ) as executor:
            dock_futures = []
            worker_index = 0
            for f_idx, f_time, p_path, p_str in frames_to_dock:
                pocket_filter_for_frame = dock_filter[f_idx] if dock_filter is not None else None
                assigned_gpu = gpu_assignment[worker_index % max_docking_workers] if gpu_assignment else None
                worker_index += 1
                future = executor.submit(
                    _dock_frame_worker,
                    frame_index=f_idx, frame_time_ps=f_time, protein_pdb_path=p_path, protein_pdb_string=p_str,
                    ligand_sdf_paths=ligand_sdf_paths, name_map=name_map, parent_smiles_map=parent_smiles_map, active_names=active_names, inactive_names=inactive_names,
                    pocket_score_threshold=args.pocket_score_threshold, gnina_binary=args.gnina_binary,
                    exhaustiveness=args.exhaustiveness, num_modes=args.num_modes, cnn_scoring=args.cnn_scoring,
                    gnina_seed=args.gnina_seed, dry_run=args.dry_run, scoring_method=args.scoring_method,
                    sar_metric=args.sar_metric, docking_output_dir=str(docking_output_dir), gnina_timeout_seconds=args.gnina_timeout_seconds,
                    steps=["docking"], project_output_dir=str(outdir),
                    pocket_filter=pocket_filter_for_frame,
                    gnina_cpu_threads=gnina_cpu_threads,
                    gpu_id=assigned_gpu,
                )
                future.add_done_callback(on_docking_only_done)
                dock_futures.append(future)
            concurrent.futures.wait(dock_futures)

    # PHASE 2B: Scoring (scan all existing docking_output frame/pocket folders)
    if "scoring" in args.steps:
        import re
        frame_pocket_dirs = sorted((outdir / "docking_output").glob("frame*_pocket*"))
        pockets_by_frame: dict[int, set[int]] = {}
        for d in frame_pocket_dirs:
            if not d.is_dir():
                continue
            m = re.fullmatch(r"frame(\d+)_pocket(\d+)", d.name)
            if not m:
                continue
            f_idx = int(m.group(1))
            p_idx = int(m.group(2))
            pockets_by_frame.setdefault(f_idx, set()).add(p_idx)

        frames_to_score: list[tuple[int, float, str, str, set[int] | None]] = []
        for f_idx, pocket_ids in sorted(pockets_by_frame.items()):
            frame_pdb_path = frame_dir / f"frame_{f_idx:04d}.pdb"
            if not frame_pdb_path.is_file():
                # Keep scoring robust for partial outputs; skip frames lacking a local snapshot.
                continue
            frame_pdb_str = frame_pdb_path.read_text()
            frames_to_score.append((f_idx, 0.0, str(frame_pdb_path), frame_pdb_str, pocket_ids))

        _dock_progress.update({"frames_done": 0, "poses_ok": 0, "poses_err": 0})
        _dock_total_frames[0] = len(frames_to_score)
        _dock_phase_start[0] = time.monotonic()

        print(f"\n  [Phase 2B] Scoring  [{_ts()}]  —  "
              f"{_dock_total_frames[0]} frames discovered in docking_output  |"
              f"  filter scope: ALL docking_output entries")
        print(f"  {'─' * 66}")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_docking_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
        ) as executor:
            score_futures = []
            worker_index = 0
            for f_idx, f_time, p_path, p_str, scored_pockets in frames_to_score:
                assigned_gpu = gpu_assignment[worker_index % max_docking_workers] if gpu_assignment else None
                worker_index += 1
                future = executor.submit(
                    _dock_frame_worker,
                    frame_index=f_idx, frame_time_ps=f_time, protein_pdb_path=p_path, protein_pdb_string=p_str,
                    ligand_sdf_paths=ligand_sdf_paths, name_map=name_map, parent_smiles_map=parent_smiles_map, active_names=active_names, inactive_names=inactive_names,
                    pocket_score_threshold=args.pocket_score_threshold, gnina_binary=args.gnina_binary,
                    exhaustiveness=args.exhaustiveness, num_modes=args.num_modes, cnn_scoring=args.cnn_scoring,
                    gnina_seed=args.gnina_seed, dry_run=args.dry_run, scoring_method=args.scoring_method,
                    sar_metric=args.sar_metric, docking_output_dir=str(docking_output_dir), gnina_timeout_seconds=args.gnina_timeout_seconds,
                    steps=["scoring"], project_output_dir=str(outdir),
                    pocket_filter=scored_pockets,
                    gnina_cpu_threads=gnina_cpu_threads,
                    gpu_id=assigned_gpu,
                )
                future.add_done_callback(on_dock_done)
                score_futures.append(future)
            concurrent.futures.wait(score_futures)

    csv_fh.close()

    # Reorder summary rows by the same ranking used for top-k selection.
    # Successful rows first, then descending ranking score.
    def _summary_rank_key(row: dict[str, Any]) -> tuple[int, float]:
        try:
            val = float(row.get(ranking_metric, ""))
            if ranking_metric == "vina_score":
                val = -val
        except (TypeError, ValueError):
            val = float("-inf")

        is_error = 1 if str(row.get("status", "")).lower() == "error" else 0
        return (is_error, -val)

    all_rows_sorted = sorted(all_rows, key=_summary_rank_key)
    _write_csv(summary_csv_path, all_rows_sorted, fieldnames=summary_fieldnames)

    if "dynamics" in args.steps and args.dynamics_backend == "bioemu":
        _complete_mid = (
            f"{dynamics_frames_len} BioEmu samples, {len(all_rows)} poses assessed."
        )
    else:
        _complete_mid = (
            f"{dynamics_sim_time:.1f} ps simulated, "
            f"{dynamics_frames_len} frames analyzed, {len(all_rows)} poses assessed."
        )
    print(
        f"\n  ✓ All stages complete  [{_ts()}]  elapsed {_elapsed(pipeline_start)}: "
        f"{_complete_mid}"
    )
    sys.stdout.flush()

    # ── [4/4] Write outputs ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"[4/4]  Finalizing outputs  [{_ts()}]")
    print("═" * 70)
    sys.stdout.flush()

    print(f"  Summary CSV     : {summary_csv_path}  ({len(all_rows)} rows)")

    top_poses = top_heap.sorted_best()
    if top_poses:
        frame_dir = outdir / "frames"
        if args.mode == "sar":
            # For SAR, write the exported pose models ordered by CNN_VS.
            top_poses = sorted(top_poses, key=lambda rr: rr.cnn_vs, reverse=True)
            # Also export ligand-only SDFs:
            # - per-rank: all ligands' best poses for that frame/pocket, sorted by scoring_method
            # - combined: the single best ligand per selected frame/pocket (top-K pockets)
            topk_best_ligand_blocks: list[str] = []
            for rank, r in enumerate(top_poses, start=1):
                topk_best_ligand_blocks.append(r.docked_sdf_block)
                # Copy the pocket-ranked ligand SDF (written by the worker) into a stable topXX filename.
                pocket_ranked = (
                    outdir
                    / "docking_output"
                    / f"frame{r.frame_index:04d}_pocket{r.pocket_id}"
                    / f"ligands_ranked_by_{args.scoring_method}.sdf"
                )
                if pocket_ranked.is_file():
                    dst = outdir / f"top{rank:02d}_frame{r.frame_index:04d}_pocket{r.pocket_id:03d}_ligands_ranked.sdf"
                    dst.write_text(pocket_ranked.read_text(encoding="utf-8"), encoding="utf-8")
                    print(f"  Written         : {dst}")
            # Combined top-K: one best ligand per selected (frame,pocket)
            topk_ligands_path = outdir / f"top{args.top_k}_best_ligands.sdf"
            _write_multimol_sdf(topk_ligands_path, topk_best_ligand_blocks)
            print(f"  Top-{args.top_k} ligands : {topk_ligands_path}")
        if "refinement" not in args.steps:
            print("\n  Writing top-k docking poses without refinement ...")
        else:
            print("\n  Refining top-k poses ...")

        final_top_poses = []
        for i, r in enumerate(top_poses, start=1):
            frame_pdb = str(frame_dir / f"frame_{r.frame_index:04d}.pdb")

            if "refinement" not in args.steps:
                print(f"    Using docked rank {i} (frame {r.frame_index}, pocket {r.pocket_id}) ...")
                try:
                    complex_pdb, complex_cif = _build_complex_from_frame_and_docked_pose(
                        frame_pdb_path=frame_pdb,
                        docked_sdf_block=r.docked_sdf_block,
                        ligand_residue_name=args.ligand_name,
                    )
                    import dataclasses
                    final_top_poses.append(
                        dataclasses.replace(
                            r,
                            refined_complex_pdb=complex_pdb,
                            refined_complex_cif=complex_cif,
                            status="docked",
                        )
                    )
                except Exception as exc:
                    print(f"    Docked-pose export failed for rank {i}: {exc}")
                continue

            print(f"    Refining rank {i}/{len(top_poses)}  [{_ts()}]  (frame {r.frame_index}, pocket {r.pocket_id}) …")
            _refine_start = time.monotonic()
            refined_r = _refine_pose_worker(
                result=r,
                protein_pdb_path=frame_pdb,
                docked_sdf_block=r.docked_sdf_block,
                shell_radius_angstrom=args.shell_radius_angstrom,
                protein_restraint_k=args.protein_restraint_k,
                refine_iterations=args.refine_iterations,
                compute_ie=not args.no_interaction_energy,
            )
            if refined_r.status == "error" or not refined_r.refined_complex_pdb.strip():
                err_head = (refined_r.error or "empty refined PDB").splitlines()[0]
                print(f"    Refinement FAILED rank {i}: {err_head}")
                continue
            print(f"    Rank {i} refined  (elapsed {_elapsed(_refine_start)})  ΔE={refined_r.final_energy_kj_per_mol - refined_r.initial_energy_kj_per_mol:.1f} kJ/mol", flush=True)
            final_top_poses.append(refined_r)

        top_poses = final_top_poses
        if top_poses:
            top_cif_path = outdir / args.top_output
            _write_multiblock_cif(top_cif_path, top_poses)
            print(f"  Top-{args.top_k} CIF      : {top_cif_path}  ({len(top_poses)} blocks)")
            write_per_rank = not (args.mode == "sar" and "refinement" not in args.steps)
            if write_per_rank:
                print(f"  Top-{args.top_k} output   : per-rank CIF files ({len(top_poses)} models)")
                for rank, r in enumerate(top_poses, start=1):
                    if "refinement" not in args.steps:
                        fname = (
                            f"top{rank:02d}_frame{r.frame_index:04d}_"
                            f"pocket{r.pocket_id:03d}_docked_only.cif"
                        )
                    else:
                        ie_tag = (
                            f"{r.interaction_energy_kj_per_mol:.1f}"
                            if r.interaction_energy_kj_per_mol is not None
                            else "noE"
                        )
                        fname = (
                            f"top{rank:02d}_frame{r.frame_index:04d}_"
                            f"pocket{r.pocket_id:03d}_"
                            f"ie{ie_tag.replace('-', 'n')}_kJmol.cif"
                        )
                    cif_text = r.refined_complex_cif
                    if not cif_text.strip() and r.refined_complex_pdb.strip():
                        cif_text = _pdb_text_to_cif(r.refined_complex_pdb)
                    if not cif_text.strip():
                        print(f"  Skipped         : {outdir / fname} (no CIF content)")
                        continue
                    (outdir / fname).write_text(cif_text)
                    print(f"  Written         : {outdir / fname}")
            elif args.mode == "sar":
                print("  SAR mode        : skipping per-rank *_docked_only.cif outputs.")
        else:
            if "refinement" not in args.steps:
                print("  Docked-pose export failed for all top poses; no CIF models written.")
            else:
                print("  Refinement failed for all top poses; no refined CIF models written.")
    else:
        print("  No successful poses – nothing to write.")

    if dynamics_trajectory:
        print(f"  Trajectory       : {dynamics_trajectory}")
        print(f"  Topology PDB     : {dynamics_topology_pdb}")

    print("\n  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _energy_for_topk_ranking(r: _FramePoseResult) -> float:
    """Scalar used for top-k ordering (must match _TopKHeap.push)."""
    v = r.interaction_energy_kj_per_mol
    return v if v is not None else r.final_energy_kj_per_mol


_FIELDNAMES_SINGLE = [
    "status", "frame_index", "frame_time_ps",
    "pocket_id", "pocket_score",
    "ligand_name", "smiles", "cnn_vs", "cnn_affinity", "cnn_score", "vina_score", "auc_roc",
    "initial_energy_kj_per_mol", "final_energy_kj_per_mol",
    "interaction_energy_kj_per_mol",
    "ligand_rmsd_from_dock_angstrom",
    "protein_atoms_flexible", "protein_atoms_restrained",
    "error",
]

# SAR summaries are per (frame, pocket) discrimination metrics.
_FIELDNAMES_SAR = [
    "status", "frame_index", "frame_time_ps",
    "pocket_id", "pocket_score",
    # Representative ligand used to export the pocket's top pose.
    "best_active_ligand_name",
    "auc_roc",
    "auc_pr",
    "n_actives", "n_inactives",
    "active_mean_score", "inactive_mean_score",
    "active_std_score", "inactive_std_score",
    "active_min_score", "active_max_score",
    "inactive_min_score", "inactive_max_score",
    "active_best_score", "inactive_best_score",
    "overall_min_score", "overall_max_score",
    "mean_rank_active_minus_inactive",
    "enrichment_1pct", "enrichment_5pct", "enrichment_10pct",
    "initial_energy_kj_per_mol", "final_energy_kj_per_mol",
    "interaction_energy_kj_per_mol",
    "ligand_rmsd_from_dock_angstrom",
    "protein_atoms_flexible", "protein_atoms_restrained",
    "error",
]


def _to_row_single(r: _FramePoseResult) -> dict[str, Any]:
    return {
        "status": r.status,
        "frame_index": r.frame_index,
        "frame_time_ps": f"{r.frame_time_ps:.3f}",
        "pocket_id": r.pocket_id,
        "pocket_score": f"{r.pocket_score:.3f}",
        "ligand_name": r.ligand_name,
        "smiles": r.smiles or "",
        "cnn_vs": f"{r.cnn_vs:.3f}" if r.cnn_vs != "" else "",
        "cnn_affinity": f"{r.cnn_affinity:.3f}" if r.cnn_affinity != "" else "",
        "cnn_score": f"{r.cnn_score:.3f}" if r.cnn_score != "" else "",
        "vina_score": f"{r.vina_score:.3f}" if r.vina_score != "" else "",
        "auc_roc": f"{r.auc_roc:.4f}" if r.auc_roc is not None and r.auc_roc != "" else "",
        "initial_energy_kj_per_mol": (
            f"{r.initial_energy_kj_per_mol:.2f}"
            if isinstance(r.initial_energy_kj_per_mol, (int, float)) else r.initial_energy_kj_per_mol
        ),
        "final_energy_kj_per_mol": f"{r.final_energy_kj_per_mol:.2f}",
        "interaction_energy_kj_per_mol": (
            f"{r.interaction_energy_kj_per_mol:.2f}"
            if r.interaction_energy_kj_per_mol is not None else ""
        ),
        "ligand_rmsd_from_dock_angstrom": f"{r.ligand_rmsd_from_dock_angstrom:.3f}",
        "protein_atoms_flexible": r.protein_atoms_flexible,
        "protein_atoms_restrained": r.protein_atoms_restrained,
        "error": r.error,
    }


def _to_row_sar(r: _FramePoseResult) -> dict[str, Any]:
    # Enrichment metrics are set dynamically on SAR results via `setattr`.
    e1 = getattr(r, "enrichment_1pct", None)
    e5 = getattr(r, "enrichment_5pct", None)
    e10 = getattr(r, "enrichment_10pct", None)
    auc_pr = getattr(r, "auc_pr", None)
    n_actives = getattr(r, "n_actives", None)
    n_inactives = getattr(r, "n_inactives", None)
    active_mean_score = getattr(r, "active_mean_score", None)
    inactive_mean_score = getattr(r, "inactive_mean_score", None)
    active_std_score = getattr(r, "active_std_score", None)
    inactive_std_score = getattr(r, "inactive_std_score", None)
    active_min_score = getattr(r, "active_min_score", None)
    active_max_score = getattr(r, "active_max_score", None)
    inactive_min_score = getattr(r, "inactive_min_score", None)
    inactive_max_score = getattr(r, "inactive_max_score", None)
    active_best_score = getattr(r, "active_best_score", None)
    inactive_best_score = getattr(r, "inactive_best_score", None)
    overall_min_score = getattr(r, "overall_min_score", None)
    overall_max_score = getattr(r, "overall_max_score", None)
    mean_rank_active_minus_inactive = getattr(r, "mean_rank_active_minus_inactive", None)
    return {
        "status": r.status,
        "frame_index": r.frame_index,
        "frame_time_ps": f"{r.frame_time_ps:.3f}",
        "pocket_id": r.pocket_id,
        "pocket_score": f"{r.pocket_score:.3f}",
        "best_active_ligand_name": r.ligand_name,
        "auc_roc": f"{r.auc_roc:.4f}" if r.auc_roc is not None and r.auc_roc != "" else "",
        "auc_pr": f"{auc_pr:.4f}" if auc_pr is not None and auc_pr != "" else "",
        "n_actives": n_actives if n_actives is not None else "",
        "n_inactives": n_inactives if n_inactives is not None else "",
        "active_mean_score": f"{active_mean_score:.4f}" if active_mean_score is not None and active_mean_score != "" else "",
        "inactive_mean_score": f"{inactive_mean_score:.4f}" if inactive_mean_score is not None and inactive_mean_score != "" else "",
        "active_std_score": f"{active_std_score:.4f}" if active_std_score is not None and active_std_score != "" else "",
        "inactive_std_score": f"{inactive_std_score:.4f}" if inactive_std_score is not None and inactive_std_score != "" else "",
        "active_min_score": f"{active_min_score:.4f}" if active_min_score is not None and active_min_score != "" else "",
        "active_max_score": f"{active_max_score:.4f}" if active_max_score is not None and active_max_score != "" else "",
        "inactive_min_score": f"{inactive_min_score:.4f}" if inactive_min_score is not None and inactive_min_score != "" else "",
        "inactive_max_score": f"{inactive_max_score:.4f}" if inactive_max_score is not None and inactive_max_score != "" else "",
        "active_best_score": f"{active_best_score:.4f}" if active_best_score is not None and active_best_score != "" else "",
        "inactive_best_score": f"{inactive_best_score:.4f}" if inactive_best_score is not None and inactive_best_score != "" else "",
        "overall_min_score": f"{overall_min_score:.4f}" if overall_min_score is not None and overall_min_score != "" else "",
        "overall_max_score": f"{overall_max_score:.4f}" if overall_max_score is not None and overall_max_score != "" else "",
        "mean_rank_active_minus_inactive": f"{mean_rank_active_minus_inactive:.4f}" if mean_rank_active_minus_inactive is not None and mean_rank_active_minus_inactive != "" else "",
        "enrichment_1pct": f"{e1:.4f}" if e1 is not None and e1 != "" else "",
        "enrichment_5pct": f"{e5:.4f}" if e5 is not None and e5 != "" else "",
        "enrichment_10pct": f"{e10:.4f}" if e10 is not None and e10 != "" else "",
        "initial_energy_kj_per_mol": (
            f"{r.initial_energy_kj_per_mol:.2f}"
            if isinstance(r.initial_energy_kj_per_mol, (int, float)) else r.initial_energy_kj_per_mol
        ),
        "final_energy_kj_per_mol": f"{r.final_energy_kj_per_mol:.2f}",
        "interaction_energy_kj_per_mol": (
            f"{r.interaction_energy_kj_per_mol:.2f}"
            if r.interaction_energy_kj_per_mol is not None else ""
        ),
        "ligand_rmsd_from_dock_angstrom": f"{r.ligand_rmsd_from_dock_angstrom:.3f}",
        "protein_atoms_flexible": r.protein_atoms_flexible,
        "protein_atoms_restrained": r.protein_atoms_restrained,
        "error": r.error,
    }



def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_multimol_sdf(path: Path, sdf_blocks: list[str]) -> None:
    """Write a multi-molecule SDF file from SDF blocks.

    Each block should be a complete molecule record (ideally ending with '$$$$').
    """
    # Preserve SD properties (GNINA scores, SMILES, etc.) by writing the raw
    # SDF record text. Do NOT strip the block: stripping can remove the blank
    # line that terminates SD property fields and break parsers.
    out: list[str] = []
    for block in sdf_blocks:
        b = (block or "")
        if not b.strip():
            continue
        b = b.replace("\r\n", "\n")
        # Avoid leading blank lines which can confuse some SDF readers.
        b = b.lstrip("\n")
        if "$$$$" in b:
            head = b.split("$$$$", 1)[0]
        else:
            head = b
        # Ensure record ends with a newline before the delimiter.
        if not head.endswith("\n"):
            head += "\n"
        out.append(head + "$$$$\n")
    path.write_text("".join(out), encoding="utf-8")


def _print_dock_progress(
    progress: dict[str, int],
    total_frames: int,
    phase_start: float,
) -> None:
    """Print a one-line docking progress summary to stdout.

    Called under ``results_lock`` after every frame completes so the log file
    always has an up-to-date status line.  Format::

        [Progress]  7/22 frames  |  143 poses  |  2 errors  |  elapsed 4m 12s  |  ETA ~8m 03s
    """
    done = progress["frames_done"]
    ok   = progress["poses_ok"]
    err  = progress["poses_err"]
    elapsed_s = time.monotonic() - phase_start

    # Estimate time remaining using average frame rate so far
    if done > 0 and total_frames > 0:
        rate = elapsed_s / done          # seconds per frame
        remaining = max(0, (total_frames - done) * rate)
        r_h, r_rem = divmod(int(remaining), 3600)
        r_m, r_s   = divmod(r_rem, 60)
        eta_str = (
            f"{r_h}h {r_m:02d}m {r_s:02d}s"
            if r_h
            else f"{r_m}m {r_s:02d}s"
            if r_m
            else f"{r_s}s"
        )
        eta_part = f"  |  ETA ~{eta_str}"
    else:
        eta_part = ""

    err_part = f"  |  {err} errors" if err else ""
    frac = f"{done}/{total_frames}" if total_frames else str(done)
    print(
        f"  [Progress]  {frac} frames  |  {ok} poses{err_part}"
        f"  |  elapsed {_elapsed(phase_start)}{eta_part}",
        flush=True,
    )



def _write_multimodel_pdb(path: Path, results: list[_FramePoseResult]) -> None:
    lines: list[str] = []
    for rank, r in enumerate(results, start=1):
        ie_str = (
            f"{r.interaction_energy_kj_per_mol:.2f}"
            if r.interaction_energy_kj_per_mol is not None else "NA"
        )
        lines.append(f"MODEL     {rank:4d}")
        lines.append(
            f"REMARK RANK {rank} "
            f"FRAME {r.frame_index} "
            f"TIME_PS {r.frame_time_ps:.3f} "
            f"POCKET_ID {r.pocket_id} "
            f"CNN_AFFINITY {r.cnn_affinity:.4f} "
            f"VINA_SCORE {r.vina_score:.4f} "
            f"INTERACTION_ENERGY_KJ_MOL {ie_str} "
            f"FINAL_ENERGY_KJ_MOL {r.final_energy_kj_per_mol:.2f}"
        )
        for line in r.refined_complex_pdb.splitlines():
            if line.strip() in {"END", "ENDMDL", "MODEL"}:
                continue
            lines.append(line)
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _write_multiblock_cif(path: Path, results: list[_FramePoseResult]) -> None:
    """Write multiple poses as a valid multi-data-block mmCIF file."""
    blocks: list[str] = []
    for rank, r in enumerate(results, start=1):
        cif_text = r.refined_complex_cif
        if not cif_text.strip() and r.refined_complex_pdb.strip():
            cif_text = _pdb_text_to_cif(r.refined_complex_pdb)
        if not cif_text.strip():
            continue

        lines = cif_text.splitlines()
        if not lines:
            continue

        # Ensure each data block has a unique name.
        header = (
            f"data_top{rank:02d}_frame{r.frame_index:04d}_pocket{r.pocket_id:03d}"
        )
        if lines[0].startswith("data_"):
            lines[0] = header
        else:
            lines = [header] + lines
        blocks.append("\n".join(lines).rstrip() + "\n")

    path.write_text("\n".join(blocks))


def _build_complex_from_frame_and_docked_pose(
    *,
    frame_pdb_path: str,
    docked_sdf_block: str,
    ligand_residue_name: str = "LIG",
) -> tuple[str, str]:
    """Build a protein+ligand complex and return (PDB text, mmCIF text)."""
    from rdkit import Chem
    try:
        from openmm import Vec3
        from openmm.app import Topology, Element
    except ImportError:  # pragma: no cover
        from simtk.openmm import Vec3  # type: ignore
        from simtk.openmm.app import Topology, Element  # type: ignore

    protein = PDBFile(str(frame_pdb_path))

    mol_block = docked_sdf_block.split("$$$$")[0].strip()
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromMolBlock(mol_block, removeHs=True, sanitize=True)
    if mol is None:
        raise ValueError("Unable to parse docked SDF block for complex export")

    top = Topology()
    chain = top.addChain(id="L")
    res = top.addResidue(ligand_residue_name[:3], chain, id="1")
    conf = mol.GetConformer(0)
    atoms = []
    positions = []
    for idx, r_atom in enumerate(mol.GetAtoms(), start=1):
        sym = r_atom.GetSymbol()
        elem = Element.getBySymbol(sym)
        atom_name = f"{sym}{idx % 1000:03d}"[:4]
        atoms.append(top.addAtom(atom_name, elem, res, id=str(idx)))
        pos = conf.GetAtomPosition(r_atom.GetIdx())
        positions.append(Vec3(pos.x, pos.y, pos.z))
    for bond in mol.GetBonds():
        top.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])

    modeller = Modeller(protein.topology, protein.positions)
    modeller.add(top, positions * unit.angstrom)

    pdb_out = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, pdb_out, keepIds=True)
    cif_out = io.StringIO()
    PDBxFile.writeFile(modeller.topology, modeller.positions, cif_out, keepIds=True)
    return pdb_out.getvalue(), cif_out.getvalue()


def _pdb_text_to_cif(pdb_text: str) -> str:
    """Convert PDB text to mmCIF text via OpenMM I/O."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb_text)
        tmp_path = f.name
    try:
        pdb = PDBFile(tmp_path)
        out = io.StringIO()
        PDBxFile.writeFile(pdb.topology, pdb.positions, out, keepIds=True)
        return out.getvalue()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
