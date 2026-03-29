"""cli.py – Entry point for the Draco pipeline: MD + GNINA docking.

Pipeline
--------
                  ┌──────────────────────────────────────────────────────────┐
  protein-pdb +   │               dynamics.py (this script calls it)          │
  compounds  ──►  │  prepare_protein → solvate → minimize → NVT/NPT →        │
  .csv or         │  production MD → Cα RMSD-change frame extraction          │
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
                              │  final refinement (OpeNMM)
                              │  (top-K poses only)     │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Output                │
                              │  summary.csv            │
                              │  top5_poses.pdb         │
                              │  trajectory.dcd         │
                              └─────────────────────────┘

Usage (SAR series mode):
    draco --protein-pdb 6pbc-prepared.pdb \\
        --compound-csv compounds.csv \\
        --output-dir dynamics_gnina_output

Usage (single-compound mode):
    draco --protein-pdb 6pbc-prepared.pdb \\
        --ligand-smiles "CN(CC1(CC1)c1ccc(F)cc1)C(=O)..." \\
        --output-dir dynamics_gnina_single
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

import argparse
import concurrent.futures
import csv
import heapq
import io
import threading
import time
import multiprocessing
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from draco.dynamics import DynamicsFrame, DynamicsResult, run_dynamics
from draco.gnina_docking import (
    DockingBox, GninaDockResult, PocketDockResult,
    docking_box_from_pocket, dock_ligands_to_pocket,
)
from draco.ligand_preparation import (
    PreparedLigand, prepare_ligand_from_smiles, prepare_protonation_states,
    load_compound_csv, write_ligand_sdf, write_ligands_for_docking,
)
from draco.final_refinement import RefinementResult, refine_docked_pose
from draco.sar_scoring import SARScoreResult, compute_sar_discrimination
from draco.protein_preparation import PreparedProtein, prepare_protein

import pocketeer as pt

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
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draco: MD → GNINA docking → SAR scoring → top-k refinement."
    )

    # ── Input / output ──────────────────────────────────────────────────────
    parser.add_argument("--protein-pdb", required=True, help="Input protein PDB path")
    # Ligand input — mutually exclusive modes
    lig_grp = parser.add_mutually_exclusive_group(required=True)
    lig_grp.add_argument(
        "--compound-csv",
        help="CSV with columns name,smiles,active (1/0). Enables SAR discrimination scoring."
    )
    lig_grp.add_argument(
        "--ligand-smiles",
        help="Single ligand SMILES. Poses ranked by CNN affinity (no SAR scoring)."
    )
    parser.add_argument("--ligand-name", default="LIG", help="Name for --ligand-smiles compound")
    parser.add_argument("--output-dir",  default="dynamics_gnina_output")
    parser.add_argument("--ph",          type=float, default=7.4)

    # ── MD / dynamics ────────────────────────────────────────────────────────
    parser.add_argument("--platform-name",          default="CUDA")
    parser.add_argument("--cuda-precision",         default="mixed")
    parser.add_argument("--box-padding-nm",         type=float, default=1.0)
    parser.add_argument("--ionic-strength",         type=float, default=0.15)
    parser.add_argument("--nvt-steps",              type=int, default=50_000)
    parser.add_argument("--npt-steps",              type=int, default=50_000)
    parser.add_argument("--production-steps",       type=int, default=2_500_000)
    parser.add_argument("--timestep-fs",            type=float, default=2.0)
    parser.add_argument("--friction-per-ps",        type=float, default=1.0)
    parser.add_argument("--temperature-kelvin",     type=float, default=300.0)
    parser.add_argument("--water-model",            default="tip3pfb")
    parser.add_argument("--report-interval-steps", type=int, default=5_000)
    parser.add_argument("--rmsd-threshold-angstrom", type=float, default=1.5)
    parser.add_argument("--no-trajectory",          action="store_true", default=False)

    # ── Pocketeer ─────────────────────────────────────────────────────────────
    parser.add_argument("--pocket-score-threshold", type=float, default=5.0)
    parser.add_argument("--num-conformers",         type=int,   default=20)

    # ── GNINA docking ─────────────────────────────────────────────────────────
    parser.add_argument("--gnina-binary",    default="gnina",
                        help="Path or name of the gnina binary (default: gnina)")
    parser.add_argument("--exhaustiveness",  type=int, default=8,
                        help="GNINA exhaustiveness (default 8)")
    parser.add_argument("--num-modes",       type=int, default=9,
                        help="Number of docked poses per ligand (default 9)")
    parser.add_argument("--cnn-scoring",     default="rescore",
                        choices=["rescore", "refinement", "none"],
                        help="GNINA CNN scoring mode (default: rescore)")
    parser.add_argument("--gnina-seed",      type=int, default=0)
    parser.add_argument(
        "--gnina-timeout-seconds",
        type=int,
        default=None,
        metavar="N",
        help="Wall-clock limit per GNINA subprocess (seconds). "
        "Omit for no limit (GNINA runs to completion).",
    )

    # ── Final refinement ─────────────────────────────────────────────────────
    parser.add_argument("--shell-radius-angstrom", type=float, default=8.0)
    parser.add_argument("--protein-restraint-k",   type=float, default=10.0)
    parser.add_argument("--refine-iterations",     type=int,   default=500)
    parser.add_argument("--no-interaction-energy", action="store_true", default=False)
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        default=False,
        help="Skip OpenMM refinement and write top poses directly from GNINA docking + frame PDB.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--top-k",   type=int, default=5)
    parser.add_argument("--top-pdb", default="top5_poses.cif")

    # ── Debug ─────────────────────────────────────────────────────────────────
    parser.add_argument("--dry-run",            action="store_true", default=False)
    parser.add_argument("--analyze-frames-only", action="store_true", default=False)
    parser.add_argument("--compile-results-only", action="store_true", default=False,
                        help="Skip MD and docking. Compile top poses directly from existing GNINA output.")
    return parser.parse_args()


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
    auc_roc: float | None        # SAR discrimination AUC (None in single-compound mode)
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


@dataclass
class _TopKHeap:
    k: int
    _heap: list[tuple[float, int, _FramePoseResult]] = field(default_factory=list)
    _counter: int = 0

    def push(self, result: _FramePoseResult) -> bool:
        """Push by primary ranking score (AUC-ROC if available, else CNN affinity)."""
        # Higher AUC = better; higher CNN affinity (pK) = better
        # Normalize to: lower raw_score = better (for min-heap inversion trick)
        if result.auc_roc is not None:
            raw_score = -result.auc_roc          # negate so most negative = worst (pushed out)
        else:
            raw_score = -result.cnn_affinity     # negate pK so tighter binders sort first
        entry = (raw_score, self._counter, result)
        self._counter += 1
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            return True
        elif raw_score < self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            return True
        return False

    def sorted_best(self) -> list[_FramePoseResult]:
        return [r for _, _, r in sorted(self._heap, key=lambda t: t[0])]

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

def _worker_init() -> None:
    """Initializer for ProcessPoolExecutor workers."""
    os.environ["OPENMM_CPU_THREADS"] = "1"
    try:
        import openmm
        import rdkit
        import pocketeer
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
    active_names: list[str],           # parent-level active names
    inactive_names: list[str],         # parent-level inactive names
    pocket_score_threshold: float,
    gnina_binary: str,
    exhaustiveness: int,
    num_modes: int,
    cnn_scoring: str,
    gnina_seed: int,
    dry_run: bool,
    gnina_output_dir: str | None = None,
    gnina_timeout_seconds: int | None = None,
    compile_results_only: bool = False,
) -> list[_FramePoseResult]:
    """Worker: Detect pockets, dock all ligands with GNINA, compute SAR score.

    All protonation states of each parent ligand are docked. Scores are
    aggregated to report the **best score per parent ligand** for SAR
    discrimination and single-compound ranking.

    Returns one _FramePoseResult per (pocket, parent_ligand) pair.
    """
    import io, tempfile
    import biotite.structure.io.pdb as _biotite_pdb
    import pocketeer as pt
    from draco.gnina_docking import docking_box_from_pocket, dock_ligands_to_pocket
    from draco.sar_scoring import compute_sar_discrimination
    from draco.gnina_docking import PocketDockResult

    if dry_run:
        return []

    try:
        frame_atomarray = _biotite_pdb.PDBFile.read(
            io.StringIO(protein_pdb_string)
        ).get_structure(model=1)
        pockets = pt.find_pockets(frame_atomarray)
        pockets = [p for p in pockets
                   if float(getattr(p, "score", 0.0)) > pocket_score_threshold]
        if not pockets:
            return []
    except Exception as exc:
        import traceback
        return [_FramePoseResult(
            frame_index=frame_index, frame_time_ps=frame_time_ps,
            pocket_id=-1, pocket_score=0.0, ligand_name="",
            cnn_affinity=0.0, vina_score=0.0, auc_roc=None,
            status="error", error=f"Pocketeer failed: {exc}\n{traceback.format_exc()}"
        )]

    results: list[_FramePoseResult] = []
    sar_mode = bool(active_names or inactive_names)

    # Build the set of state-level active/inactive names from the name_map
    active_state_names = {sn for sn, pn in name_map.items() if pn in set(active_names)}
    inactive_state_names = {sn for sn, pn in name_map.items() if pn in set(inactive_names)}

    for pocket_idx, pocket in enumerate(pockets):
        pocket_id = int(getattr(pocket, "pocket_id", pocket_idx))
        pocket_score = float(getattr(pocket, "score", 0.0))

        # Create per-pocket output dir for GNINA
        if gnina_output_dir:
            pocket_out = __import__('pathlib').Path(gnina_output_dir) / f"frame{frame_index:04d}_pocket{pocket_id}"
            pocket_out.mkdir(parents=True, exist_ok=True)
            out_dir_str = str(pocket_out)
        else:
            out_dir_str = None

        try:
            box = docking_box_from_pocket(pocket)
            
            if compile_results_only and out_dir_str:
                import draco.gnina_docking as gnina_docking
                pocket_dock = gnina_docking.PocketDockResult(pocket_id=pocket_id, docking_box=box)
                for name, sdf_path in ligand_sdf_paths.items():
                    out_sdf = Path(out_dir_str) / f"{name}.gnina.sdf"
                    if out_sdf.exists():
                        pocket_dock.results[name] = gnina_docking._parse_gnina_sdf(out_sdf.read_text(), ligand_name=name)
                    else:
                        pocket_dock.results[name] = []
            else:
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
                    cpu=1,
                    timeout_seconds=gnina_timeout_seconds,
                    output_dir=out_dir_str,
                    write_gnina_logs=True,
                )
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
        # We group by parent name and pick the best CNN affinity per parent.
        parent_best: dict[str, tuple[float, float, str]] = {}  # parent -> (best_cnn, best_vina, best_sdf)
        for state_name, poses in pocket_dock.results.items():
            if not poses:
                continue
            parent = name_map.get(state_name, state_name)
            best_cnn = poses[0].cnn_affinity
            best_vina = poses[0].vina_score
            best_sdf = poses[0].pose_sdf_block
            if parent not in parent_best or best_cnn > parent_best[parent][0]:
                parent_best[parent] = (best_cnn, best_vina, best_sdf)

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
            sar = compute_sar_discrimination(
                frame_index=frame_index,
                pocket_result=parent_dock,
                active_names=set(active_names),
                inactive_names=set(inactive_names),
                score_key="cnn_affinity",
            )
            # Best-scoring active as the representative pose (highest pK)
            best_active = max(
                [n for n in active_names if n in parent_best],
                key=lambda n: parent_best[n][0],
                default="",
            )
            best_cnn = parent_best.get(best_active, (0.0, 0.0, ""))[0]
            best_vina = parent_best.get(best_active, (0.0, 0.0, ""))[1]
            best_sdf = parent_best.get(best_active, (0.0, 0.0, ""))[2]
            results.append(_FramePoseResult(
                frame_index=frame_index, frame_time_ps=frame_time_ps,
                pocket_id=pocket_id, pocket_score=pocket_score,
                ligand_name=best_active,
                cnn_affinity=best_cnn, vina_score=best_vina,
                auc_roc=sar.auc_roc,
                docked_sdf_block=best_sdf,
                status="ok",
            ))
        else:
            # Single compound mode: one result per pocket per parent ligand
            for parent, (best_cnn, best_vina, best_sdf) in parent_best.items():
                results.append(_FramePoseResult(
                    frame_index=frame_index, frame_time_ps=frame_time_ps,
                    pocket_id=pocket_id, pocket_score=pocket_score, ligand_name=parent,
                    cnn_affinity=best_cnn,
                    vina_score=best_vina,
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
    from draco.final_refinement import refine_docked_pose
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

    # ── Resource Allocation ──────────────────────────────────────────────────
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    # Dedicate 1 CPU for the main MD thread (which mainly orchestrates the GPU),
    # and the rest for parallel frame analysis on CPU.
    num_workers = max(1, num_cpus - 1)

    print("\n" + "═" * 70)
    print("CONCURRENT RESOURCE ALLOCATION")
    print(f"  Available CPUs (Slurm/OS): {num_cpus}")
    print(f"  Main MD process:           Dedicating 1 CPU thread + 1 {args.platform_name} device")
    print(f"  Analysis Workers:          Dedicating {num_workers} processes (1 thread each)")
    print("═" * 70)

    # ── [1/4] Explicit-solvent MD ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("[1/4]  Explicit-solvent MD  (dynamics engine)")
    print("═" * 70)

    # Shared state for analysis
    results_lock = threading.Lock()
    top_heap = _TopKHeap(k=args.top_k)
    all_rows: list[dict[str, Any]] = []
    
    # We open the CSV here to allow real-time appends in the callback
    summary_csv_path = outdir / "summary.csv"
    csv_fh = summary_csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=_FIELDNAMES)
    csv_writer.writeheader()
    csv_fh.flush()

    # ── Ligand preparation ───────────────────────────────────────────────────
    print(f"\n[2/4]  Preparing ligands …")
    ligands_dir = outdir / "ligands"
    ligands_dir.mkdir(exist_ok=True)
    gnina_output_dir = outdir / "gnina_output"
    gnina_output_dir.mkdir(exist_ok=True)

    # name_map: state_name → parent_ligand_name
    name_map: dict[str, str] = {}

    if args.compound_csv:
        actives, inactives, name_map = load_compound_csv(
            args.compound_csv, num_conformers=args.num_conformers,
        )
        all_compounds = actives + inactives
        # active_names / inactive_names are PARENT-level names
        active_names = sorted({name_map[l.name] for l in actives})
        inactive_names = sorted({name_map[l.name] for l in inactives})
        n_act_states = len(actives)
        n_inact_states = len(inactives)
        print(f"  Loaded {len(active_names)} actives ({n_act_states} states), "
              f"{len(inactive_names)} inactives ({n_inact_states} states) from {args.compound_csv}")
    else:
        states = prepare_protonation_states(
            args.ligand_smiles, name=args.ligand_name,
            num_conformers=args.num_conformers,
        )
        all_compounds = states
        for s in states:
            name_map[s.name] = args.ligand_name
        active_names = []
        inactive_names = []
        n_confs = sum(len(s.conformers) for s in states)
        print(f"  Single compound mode: {args.ligand_name} "
              f"({len(states)} protonation state(s), {n_confs} total conformers)")

    # Write SDF files once; pass absolute paths to workers
    ligand_sdf_paths = write_ligands_for_docking(all_compounds, ligands_dir)
    print(f"  Written {len(ligand_sdf_paths)} SDF files to {ligands_dir}/")

    # Prepare initial protein structure
    print("  Preparing initial protein (PDBFixer + H) …")
    initial_prepared = prepare_protein(args.protein_pdb, ph=args.ph)
    _initial_buf = io.StringIO()
    PDBFile.writeFile(initial_prepared.topology, initial_prepared.positions, _initial_buf)
    initial_pdb_string = _initial_buf.getvalue()

    print("\n" + "═" * 70)
    print(f"[3/4]  Concurrent analysis")
    print("═" * 70)

    # Tracks best ranking energy seen so far (more negative = better); updated when rank-1 improves.
    rolling_best_energy: list[float | None] = [None]
    tasks_cv = threading.Condition()
    active_tasks = [0]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init
    ) as executor:
        def on_dock_done(future: concurrent.futures.Future):
            """Called when _dock_frame_worker completes for one frame."""
            try:
                frame_results: list[_FramePoseResult] = future.result()
                for r in frame_results:
                    with results_lock:
                        if r.status in ("ok", "refined"):
                            heap_updated = top_heap.push(r)
                            row = _to_row(r)
                            score_str = (
                                f"AUC={r.auc_roc:.3f}" if r.auc_roc is not None
                                else f"CNN={r.cnn_affinity:.2f}"
                            )
                            print(
                                f"  [pipeline] OK  frame={r.frame_index}  t={r.frame_time_ps:.3f} ps  "
                                f"pocket={r.pocket_id}  {score_str}"
                            )
                            best = top_heap.current_best()
                            if best is not None and args.top_k > 0:
                                prev = rolling_best_energy[0]
                                rank_score = (
                                    -best.auc_roc if best.auc_roc is not None else -best.cnn_affinity
                                )
                                if prev is None or rank_score < prev:
                                    rolling_best_energy[0] = rank_score
                                    best_str = (
                                        f"AUC={best.auc_roc:.3f}" if best.auc_roc is not None
                                        else f"CNN={best.cnn_affinity:.2f}"
                                    )
                                    print(
                                        f"  [top-{args.top_k}] new best rank-1: {best_str} | "
                                        f"frame={best.frame_index} pocket={best.pocket_id} "
                                        f"ligand={best.ligand_name}"
                                    )
                            if heap_updated and args.top_k > 0:
                                _write_multimodel_pdb(outdir / args.top_pdb, top_heap.sorted_best())
                        else:
                            print(f"     Frame {r.frame_index} pocket {r.pocket_id} FAILED: {r.error[:120]}")
                            row = {"status": "error", "frame_index": r.frame_index,
                                   "frame_time_ps": f"{r.frame_time_ps:.3f}",
                                   "pocket_id": r.pocket_id, "pocket_score": "",
                                   "ligand_name": r.ligand_name, "cnn_affinity": "",
                                   "vina_score": "", "auc_roc": "",
                                   "initial_energy_kj_per_mol": "", "final_energy_kj_per_mol": "",
                                   "interaction_energy_kj_per_mol": "",
                                   "ligand_rmsd_from_dock_angstrom": "",
                                   "protein_atoms_flexible": "", "protein_atoms_restrained": "",
                                   "error": r.error}
                        all_rows.append(row)
                        csv_writer.writerow(row)
                        csv_fh.flush()
                        sys.stdout.flush()
            except Exception as exc:
                import traceback
                print(f"  [ERROR] dock worker crashed: {exc}\n{traceback.format_exc()}")
            finally:
                with tasks_cv:
                    active_tasks[0] -= 1
                    if active_tasks[0] == 0:
                        tasks_cv.notify_all()

        def submit_pipeline_job(
            protein_pdb_string: str,
            protein_pdb_path: str,
            frame_index: int,
            frame_time_ps: float,
        ) -> None:
            with tasks_cv:
                active_tasks[0] += 1
            try:
                future = executor.submit(
                    _dock_frame_worker,
                    frame_index=frame_index,
                    frame_time_ps=frame_time_ps,
                    protein_pdb_path=protein_pdb_path,
                    protein_pdb_string=protein_pdb_string,
                    ligand_sdf_paths=ligand_sdf_paths,
                    name_map=name_map,
                    active_names=active_names,
                    inactive_names=inactive_names,
                    pocket_score_threshold=args.pocket_score_threshold,
                    gnina_binary=args.gnina_binary,
                    exhaustiveness=args.exhaustiveness,
                    num_modes=args.num_modes,
                    cnn_scoring=args.cnn_scoring,
                    gnina_seed=args.gnina_seed,
                    dry_run=args.dry_run,
                    gnina_output_dir=str(gnina_output_dir),
                    gnina_timeout_seconds=args.gnina_timeout_seconds,
                    compile_results_only=args.compile_results_only,
                )
                future.add_done_callback(on_dock_done)
            except Exception:
                with tasks_cv:
                    active_tasks[0] -= 1
                    if active_tasks[0] == 0:
                        tasks_cv.notify_all()
                raise



        # We removed the pre-warming loop because concurrently spawning 31+ processes
        # that each load PyTorch/Pocketeer simultaneously can cause a massive thread/I/O
        # bottleneck, resulting in an indefinite hang on startup.
        print(f"  Worker pool created ({num_workers} processes). Workers will spin up on demand.")

        if args.analyze_frames_only or args.compile_results_only:
            print(f"\n  [MD Skipped] {'--compile-results-only' if args.compile_results_only else '--analyze-frames-only'} is active.")
            frame_dir = outdir / "frames"
            if not frame_dir.exists():
                print(f"  Error: Frame directory {frame_dir} not found.")
                return
            
            pdb_files = sorted(frame_dir.glob("*.pdb"))
            if not pdb_files:
                print(f"  Error: No PDB files found in {frame_dir}.")
                return

            print(f"  Found {len(pdb_files)} frame PDB files. Submitting to worker pool...")
            for pdb_path in pdb_files:
                pdb_str = pdb_path.read_text()
                import re
                m = re.search(r"frame_(\d+)", pdb_path.name)
                if m:
                    f_idx = int(m.group(1))
                elif "initial" in pdb_path.name:
                    f_idx = -1
                else:
                    f_idx = 0

                submit_pipeline_job(
                    protein_pdb_string=pdb_str,
                    protein_pdb_path=str(pdb_path),
                    frame_index=f_idx,
                    frame_time_ps=0.0
                )
            
            print("\n  All remaining frames submitted. Waiting for workers to finish...")
            dynamics_sim_time = 0.0
            dynamics_frames_len = len(pdb_files)
            dynamics_trajectory_dcd = None
            dynamics_topology_pdb = None

        else:
            # Run full pipeline on the input (pre-solvation) prepared protein concurrently with MD setup/run.
            frame_dir = outdir / "frames"
            frame_dir.mkdir(exist_ok=True)
            (frame_dir / "frame_initial_input.pdb").write_text(initial_pdb_string)
            print("  Submitted initial (pre-MD) protein to analysis pipeline (frame_index=-1).")
            initial_pdb_path = str(frame_dir / "frame_initial_input.pdb")
            submit_pipeline_job(initial_pdb_string, initial_pdb_path, frame_index=-1, frame_time_ps=0.0)

            def frame_callback(frame: DynamicsFrame) -> None:
                frame_pdb_path = frame_dir / f"frame_{frame.frame_index:04d}.pdb"
                frame_pdb_path.write_text(frame.protein_pdb_string)
                submit_pipeline_job(
                    frame.protein_pdb_string,
                    str(frame_pdb_path),
                    frame_index=frame.frame_index,
                    frame_time_ps=frame.simulation_time_ps,
                )
    
            dynamics_result: DynamicsResult = run_dynamics(
                args.protein_pdb,
                ph=args.ph,
                box_padding_nm=args.box_padding_nm,
                ionic_strength_molar=args.ionic_strength,
                nvt_steps=args.nvt_steps,
                npt_steps=args.npt_steps,
                production_steps=args.production_steps,
                temperature_kelvin=args.temperature_kelvin,
                friction_per_ps=args.friction_per_ps,
                timestep_fs=args.timestep_fs,
                report_interval_steps=args.report_interval_steps,
                rmsd_threshold_angstrom=args.rmsd_threshold_angstrom,
                platform_name=args.platform_name,
                cuda_precision=args.cuda_precision,
                output_dir=outdir,
                save_trajectory=not args.no_trajectory,
                water_model=args.water_model,
                frame_callback=frame_callback,
                verbose=True,
            )
            dynamics_sim_time = dynamics_result.simulation_time_ps
            dynamics_frames_len = len(dynamics_result.frames)
            dynamics_trajectory_dcd = dynamics_result.trajectory_dcd
            dynamics_topology_pdb = dynamics_result.topology_pdb

            print("\n  Main MD complete. Waiting for remaining analysis workers …")
        with tasks_cv:
            while active_tasks[0] > 0:
                tasks_cv.wait()
        # Executor context manager will wait for all tasks on __exit__

    csv_fh.close()

    print(
        f"\n  ✓ All stages complete: {dynamics_sim_time:.1f} ps simulated, "
        f"{dynamics_frames_len} frames analyzed, {len(all_rows)} poses assessed."
    )

    # ── [4/4] Write outputs ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("[4/4]  Finalizing outputs")
    print("═" * 70)

    print(f"  Summary CSV     : {summary_csv_path}  ({len(all_rows)} rows)")

    top_poses = top_heap.sorted_best()
    if top_poses:
        frame_dir = outdir / "frames"
        if args.skip_refinement:
            print("\n  Writing top-k docking poses without refinement ...")
        else:
            print("\n  Refining top-k poses ...")

        final_top_poses = []
        for i, r in enumerate(top_poses, start=1):
            if r.frame_index == -1:
                frame_pdb = str(frame_dir / "frame_initial_input.pdb")
            else:
                frame_pdb = str(frame_dir / f"frame_{r.frame_index:04d}.pdb")

            if args.skip_refinement:
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

            print(f"    Refining rank {i} (frame {r.frame_index}, pocket {r.pocket_id}) ...")
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
                print(f"    Refinement failed for rank {i}: {err_head}")
                continue
            final_top_poses.append(refined_r)

        top_poses = final_top_poses
        if top_poses:
            top_cif_path = outdir / args.top_pdb
            _write_multiblock_cif(top_cif_path, top_poses)
            print(f"  Top-{args.top_k} CIF      : {top_cif_path}  ({len(top_poses)} blocks)")
            print(f"  Top-{args.top_k} output   : per-rank CIF files ({len(top_poses)} models)")
            for rank, r in enumerate(top_poses, start=1):
                if args.skip_refinement:
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
        else:
            if args.skip_refinement:
                print("  Docked-pose export failed for all top poses; no CIF models written.")
            else:
                print("  Refinement failed for all top poses; no refined CIF models written.")
    else:
        print("  No successful poses – nothing to write.")

    if dynamics_trajectory_dcd:
        print(f"  Trajectory DCD  : {dynamics_trajectory_dcd}")
        print(f"  Topology PDB    : {dynamics_topology_pdb}")

    print("\n  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _energy_for_topk_ranking(r: _FramePoseResult) -> float:
    """Scalar used for top-k ordering (must match _TopKHeap.push)."""
    v = r.interaction_energy_kj_per_mol
    return v if v is not None else r.final_energy_kj_per_mol


_FIELDNAMES = [
    "status", "frame_index", "frame_time_ps",
    "pocket_id", "pocket_score",
    "ligand_name", "cnn_affinity", "vina_score", "auc_roc",
    "initial_energy_kj_per_mol", "final_energy_kj_per_mol",
    "interaction_energy_kj_per_mol",
    "ligand_rmsd_from_dock_angstrom",
    "protein_atoms_flexible", "protein_atoms_restrained",
    "error",
]


def _to_row(r: _FramePoseResult) -> dict[str, Any]:
    return {
        "status": r.status,
        "frame_index": r.frame_index,
        "frame_time_ps": f"{r.frame_time_ps:.3f}",
        "pocket_id": r.pocket_id,
        "pocket_score": f"{r.pocket_score:.3f}",
        "ligand_name": r.ligand_name,
        "cnn_affinity": f"{r.cnn_affinity:.3f}",
        "vina_score": f"{r.vina_score:.3f}",
        "auc_roc": f"{r.auc_roc:.4f}" if r.auc_roc is not None else "",
        "initial_energy_kj_per_mol": f"{r.initial_energy_kj_per_mol:.2f}",
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



def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


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
