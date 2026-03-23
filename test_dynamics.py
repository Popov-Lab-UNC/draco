"""test_dynamics.py – End-to-end Draco pipeline test driven by MD.

Pipeline
--------
                  ┌──────────────────────────────────────────────────────────┐
  6pbc-prepared   │               dynamics.py (this script calls it)          │
  .pdb   +        │  prepare_protein → solvate (TIP3P-FB) → minimize →      │
  SMILES  ──────► │  NVT production  →  Cα RMSD-change frame extraction      │
                  └─────────────────────────┬────────────────────────────────┘
                                            │  conformational frames
                                            ▼  (protein-only snapshots)
                              ┌─────────────────────────┐
                              │   per-frame pipeline     │
                              │                          │
                              │  1. pocketeer.find_pockets
                              │  2. pocket_coloring      │
                              │  3. overlay.rank_ligand  │
                              │  4. local_minimization   │
                              │     (no induced-fit)     │
                              │  5. interaction_energy   │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  top-k heap (rolling)   │
                              │  ranked by ΔE_int        │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Output                │
                              │  summary.csv            │
                              │  top5_poses.pdb         │
                              │  top0N_frame…_ie….pdb   │
                              │  trajectory.dcd         │
                              └─────────────────────────┘

Usage
-----
    pixi run python test_dynamics.py \\
        --protein-pdb  6pbc-prepared.pdb \\
        --ligand-smiles "CN(CC1(CC1)c1ccc(F)cc1)C(=O)[C@H](Cc1csc2ccccc12)Nc1cc(ncn1)C(N)=O" \\
        --output-dir   dynamics_test \\
        --platform-name CUDA

    # Quick smoke-test on CPU (short run):
    pixi run python test_dynamics.py \\
        --protein-pdb 6pbc-prepared.pdb \\
        --ligand-smiles "CN(CC1(CC1)c1ccc(F)cc1)C(=O)[C@H](Cc1csc2ccccc12)Nc1cc(ncn1)C(N)=O" \\
        --production-steps 10000 --report-interval-steps 2000 \\
        --platform-name CPU --output-dir dynamics_test_cpu
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import heapq
import io
import os
import threading
import time
import multiprocessing
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dynamics import DynamicsFrame, DynamicsResult, run_dynamics
from ligand_preparation import PreparedLigand, conformer_to_pdb_block, prepare_ligand_from_smiles
from local_minimization import (
    LocalMinimizationResult,
    _add_positional_restraints,
    _compute_interaction_energy,
    _partition_protein_atoms_by_shell,
    _register_ligand_template,
    _rmsd,
)
from overlay import OverlayResult, rank_ligand_over_pockets, rank_ligand_over_pockets_multi
from pocket_coloring import color_pockets
from protein_preparation import PreparedProtein, prepare_protein

import pocketeer as pt

try:
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
except ImportError:  # pragma: no cover
    from simtk.openmm import LangevinMiddleIntegrator, Platform, unit  # type: ignore
    from simtk.openmm.app import (  # type: ignore
        ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation,
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
        description=(
            "Run Draco end-to-end: explicit-solvent MD → pocketeer → overlay"
            " → local minimization → top-k poses by interaction energy."
        )
    )

    # ── Input / output ──────────────────────────────────────────────────────
    parser.add_argument("--protein-pdb",    required=True,  help="Input protein PDB path")
    parser.add_argument("--ligand-smiles",  required=True,  help="Ligand SMILES string")
    parser.add_argument("--ligand-name",    default="LIG",  help="Ligand display name")
    parser.add_argument("--output-dir",     default="dynamics_test_output")
    parser.add_argument("--ph",             type=float, default=7.4)

    # ── MD / dynamics parameters ────────────────────────────────────────────
    parser.add_argument("--platform-name",     default="CUDA",
                        help="OpenMM platform: CUDA (default), OpenCL, CPU")
    parser.add_argument("--cuda-precision",    default="mixed",
                        help="CUDA precision: mixed (default), single, double")
    parser.add_argument("--box-padding-nm",    type=float, default=1.0,
                        help="Water box padding in nm (default 1.0)")
    parser.add_argument("--ionic-strength",    type=float, default=0.15,
                        help="NaCl ionic strength in mol/L (default 0.15)")
    parser.add_argument("--nvt-steps",          type=int, default=50_000,
                        help="NVT equilibration steps with restraints (default 50k = 100 ps @ 2 fs)")
    parser.add_argument("--npt-steps",          type=int, default=50_000,
                        help="NPT equilibration steps with barostat + restraints (default 50k = 100 ps @ 2 fs)")
    parser.add_argument("--production-steps",  type=int, default=2_500_000,
                        help="Production MD steps (default 2.5M = 5 ns @ 2 fs)")
    parser.add_argument("--timestep-fs",       type=float, default=2.0)
    parser.add_argument("--friction-per-ps",    type=float, default=1.0,
                        help="Langevin friction γ in 1/ps (default 1.0; matches dynamics.py)")
    parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    parser.add_argument("--water-model",        default="tip3pfb",
                        help="Water model label for Modeller/FF (default tip3pfb; see dynamics.py)")
    parser.add_argument("--report-interval-steps", type=int, default=5_000,
                        help="RMSD check interval in steps (default 5000 = 10 ps)")
    parser.add_argument("--rmsd-threshold-angstrom", type=float, default=1.5,
                        help="Backbone RMSD threshold (Å) to trigger frame extraction (default 1.5)")
    parser.add_argument("--no-trajectory",     action="store_true", default=False,
                        help="Skip writing DCD trajectory (saves disk space)")

    # ── Pocketeer / overlay ─────────────────────────────────────────────────
    parser.add_argument("--pocket-score-threshold", type=float, default=5.0)
    parser.add_argument("--num-conformers",   type=int, default=10)
    parser.add_argument("--poses-per-pocket", type=int, default=5)
    parser.add_argument("--overlay-dedupe-rmsd", type=float, default=1.0)

    # ── Local minimization ──────────────────────────────────────────────────
    parser.add_argument("--shell-radius-angstrom",   type=float, default=8.0)
    parser.add_argument("--protein-restraint-k",     type=float, default=10.0)
    parser.add_argument("--ligand-restraint-k",      type=float, default=1.0)
    parser.add_argument("--minimize-pose-iterations", type=int, default=1000)
    parser.add_argument("--no-interaction-energy",   action="store_true", default=False)

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument("--top-k",   type=int, default=5,
                        help="Number of best-energy poses to save (default 5)")
    parser.add_argument("--top-pdb", default="top5_poses.pdb")

    # ── Debug / Testing ─────────────────────────────────────────────────────
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Skip actual analysis and return dummy results for concurrency testing")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Top-k tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FramePoseResult:
    frame_index: int
    frame_time_ps: float
    pocket_id: int
    pocket_score: float
    conformer_id: int
    gaussian_fit_score: float
    initial_energy_kj_per_mol: float
    final_energy_kj_per_mol: float
    interaction_energy_kj_per_mol: float | None
    ligand_heavy_atom_rmsd_angstrom: float
    protein_atoms_flexible: int
    protein_atoms_restrained: int
    ligand_atoms_restrained: int
    minimized_complex_pdb: str
    status: str = "ok"
    error: str = ""


@dataclass
class _TopKHeap:
    k: int
    _heap: list[tuple[float, int, _FramePoseResult]] = field(default_factory=list)
    _counter: int = 0

    def push(self, result: _FramePoseResult) -> None:
        val = result.interaction_energy_kj_per_mol
        energy: float = val if val is not None else result.final_energy_kj_per_mol
        entry = (energy, self._counter, result)
        self._counter += 1
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif energy < self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def sorted_best(self) -> list[_FramePoseResult]:
        """Most → least negative energy."""
        return [r for _, _, r in sorted(self._heap)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Parallel Worker Logic
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init() -> None:
    """Initializer for ProcessPoolExecutor workers."""
    # 1. Force each worker to use only 1 CPU thread for OpenMM CPU platform.
    #    Without this, 15 workers in a 16-core Slurm allocation would each
    #    try to spin up 16 threads, leading to severe oversubscription.
    os.environ["OPENMM_CPU_THREADS"] = "1"

    # 2. Pre-import heavy modules to avoid overhead on first task.
    try:
        import openmm
        import rdkit
        import pocketeer
        import protein_preparation
    except ImportError:
        pass


def _process_frame_worker(
    frame_index: int,
    frame_time_ps: float,
    protein_pdb_string: str,
    prepared_ligand: PreparedLigand,
    # Parameters from args
    pocket_score_threshold: float,
    poses_per_pocket: int,
    overlay_dedupe_rmsd: float,
    shell_radius_angstrom: float,
    protein_restraint_k: float,
    ligand_restraint_k: float,
    minimize_pose_iterations: int,
    compute_ie: bool,
    dry_run: bool = False,
) -> list[_FramePoseResult]:
    """Process a single frame: pocketeer → overlay → local minimization.

    Executed in a sub-process via ProcessPoolExecutor.
    """
    if dry_run:
        # Concurrency smoke-test: simulate work and return a dummy result.
        time.sleep(0.5)
        return [_FramePoseResult(
            frame_index=frame_index,
            frame_time_ps=frame_time_ps,
            pocket_id=1,
            pocket_score=10.0,
            conformer_id=0,
            gaussian_fit_score=0.99,
            initial_energy_kj_per_mol=0.0,
            final_energy_kj_per_mol=-100.0,
            interaction_energy_kj_per_mol=-50.0 if compute_ie else None,
            ligand_heavy_atom_rmsd_angstrom=0.1,
            protein_atoms_flexible=10,
            protein_atoms_restrained=100,
            ligand_atoms_restrained=20,
            minimized_complex_pdb=protein_pdb_string, # dummy
            status="ok"
        )]

    # 1. Parse structure from PDB string
    #    We use any available platform (likely CPU since CUDA is used by main)
    #    for pocketeer/overlay setup.
    import io
    import pocketeer as pt
    import biotite.structure.io.pdb as _biotite_pdb

    frame_atomarray = _biotite_pdb.PDBFile.read(
        io.StringIO(protein_pdb_string)
    ).get_structure(model=1)

    pockets = pt.find_pockets(frame_atomarray)
    pockets = [
        p for p in pockets
        if float(getattr(p, "score", 0.0)) > pocket_score_threshold
    ]
    if not pockets:
        return []

    pocket_score_map = {
        int(getattr(p, "pocket_id")): float(getattr(p, "score", 0.0)) for p in pockets
    }

    # 3. Color pockets
    colored_pockets = color_pockets(frame_atomarray, pockets)

    # 4. Overlay ligand conformers
    dedupe = overlay_dedupe_rmsd if overlay_dedupe_rmsd > 0.0 else None
    if poses_per_pocket <= 1:
        overlay_results = rank_ligand_over_pockets(prepared_ligand, colored_pockets)
    else:
        overlay_results = rank_ligand_over_pockets_multi(
            prepared_ligand,
            colored_pockets,
            poses_per_pocket=poses_per_pocket,
            dedupe_heavy_atom_rmsd=dedupe,
        )

    positive_poses = [r for r in overlay_results if r.gaussian_fit_score > 0]
    if not positive_poses:
        return []

    # 5. Build PreparedProtein for this frame (full preparation needed in worker)
    frame_protein = _protein_from_pdb_string(protein_pdb_string)

    # 6. Local minimization (CRITICAL: Force CPU platform)
    results = []
    for pose in positive_poses:
        try:
            minim = _local_minimize(
                frame_protein=frame_protein,
                overlay_result=pose,
                shell_radius_angstrom=shell_radius_angstrom,
                protein_restraint_k=protein_restraint_k,
                ligand_restraint_k=ligand_restraint_k,
                max_iterations=minimize_pose_iterations,
                platform_name="CPU",  # Force CPU to avoid GPU contention
                cuda_precision="mixed",
                compute_ie=compute_ie,
            )
            ie = minim.interaction_energy_kj_per_mol
            results.append(_FramePoseResult(
                frame_index=frame_index,
                frame_time_ps=frame_time_ps,
                pocket_id=pose.pocket_id,
                pocket_score=pocket_score_map.get(pose.pocket_id, float("nan")),
                conformer_id=pose.conformer_id,
                gaussian_fit_score=pose.gaussian_fit_score,
                initial_energy_kj_per_mol=minim.initial_energy_kj_per_mol,
                final_energy_kj_per_mol=minim.final_energy_kj_per_mol,
                interaction_energy_kj_per_mol=ie,
                ligand_heavy_atom_rmsd_angstrom=minim.ligand_heavy_atom_rmsd_angstrom,
                protein_atoms_flexible=minim.protein_atoms_flexible,
                protein_atoms_restrained=minim.protein_atoms_restrained,
                ligand_atoms_restrained=minim.ligand_atoms_restrained,
                minimized_complex_pdb=minim.minimized_complex_pdb,
            ))
        except Exception as exc:
            results.append(_FramePoseResult(
                frame_index=frame_index,
                frame_time_ps=frame_time_ps,
                pocket_id=pose.pocket_id,
                pocket_score=pocket_score_map.get(pose.pocket_id, float("nan")),
                conformer_id=pose.conformer_id,
                gaussian_fit_score=pose.gaussian_fit_score,
                initial_energy_kj_per_mol=0.0,
                final_energy_kj_per_mol=0.0,
                interaction_energy_kj_per_mol=None,
                ligand_heavy_atom_rmsd_angstrom=0.0,
                protein_atoms_flexible=0,
                protein_atoms_restrained=0,
                ligand_atoms_restrained=0,
                minimized_complex_pdb="",
                status="error",
                error=str(exc)
            ))
    return results


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

    # Prep ligand once (needed for all frames)
    print(f"\n[2/4]  Preparing ligand {args.ligand_name} …")
    prepared_ligand: PreparedLigand = prepare_ligand_from_smiles(
        args.ligand_smiles,
        name=args.ligand_name,
        num_conformers=args.num_conformers,
    )
    print(f"  {len(prepared_ligand.conformers)} conformers generated.")

    print("\n" + "═" * 70)
    print(f"[3/4]  Concurrent analysis")
    print("═" * 70)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init
    ) as executor:

        def on_frame_done(future: concurrent.futures.Future):
            try:
                frame_results = future.result()
                with results_lock:
                    for r in frame_results:
                        if r.status == "ok":
                            top_heap.push(r)
                            row = _to_row(r)
                        else:
                            print(f"     Frame {r.frame_index} pocket {r.pocket_id} FAILED: {r.error}")
                            row = {
                                "status": "error",
                                "frame_index": r.frame_index,
                                "frame_time_ps": f"{r.frame_time_ps:.3f}",
                                "pocket_id": r.pocket_id,
                                "pocket_score": f"{r.pocket_score:.3f}",
                                "gaussian_fit_score": f"{r.gaussian_fit_score:.4f}",
                                "conformer_id": r.conformer_id,
                                "error": r.error
                            }
                        all_rows.append(row)
                        csv_writer.writerow(row)
                    csv_fh.flush() # Ensure it hits disk immediately
            except Exception as exc:
                print(f"  [ERROR] Worker crashed processing frame: {exc}")

        def frame_callback(frame: DynamicsFrame) -> None:
            # 1. Save frame PDB immediately for user/PyMol
            frame_dir = outdir / "frames"
            frame_dir.mkdir(exist_ok=True)
            (frame_dir / f"frame_{frame.frame_index:04d}.pdb").write_text(frame.protein_pdb_string)

            # 2. Submit to worker pool
            future = executor.submit(
                _process_frame_worker,
                frame_index=frame.frame_index,
                frame_time_ps=frame.simulation_time_ps,
                protein_pdb_string=frame.protein_pdb_string,
                prepared_ligand=prepared_ligand,
                pocket_score_threshold=args.pocket_score_threshold,
                poses_per_pocket=args.poses_per_pocket,
                overlay_dedupe_rmsd=args.overlay_dedupe_rmsd,
                shell_radius_angstrom=args.shell_radius_angstrom,
                protein_restraint_k=args.protein_restraint_k,
                ligand_restraint_k=args.ligand_restraint_k,
                minimize_pose_iterations=args.minimize_pose_iterations,
                compute_ie=not args.no_interaction_energy,
                dry_run=args.dry_run,
            )
            future.add_done_callback(on_frame_done)

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

        print("\n  Main MD complete. Waiting for remaining analysis workers …")
        # Executor context manager will wait for all tasks on __exit__

    csv_fh.close()

    print(
        f"\n  ✓ All stages complete: {dynamics_result.simulation_time_ps:.1f} ps simulated, "
        f"{len(dynamics_result.frames)} frames generated, {len(all_rows)} poses analysed."
    )

    # ── [4/4] Write outputs ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("[4/4]  Finalizing outputs")
    print("═" * 70)

    print(f"  Summary CSV     : {summary_csv_path}  ({len(all_rows)} rows)")

    top_poses = top_heap.sorted_best()
    if top_poses:
        top_pdb_path = outdir / args.top_pdb
        _write_multimodel_pdb(top_pdb_path, top_poses)
        print(f"  Top-{args.top_k} PDB      : {top_pdb_path}  ({len(top_poses)} models)")

        for rank, r in enumerate(top_poses, start=1):
            ie_tag = (
                f"{r.interaction_energy_kj_per_mol:.1f}"
                if r.interaction_energy_kj_per_mol is not None
                else "noE"
            )
            fname = (
                f"top{rank:02d}_frame{r.frame_index:04d}_"
                f"pocket{r.pocket_id:03d}_"
                f"ie{ie_tag.replace('-', 'n')}_kJmol.pdb"
            )
            (outdir / fname).write_text(r.minimized_complex_pdb)
            print(f"  Written         : {outdir / fname}")
    else:
        print("  No successful poses – nothing to write.")

    if dynamics_result.trajectory_dcd:
        print(f"  Trajectory DCD  : {dynamics_result.trajectory_dcd}")
        print(f"  Topology PDB    : {dynamics_result.topology_pdb}")

    print("\n  Done.")

    top_poses = top_heap.sorted_best()
    if top_poses:
        top_pdb_path = outdir / args.top_pdb
        _write_multimodel_pdb(top_pdb_path, top_poses)
        print(f"  Top-{args.top_k} PDB      : {top_pdb_path}  ({len(top_poses)} models)")

        for rank, r in enumerate(top_poses, start=1):
            ie_tag = (
                f"{r.interaction_energy_kj_per_mol:.1f}"
                if r.interaction_energy_kj_per_mol is not None
                else "noE"
            )
            fname = (
                f"top{rank:02d}_frame{r.frame_index:04d}_"
                f"pocket{r.pocket_id:03d}_"
                f"ie{ie_tag.replace('-', 'n')}_kJmol.pdb"
            )
            (outdir / fname).write_text(r.minimized_complex_pdb)
            print(f"  Written         : {outdir / fname}")
    else:
        print("  No successful poses – nothing to write.")

    if dynamics_result.trajectory_dcd:
        print(f"  Trajectory DCD  : {dynamics_result.trajectory_dcd}")
        print(f"  Topology PDB    : {dynamics_result.topology_pdb}")

    print("\n  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level protein helpers
# ─────────────────────────────────────────────────────────────────────────────

def _protein_from_frame(
    *,
    frame: DynamicsFrame,
    base_prepared_protein: PreparedProtein,
) -> PreparedProtein | None:
    """Return a PreparedProtein with positions from *frame*, reusing topology.

    Tries to substitute the frame positions into the base topology directly.
    Returns None if atom counts don't match (use _protein_from_pdb_string then).
    """
    n_frame = frame.protein_positions_nm.shape[0]
    topo_atoms = list(base_prepared_protein.topology.atoms())
    if n_frame != len(topo_atoms):
        return None

    new_pos = (frame.protein_positions_nm * unit.nanometer)
    return PreparedProtein(
        topology=base_prepared_protein.topology,
        positions=new_pos,
        source_path=base_prepared_protein.source_path,
    )


def _protein_from_pdb_string(pdb_string: str) -> PreparedProtein:
    """Build a PreparedProtein from a PDB string via PDBFixer.

    Slower fallback used when the frame topology doesn't match the base.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb_string)
        tmp_path = f.name
    try:
        return prepare_protein(tmp_path, add_hydrogens=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pose local minimization (no induced-fit)
# ─────────────────────────────────────────────────────────────────────────────

def _local_minimize(
    *,
    frame_protein: PreparedProtein,
    overlay_result: OverlayResult,
    shell_radius_angstrom: float,
    protein_restraint_k: float,
    ligand_restraint_k: float,
    max_iterations: int,
    platform_name: str,
    cuda_precision: str,
    compute_ie: bool,
    protein_forcefield_files: tuple[str, ...] = ("amber14-all.xml", "amber14/tip3pfb.xml"),
    openff_forcefield: str = "openff-2.3.0",
    ligand_residue_name: str = "LIG",
    temperature_kelvin: float = 300.0,
    friction_per_ps: float = 1.0,
    timestep_fs: float = 2.0,
) -> LocalMinimizationResult:
    """Run local energy minimization of one overlay pose at one MD frame.

    No induced-fit MD — this keeps each frame analysis fast.
    """
    ligand_pdb = PDBFile(
        io.StringIO(
            conformer_to_pdb_block(
                overlay_result.conformer,
                overlay_result.transformed_all_atom_coords,
                residue_name=ligand_residue_name,
                chain_id="L",
                residue_id=1,
            )
        )
    )

    modeller = Modeller(frame_protein.topology, frame_protein.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.positions)

    forcefield = ForceField(*protein_forcefield_files)
    _register_ligand_template(
        forcefield, overlay_result, openff_forcefield=openff_forcefield
    )

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )

    topo_atoms = list(modeller.topology.atoms())
    positions_nm = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )

    ligand_idxs = [
        a.index for a in topo_atoms if a.residue.name == ligand_residue_name
    ]
    if not ligand_idxs:
        raise ValueError(f"Ligand residue '{ligand_residue_name}' not found in topology")

    protein_idxs = [
        a.index for a in topo_atoms
        if a.residue.name != ligand_residue_name and a.element is not None
    ]

    restrained_prot, flexible_prot = _partition_protein_atoms_by_shell(
        positions_nm=positions_nm,
        protein_atom_indices=protein_idxs,
        ligand_atom_indices=ligand_idxs,
        shell_radius_angstrom=shell_radius_angstrom,
    )
    n_prot_restrained = _add_positional_restraints(
        system=system, positions_nm=positions_nm,
        atom_indices=restrained_prot,
        k_kcal_per_mol_a2=protein_restraint_k,
        k_param_name="k_protein_posres",
    )
    n_lig_restrained = _add_positional_restraints(
        system=system, positions_nm=positions_nm,
        atom_indices=ligand_idxs,
        k_kcal_per_mol_a2=ligand_restraint_k,
        k_param_name="k_ligand_posres",
    )

    integrator = LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
        props: dict[str, str] = {}
        if platform_name.upper() in {"CUDA", "OPENCL"}:
            props["Precision"] = cuda_precision
        simulation = Simulation(modeller.topology, system, integrator, platform, props)
    except Exception:
        simulation = Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)

    init_state = simulation.context.getState(getEnergy=True, getPositions=True)
    init_energy = float(
        init_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    init_pos_nm = np.asarray(
        init_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )

    simulation.minimizeEnergy(maxIterations=max_iterations)

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = float(
        final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    final_pos_nm = np.asarray(
        final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )

    heavy_rel = list(overlay_result.conformer.heavy_atom_indices)
    heavy_abs = [ligand_idxs[i] for i in heavy_rel if i < len(ligand_idxs)]
    lig_rmsd = _rmsd(
        init_pos_nm[heavy_abs] * 10.0,
        final_pos_nm[heavy_abs] * 10.0,
    )

    ie: float | None = None
    if compute_ie:
        ie = _compute_interaction_energy(
            simulation=simulation,
            ligand_atom_indices=ligand_idxs,
            protein_atom_indices=protein_idxs,
            final_energy_kj_per_mol=final_energy,
        )

    final_out = io.StringIO()
    PDBFile.writeFile(modeller.topology, final_state.getPositions(), final_out, keepIds=True)

    return LocalMinimizationResult(
        initial_energy_kj_per_mol=init_energy,
        final_energy_kj_per_mol=final_energy,
        ligand_heavy_atom_rmsd_angstrom=lig_rmsd,
        protein_atoms_flexible=len(flexible_prot),
        protein_atoms_restrained=n_prot_restrained,
        ligand_atoms_restrained=n_lig_restrained,
        minimized_positions_angstrom=final_pos_nm * 10.0,
        minimized_complex_pdb=final_out.getvalue(),
        interaction_energy_kj_per_mol=ie,
        induced_fit_ligand_rmsd_angstrom=None,
        induced_fit_final_energy_kj_per_mol=None,
        induced_fit_complex_pdb=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

_FIELDNAMES = [
    "status", "frame_index", "frame_time_ps",
    "pocket_id", "pocket_score",
    "gaussian_fit_score", "conformer_id",
    "initial_energy_kj_per_mol", "final_energy_kj_per_mol",
    "interaction_energy_kj_per_mol",
    "ligand_heavy_atom_rmsd_angstrom",
    "protein_atoms_flexible", "protein_atoms_restrained", "ligand_atoms_restrained",
    "error",
]


def _to_row(r: _FramePoseResult) -> dict[str, Any]:
    return {
        "status": r.status,
        "frame_index": r.frame_index,
        "frame_time_ps": f"{r.frame_time_ps:.3f}",
        "pocket_id": r.pocket_id,
        "pocket_score": f"{r.pocket_score:.3f}",
        "gaussian_fit_score": f"{r.gaussian_fit_score:.4f}",
        "conformer_id": r.conformer_id,
        "initial_energy_kj_per_mol": f"{r.initial_energy_kj_per_mol:.2f}",
        "final_energy_kj_per_mol": f"{r.final_energy_kj_per_mol:.2f}",
        "interaction_energy_kj_per_mol": (
            f"{r.interaction_energy_kj_per_mol:.2f}"
            if r.interaction_energy_kj_per_mol is not None else ""
        ),
        "ligand_heavy_atom_rmsd_angstrom": f"{r.ligand_heavy_atom_rmsd_angstrom:.3f}",
        "protein_atoms_flexible": r.protein_atoms_flexible,
        "protein_atoms_restrained": r.protein_atoms_restrained,
        "ligand_atoms_restrained": r.ligand_atoms_restrained,
        "error": r.error,
    }


def _error_row(
    frame: DynamicsFrame,
    pose: OverlayResult,
    pocket_score: float,
    error: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "frame_index": frame.frame_index,
        "frame_time_ps": f"{frame.simulation_time_ps:.3f}",
        "pocket_id": pose.pocket_id,
        "pocket_score": f"{pocket_score:.3f}",
        "gaussian_fit_score": f"{pose.gaussian_fit_score:.4f}",
        "conformer_id": pose.conformer_id,
        "initial_energy_kj_per_mol": "",
        "final_energy_kj_per_mol": "",
        "interaction_energy_kj_per_mol": "",
        "ligand_heavy_atom_rmsd_angstrom": "",
        "protein_atoms_flexible": "",
        "protein_atoms_restrained": "",
        "ligand_atoms_restrained": "",
        "error": error,
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
            f"GAUSSIAN_FIT {r.gaussian_fit_score:.4f} "
            f"INTERACTION_ENERGY_KJ_MOL {ie_str} "
            f"FINAL_ENERGY_KJ_MOL {r.final_energy_kj_per_mol:.2f}"
        )
        for line in r.minimized_complex_pdb.splitlines():
            if line.strip() in {"END", "ENDMDL", "MODEL"}:
                continue
            lines.append(line)
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
