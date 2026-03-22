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
import csv
import heapq
import io
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
        energy = (
            result.interaction_energy_kj_per_mol
            if result.interaction_energy_kj_per_mol is not None
            else result.final_energy_kj_per_mol
        )
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

def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── [1/4] Explicit-solvent MD ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("[1/4]  Explicit-solvent MD  (dynamics engine)")
    print("═" * 70)
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
        verbose=True,
    )

    print(
        f"\n  ✓ MD complete:  {dynamics_result.simulation_time_ps:.1f} ps simulated, "
        f"{len(dynamics_result.frames)} conformational frames extracted."
    )
    if not dynamics_result.frames:
        print("  No frames extracted — nothing to analyse. Exiting.")
        return

    # ── [2/4] Prepare ligand conformers (once, reused for all frames) ────────
    print("\n" + "═" * 70)
    print("[2/4]  Preparing ligand conformers")
    print("═" * 70)
    prepared_ligand: PreparedLigand = prepare_ligand_from_smiles(
        args.ligand_smiles,
        name=args.ligand_name,
        num_conformers=args.num_conformers,
    )
    print(f"  {len(prepared_ligand.conformers)} conformers generated.")

    # ── [3/4] Per-frame pocketeer + overlay + minimization ─────────────────
    print("\n" + "═" * 70)
    print(f"[3/4]  Per-frame analysis  ({len(dynamics_result.frames)} frames)")
    print("═" * 70)

    top_heap = _TopKHeap(k=args.top_k)
    all_rows: list[dict[str, Any]] = []

    for frame in dynamics_result.frames:
        print(
            f"\n  ── Frame {frame.frame_index:4d}  "
            f"t={frame.simulation_time_ps:8.1f} ps  "
            f"RMSD={frame.rmsd_from_prev_angstrom:.2f} Å ──"
        )

        # 3a. Find pockets on this snapshot
        pockets = pt.find_pockets(frame.atomarray)
        pockets = [
            p for p in pockets
            if float(getattr(p, "score", 0.0)) > args.pocket_score_threshold
        ]
        print(f"     Pockets above threshold: {len(pockets)}")
        if not pockets:
            continue

        pocket_score_map = {
            int(p.pocket_id): float(getattr(p, "score", 0.0)) for p in pockets
        }

        # 3b. Color pockets
        colored_pockets = color_pockets(frame.atomarray, pockets)

        # 3c. Overlay ligand conformers
        dedupe = args.overlay_dedupe_rmsd if args.overlay_dedupe_rmsd > 0.0 else None
        if args.poses_per_pocket <= 1:
            overlay_results = rank_ligand_over_pockets(prepared_ligand, colored_pockets)
        else:
            overlay_results = rank_ligand_over_pockets_multi(
                prepared_ligand,
                colored_pockets,
                poses_per_pocket=args.poses_per_pocket,
                dedupe_heavy_atom_rmsd=dedupe,
            )

        positive_poses = [r for r in overlay_results if r.gaussian_fit_score > 0]
        print(f"     Positive overlay poses:  {len(positive_poses)}")
        if not positive_poses:
            continue

        # 3d. Build a PreparedProtein from this frame's PDB string
        #     (re-use prepared topology — just substitute positions)
        frame_protein = _protein_from_frame(
            frame=frame,
            base_prepared_protein=dynamics_result.prepared_protein,
        )
        if frame_protein is None:
            # Fallback: re-prepare from the frame PDB string directly
            frame_protein = _protein_from_pdb_string(frame.protein_pdb_string)

        # 3e. Local minimization (no induced-fit)
        for pose_idx, pose in enumerate(positive_poses, start=1):
            try:
                minim = _local_minimize(
                    frame_protein=frame_protein,
                    overlay_result=pose,
                    shell_radius_angstrom=args.shell_radius_angstrom,
                    protein_restraint_k=args.protein_restraint_k,
                    ligand_restraint_k=args.ligand_restraint_k,
                    max_iterations=args.minimize_pose_iterations,
                    platform_name=args.platform_name,
                    cuda_precision=args.cuda_precision,
                    compute_ie=not args.no_interaction_energy,
                )
                ie = minim.interaction_energy_kj_per_mol
                print(
                    f"     pose {pose_idx:3d}: pocket={pose.pocket_id} "
                    f"gfit={pose.gaussian_fit_score:.3f}  "
                    f"ΔE_int={'%+.1f' % ie if ie is not None else 'N/A':>10s} kJ/mol  "
                    f"lig_rmsd={minim.ligand_heavy_atom_rmsd_angstrom:.2f} Å"
                )
                result = _FramePoseResult(
                    frame_index=frame.frame_index,
                    frame_time_ps=frame.simulation_time_ps,
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
                )
                top_heap.push(result)
                all_rows.append(_to_row(result))

            except Exception as exc:
                print(
                    f"     pose {pose_idx:3d}: pocket={pose.pocket_id} "
                    f"— FAILED: {exc}"
                )
                all_rows.append(
                    _error_row(frame, pose,
                               pocket_score_map.get(pose.pocket_id, float("nan")),
                               str(exc))
                )

    # ── [4/4] Write outputs ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("[4/4]  Writing outputs")
    print("═" * 70)

    summary_csv = outdir / "summary.csv"
    _write_csv(summary_csv, all_rows)
    print(f"  Summary CSV     : {summary_csv}  ({len(all_rows)} rows)")

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
