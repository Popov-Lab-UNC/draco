from __future__ import annotations

import warnings
import argparse
import csv
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.distributed\.reduce_op.*",
)

import pocketeer as pt

from ligand_preparation import prepare_ligand_from_smiles
from local_minimization import LocalMinimizationResult, minimize_overlay_pose
from overlay import OverlayResult, rank_ligand_over_pockets, rank_ligand_over_pockets_multi
from pocket_coloring import color_pockets
from protein_preparation import prepare_protein


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pocketeer -> color+overlay -> keep rank_score>0 poses -> "
            "local OpenMM minimization -> write minimized PDBs."
        )
    )
    parser.add_argument("--protein-pdb", required=True, help="Input protein PDB path")
    parser.add_argument("--ligand-smiles", required=True, help="One ligand SMILES string")
    parser.add_argument("--ligand-name", default="LIG", help="Ligand display name")
    parser.add_argument(
        "--output-dir",
        default="minimized_overlay_test",
        help="Directory for per-pose PDBs and summary CSV",
    )

    # Notebook-like defaults
    parser.add_argument(
        "--pocket-score-threshold",
        type=float,
        default=5.0,
        help="Keep pocketeer pockets with score > threshold",
    )
    parser.add_argument(
        "--num-conformers",
        type=int,
        default=10,
        help="Number of RDKit conformers for overlay ligand",
    )
    parser.add_argument(
        "--poses-per-pocket",
        type=int,
        default=1,
        help="Number of retained poses per pocket before score filtering",
    )
    parser.add_argument(
        "--overlay-dedupe-rmsd",
        type=float,
        default=0.0,
        help="Optional heavy-atom RMSD dedupe threshold in Angstrom (0 disables)",
    )

    # Protein preparation
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH used by PDBFixer when adding missing hydrogens (default 7.4)",
    )

    # Minimization knobs
    parser.add_argument("--shell-radius-angstrom", type=float, default=8.0)
    parser.add_argument("--protein-restraint-k", type=float, default=10.0)
    parser.add_argument("--ligand-restraint-k", type=float, default=1.0)
    parser.add_argument("--minimize-max-iterations", type=int, default=2000)
    parser.add_argument(
        "--platform-name",
        default=None,
        help="Optional OpenMM platform name (e.g., CUDA, OpenCL, CPU)",
    )

    parser.add_argument(
        "--multi-pdb",
        default="minimized_poses_multimodel.pdb",
        help="Optional multi-model PDB output filename (empty string to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading protein: {args.protein_pdb}")
    atomarray = pt.load_structure(args.protein_pdb)
    print(f"      Preparing protein with PDBFixer (pH={args.ph}): stripping and re-adding all H")
    prepared_protein = prepare_protein(args.protein_pdb, ph=args.ph)

    print("[2/6] Running pocketeer with default settings")
    pockets = pt.find_pockets(atomarray)
    pockets = [p for p in pockets if float(getattr(p, "score", 0.0)) > args.pocket_score_threshold]
    print(f"      Pockets above threshold ({args.pocket_score_threshold}): {len(pockets)}")
    if not pockets:
        print("No pockets passed threshold. Exiting.")
        return

    print(f"[3/6] Preparing ligand conformers ({args.num_conformers})")
    prepared_ligand = prepare_ligand_from_smiles(
        args.ligand_smiles,
        name=args.ligand_name,
        num_conformers=args.num_conformers,
    )

    print("[4/6] Coloring pockets and ranking overlays")
    colored_pockets = color_pockets(atomarray, pockets)
    if args.poses_per_pocket <= 1:
        overlay_results = rank_ligand_over_pockets(prepared_ligand, colored_pockets)
    else:
        dedupe = args.overlay_dedupe_rmsd if args.overlay_dedupe_rmsd > 0.0 else None
        overlay_results = rank_ligand_over_pockets_multi(
            prepared_ligand,
            colored_pockets,
            poses_per_pocket=args.poses_per_pocket,
            dedupe_heavy_atom_rmsd=dedupe,
        )
    positive_poses = [result for result in overlay_results if result.ranking_score > 0]
    print(f"      Overlay poses with rank score > 0: {len(positive_poses)}")
    if not positive_poses:
        print("No positive overlay poses found. Exiting.")
        return

    pocket_score_map = {int(p.pocket_id): float(getattr(p, "score", 0.0)) for p in pockets}
    summary_rows: list[dict[str, Any]] = []
    successful: list[tuple[OverlayResult, LocalMinimizationResult, Path]] = []

    print("[5/6] Local minimization for each positive pose")
    for idx, pose in enumerate(positive_poses, start=1):
        pose_name = (
            f"pose_{idx:03d}_pocket_{pose.pocket_id:03d}"
            f"_rank_{pose.ranking_score:+d}_conf_{pose.conformer_id}.pdb"
        )
        pose_path = outdir / pose_name
        print(
            f"      Minimizing pose {idx}/{len(positive_poses)} "
            f"(pocket={pose.pocket_id}, rank={pose.ranking_score:+d})"
        )

        try:
            minim = minimize_overlay_pose(
                prepared_protein,
                overlay_result=pose,
                shell_radius_angstrom=args.shell_radius_angstrom,
                protein_restraint_k_kcal_per_mol_A2=args.protein_restraint_k,
                ligand_restraint_k_kcal_per_mol_A2=args.ligand_restraint_k,
                minimize_max_iterations=args.minimize_max_iterations,
                platform_name=args.platform_name,
                output_path=pose_path,
            )
            successful.append((pose, minim, pose_path))
            summary_rows.append(
                {
                    "status": "ok",
                    "pose_index": idx,
                    "pocket_id": pose.pocket_id,
                    "pocket_score": pocket_score_map.get(pose.pocket_id, float("nan")),
                    "rank_score": pose.ranking_score,
                    "conformer_id": pose.conformer_id,
                    "initial_energy_kj_per_mol": minim.initial_energy_kj_per_mol,
                    "final_energy_kj_per_mol": minim.final_energy_kj_per_mol,
                    "ligand_heavy_atom_rmsd_angstrom": minim.ligand_heavy_atom_rmsd_angstrom,
                    "protein_atoms_flexible": minim.protein_atoms_flexible,
                    "protein_atoms_restrained": minim.protein_atoms_restrained,
                    "ligand_atoms_restrained": minim.ligand_atoms_restrained,
                    "output_pdb": str(pose_path),
                    "error": "",
                }
            )
        except Exception as exc:
            error_path = pose_path.with_name(pose_path.stem + "_error" + pose_path.suffix)
            summary_rows.append(
                {
                    "status": "error",
                    "pose_index": idx,
                    "pocket_id": pose.pocket_id,
                    "pocket_score": pocket_score_map.get(pose.pocket_id, float("nan")),
                    "rank_score": pose.ranking_score,
                    "conformer_id": pose.conformer_id,
                    "initial_energy_kj_per_mol": "",
                    "final_energy_kj_per_mol": "",
                    "ligand_heavy_atom_rmsd_angstrom": "",
                    "protein_atoms_flexible": "",
                    "protein_atoms_restrained": "",
                    "ligand_atoms_restrained": "",
                    "output_pdb": str(error_path),
                    "error": str(exc),
                }
            )
            print(f"        ! Minimization failed for pocket {pose.pocket_id}: {exc}")

    summary_csv = outdir / "summary.csv"
    write_summary_csv(summary_csv, summary_rows)
    print(f"      Wrote summary: {summary_csv}")

    if args.multi_pdb and successful:
        multi_path = outdir / args.multi_pdb
        write_multimodel_pdb(multi_path, successful)
        print(f"      Wrote multi-model PDB: {multi_path}")

    print("[6/6] Done")
    print(f"      Successful minimizations: {len(successful)}/{len(positive_poses)}")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "pose_index",
        "pocket_id",
        "pocket_score",
        "rank_score",
        "conformer_id",
        "initial_energy_kj_per_mol",
        "final_energy_kj_per_mol",
        "ligand_heavy_atom_rmsd_angstrom",
        "protein_atoms_flexible",
        "protein_atoms_restrained",
        "ligand_atoms_restrained",
        "output_pdb",
        "error",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_multimodel_pdb(
    path: Path,
    pose_results: list[tuple[OverlayResult, LocalMinimizationResult, Path]],
) -> None:
    lines: list[str] = []
    for model_idx, (pose, minim, _) in enumerate(pose_results, start=1):
        lines.append(f"MODEL     {model_idx:4d}")
        lines.append(
            f"REMARK POCKET_ID {pose.pocket_id} RANK_SCORE {pose.ranking_score:+d} "
            f"CONFORMER_ID {pose.conformer_id}"
        )
        for line in minim.minimized_complex_pdb.splitlines():
            if line.strip() in {"END", "ENDMDL", "MODEL"}:
                continue
            lines.append(line)
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

