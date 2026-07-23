#!/usr/bin/env python3
"""
pipeline/cli.py — Unified CLI Orchestrator for TetraDiffusion

Consolidates the full pipeline:
  1. Data Ingestion & Reorganization
  2. DMTet Grid Preprocessing
  3. CSV Registration
  4. Production Training Launcher / Resumer
  5. Inference & Mesh Generation
  6. Quantitative Evaluation & Shape Space Visualization
  7. Visual Dashboard Launcher

Usage:
    python pipeline/cli.py status
    python pipeline/cli.py ingest --input_dir /path/to/raw --output_dir data_urocell/organelles_raw
    python pipeline/cli.py preprocess --dataset urocell --category fv
    python pipeline/cli.py register --dataset urocell --category fv
    python pipeline/cli.py train --dataset urocell --category fv
    python pipeline/cli.py infer --run_name urocell_fv_final_prod
    python pipeline/cli.py evaluate --dataset urocell
    python pipeline/cli.py dashboard --port 7860
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────
def count_files(pattern: str | Path) -> int:
    return len(glob.glob(str(pattern), recursive=True))


def load_csv_counts(csv_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not csv_path.exists():
        return counts
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("synsetId", row.get("subSynsetId", "unknown")).strip()
            counts[cat] = counts.get(cat, 0) + 1
    return counts


# ── Command Handlers ──────────────────────────────────────────────────────────
def handle_status(args: argparse.Namespace) -> None:
    """Print a complete dataset status matrix across all pipeline stages."""
    print("=" * 80)
    print("                TetraDiffusion — Pipeline Status Matrix")
    print("=" * 80)

    datasets = [
        ("OpenOrganelle", REPO_ROOT / "data", REPO_ROOT / "lib" / "all.csv", ["Lysosome", "Mitochondria", "Golgi", "ER"]),
        ("UroCell", REPO_ROOT / "data_urocell", REPO_ROOT / "lib" / "all_urocell.csv", ["lyso", "mito", "fv"]),
    ]

    for name, root, csv_file, categories in datasets:
        print(f"\n▶ Dataset: {name}")
        print(f"  Root path: {root}")
        print(f"  CSV path : {csv_file}")
        csv_counts = load_csv_counts(csv_file)

        header = f"  │ {'Category':<15} │ {'Raw OBJs':<10} │ {'Preprocessed':<12} │ {'CSV Reg.':<10} │ {'Checkpoints':<12} │ {'Inferred OBJs':<14} │"
        sep = f"  ├─{'─'*15}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*14}─┤"
        print(f"  ┌─{'─'*15}─┬─{'─'*10}─┬─{'─'*12}─┬─{'─'*10}─┬─{'─'*12}─┬─{'─'*14}─┐")
        print(header)
        print(sep)

        raw_base = root / "organelles_raw"
        pre_base = root / "preprocessed"
        runs_base = REPO_ROOT / "runs"

        for cat in categories:
            raw_cnt = count_files(raw_base / cat / "**" / "*.obj")
            if raw_cnt == 0:
                raw_cnt = count_files(root / "organelles" / cat / "*.obj")

            pre_cnt = count_files(pre_base / cat / "**" / "sample.pth")
            csv_cnt = csv_counts.get(cat, 0)

            # Checkpoints in runs matching cat
            ckpt_cnt = 0
            inf_cnt = 0
            if runs_base.exists():
                for run_dir in runs_base.iterdir():
                    if run_dir.is_dir() and cat.lower() in run_dir.name.lower():
                        ckpt_cnt += count_files(run_dir / "model-*.pt")
                        inf_cnt += count_files(run_dir / "**" / "*.obj")

            print(f"  │ {cat:<15} │ {raw_cnt:<10} │ {pre_cnt:<12} │ {csv_cnt:<10} │ {ckpt_cnt:<12} │ {inf_cnt:<14} │")

        print(f"  └─{'─'*15}─┴─{'─'*10}─┴─{'─'*12}─┴─{'─'*10}─┴─{'─'*12}─┴─{'─'*14}─┘")

    print("\n" + "=" * 80 + "\n")


def handle_ingest(args: argparse.Namespace) -> None:
    """Reorganize raw flat OBJ files into category/model_id/model_id.obj layout."""
    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not input_root.exists():
        print(f"ERROR: Input directory does not exist: {input_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting raw OBJs from: {input_root}")
    print(f"Target structured layout: {output_root}")

    obj_files = list(input_root.glob("**/*.obj"))
    print(f"Found {len(obj_files)} OBJ files to organize.")

    copied = 0
    skipped = 0
    for obj_path in obj_files:
        cat = obj_path.parent.name
        model_id = obj_path.stem
        dest_dir = output_root / cat / model_id
        dest_file = dest_dir / obj_path.name

        if dest_file.exists():
            skipped += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(obj_path, dest_file)
        copied += 1

    print(f"Done. Copied: {copied}, Skipped (already existed): {skipped}")


def handle_preprocess(args: argparse.Namespace) -> None:
    """Run or submit DMTet grid fitting for a dataset/category."""
    dataset = args.dataset.lower()
    cat = args.category

    if dataset == "urocell":
        input_root = REPO_ROOT / "data_urocell" / "organelles_raw"
        output_root = REPO_ROOT / "data_urocell" / "preprocessed"
        all_csv = REPO_ROOT / "lib" / "all_urocell.csv"
    else:
        input_root = REPO_ROOT / "data" / "organelles_raw"
        output_root = REPO_ROOT / "data" / "preprocessed"
        all_csv = REPO_ROOT / "lib" / "all.csv"

    if shutil.which("sbatch") and not args.local:
        cmd = [
            "sbatch",
            str(REPO_ROOT / "submit_preprocess.sh"),
            "--input_root", str(input_root),
            "--output_root", str(output_root),
            "--all_csv", str(all_csv),
            "--sanitize",
        ]
        if cat:
            cmd.extend(["--category", cat])
        print(f"Submitting SLURM preprocessing job: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "preprocessing" / "fit_many.py"),
            "--input_root", str(input_root),
            "--output_root", str(output_root),
            "--dmtet_grid", str(args.grid_res),
            "--iter", str(args.iter),
            "--update_all_csv", str(all_csv),
            "--sanitize",
        ]
        if cat:
            cmd.extend(["--category", cat])
        print(f"Running local Python preprocessing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def handle_register(args: argparse.Namespace) -> None:
    """Register preprocessed models into CSV."""
    dataset = args.dataset.lower()
    cat = args.category

    all_csv = "lib/all_urocell.csv" if dataset == "urocell" else "lib/all.csv"
    data_root = "data_urocell/preprocessed" if dataset == "urocell" else "data/preprocessed"

    categories = [cat] if cat else (["lyso", "mito", "fv"] if dataset == "urocell" else ["Lysosome", "Mitochondria", "Golgi", "ER"])

    cmd = [
        sys.executable,
        str(REPO_ROOT / "register_category.py"),
        *categories,
        "--data_root", data_root,
        "--all_csv", all_csv,
    ]
    if args.dry_run:
        cmd.append("--dry_run")

    print(f"Registering category models: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def handle_train(args: argparse.Namespace) -> None:
    """Launch production training script."""
    dataset = args.dataset.lower()
    script = "launch_production_runs_urocell.sh" if dataset == "urocell" else "launch_production_runs.sh"

    cmd = ["bash", str(REPO_ROOT / script)]
    if args.dry_run:
        cmd.append("--dry_run")
    if args.resume:
        cmd.append("--resume")
    if args.category:
        cmd.extend(["--category", args.category])

    print(f"Launching production training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def handle_infer(args: argparse.Namespace) -> None:
    """Launch mesh sampling / inference."""
    cmd = ["bash", str(REPO_ROOT / "launch_inference.sh")]
    if args.run_name:
        cmd.extend(["--run_name", args.run_name])
    if args.num_images:
        cmd.extend(["--num_images", str(args.num_images)])

    print(f"Launching inference: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def handle_evaluate(args: argparse.Namespace) -> None:
    """Run metrics comparison and plot interactive HTML shape space."""
    dataset = args.dataset.lower()

    csv_path = "lib/all_urocell.csv" if dataset == "urocell" else "lib/all.csv"
    data_path = "data_urocell/preprocessed" if dataset == "urocell" else "data/preprocessed"

    print("Step 1: Running quantitative evaluation metrics (compare.py)...")
    comp_cmd = [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "compare.py"),
        "--data_path", data_path,
        "--csv_path", csv_path,
    ]
    subprocess.run(comp_cmd, check=False)

    print("\nStep 2: Plotting interactive HTML shape-space distribution...")
    html_cmd = [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "plot_shape_space_html.py"),
        "--data_path", data_path,
        "--csv_path", csv_path,
    ]
    subprocess.run(html_cmd, check=False)
    print("\nEvaluation complete! Interactive shape-space HTML generated.")


def handle_sync(args: argparse.Namespace) -> None:
    """Sync repository and runs from remote cluster to local machine/VM."""
    sync_script = REPO_ROOT / "pipeline" / "sync_from_cluster.sh"
    cmd = ["bash", str(sync_script), args.remote]
    if args.target:
        cmd.append(args.target)
    if args.watch:
        cmd.append("--watch")
    if args.interval:
        cmd.extend(["--interval", str(args.interval)])
    if args.dry_run:
        cmd.append("--dry_run")

    print(f"Executing rsync pipeline script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def handle_dashboard(args: argparse.Namespace) -> None:
    """Launch interactive Gradio visual suite."""
    dashboard_path = REPO_ROOT / "pipeline" / "dashboard.py"
    cmd = [sys.executable, str(dashboard_path), "--port", str(args.port)]
    if args.share:
        cmd.append("--share")

    print(f"Starting TetraDiffusion Visual Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── Parser Setup ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TetraDiffusion Unified Pipeline Orchestrator & Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Print dataset and pipeline status matrix")
    p_status.set_defaults(func=handle_status)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Reorganize raw OBJs into structured directory format")
    p_ingest.add_argument("--input_dir", required=True, help="Input directory containing raw .obj files")
    p_ingest.add_argument("--output_dir", required=True, help="Target directory (e.g. data_urocell/organelles_raw)")
    p_ingest.set_defaults(func=handle_ingest)

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Run DMTet grid fitting preprocessing")
    p_pre.add_argument("--dataset", choices=["openorganelle", "urocell"], default="urocell", help="Dataset name")
    p_pre.add_argument("--category", help="Specific category (e.g. fv, mito, lyso)")
    p_pre.add_argument("--grid_res", type=int, default=128, help="Tetrahedral grid resolution")
    p_pre.add_argument("--iter", type=int, default=3000, help="Optimization iterations")
    p_pre.add_argument("--local", action="store_true", help="Force local Python run instead of SLURM sbatch")
    p_pre.set_defaults(func=handle_preprocess)

    # register
    p_reg = subparsers.add_parser("register", help="Audit and register preprocessed samples into CSV")
    p_reg.add_argument("--dataset", choices=["openorganelle", "urocell"], default="urocell", help="Dataset name")
    p_reg.add_argument("--category", help="Specific category (e.g. fv, mito, lyso)")
    p_reg.add_argument("--dry_run", action="store_true", help="Preview without writing CSV")
    p_reg.set_defaults(func=handle_register)

    # train
    p_train = subparsers.add_parser("train", help="Launch or resume production training runs")
    p_train.add_argument("--dataset", choices=["openorganelle", "urocell"], default="urocell", help="Dataset name")
    p_train.add_argument("--category", help="Target category (e.g. fv, mito, lyso)")
    p_train.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p_train.add_argument("--dry_run", action="store_true", help="Preview sbatch commands without submitting")
    p_train.set_defaults(func=handle_train)

    # infer
    p_infer = subparsers.add_parser("infer", help="Generate 3D meshes from trained model checkpoints")
    p_infer.add_argument("--run_name", help="Specific run directory name in runs/")
    p_infer.add_argument("--num_images", type=int, default=8, help="Number of meshes to sample")
    p_infer.set_defaults(func=handle_infer)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute metrics and plot interactive HTML shape space")
    p_eval.add_argument("--dataset", choices=["openorganelle", "urocell"], default="urocell", help="Dataset name")
    p_eval.set_defaults(func=handle_evaluate)

    # sync
    p_sync = subparsers.add_parser("sync", help="Sync runs and datasets from HPC cluster to local machine/VM")
    p_sync.add_argument("--remote", required=True, help="Remote source (e.g. user@login-frida:/path/to/TetraDiffusion)")
    p_sync.add_argument("--target", help="Local destination directory")
    p_sync.add_argument("--watch", action="store_true", help="Run rsync continuously every N seconds")
    p_sync.add_argument("--interval", type=int, default=30, help="Interval in seconds for watch mode")
    p_sync.add_argument("--dry_run", action="store_true", help="Preview files to be synced without copying")
    p_sync.set_defaults(func=handle_sync)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch the interactive web browser dashboard")
    p_dash.add_argument("--port", type=int, default=7860, help="Gradio server port")
    p_dash.add_argument("--share", action="store_true", help="Create public share link")
    p_dash.set_defaults(func=handle_dashboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
