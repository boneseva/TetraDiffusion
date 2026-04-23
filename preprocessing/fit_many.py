"""
Batch-convert many OBJ meshes into tetrahedral sample.pth files using preprocessing/train.py.

Expected input layout by default:
    <input_root>/<class_id>/<model_id>/**/*.obj

Output layout written for training (MeshLoader-compatible):
    <output_root>/<class_id>/<model_id>/mesh_data/sample.pth

Notes:
- This script orchestrates the existing CUDA-heavy optimizer in train.py.
- train.py always writes under preprocessing/out/, so this script copies results to output_root.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fit .obj meshes into tetrahedral samples")
    parser.add_argument("--input_root", type=Path, required=True, help="Root folder containing class/model OBJ files")
    parser.add_argument("--output_root", type=Path, required=True, help="Root folder to write class/model/mesh_data/sample.pth")
    parser.add_argument(
        "--obj_glob",
        type=str,
        default="*/*/**/*.obj",
        help="Glob relative to input_root. Default assumes <class>/<model>/.../*.obj",
    )
    parser.add_argument(
        "--config_template",
        type=Path,
        default=Path("configs/shapenet.json"),
        help="Template JSON consumed by preprocessing/train.py",
    )
    parser.add_argument("--dmtet_grid", type=int, default=128, choices=[64, 128, 192], help="Tetra grid resolution")
    parser.add_argument("--iter", type=int, default=3000, help="Optimization iterations per object")
    parser.add_argument("--batch", type=int, default=3, help="Renderer batch size per optimization step")
    parser.add_argument("--train_res", type=int, nargs=2, default=[1024, 1024], help="Training image resolution")
    parser.add_argument("--texture_res", type=int, nargs=2, default=[512, 512], help="Texture resolution")
    parser.add_argument("--overwrite", action="store_true", help="Re-fit even if output sample.pth already exists")
    parser.add_argument("--dry_run", action="store_true", help="Print planned jobs and exit")
    parser.add_argument("--sanitize", action="store_true", help="Sanitize OBJ files in-place before fitting (creates .bak backups)")
    parser.add_argument(
        "--update_all_csv",
        type=Path,
        default=None,
        help="Optional path to all.csv (e.g. ../lib/all.csv). Appends missing modelIds with split=train.",
    )
    return parser.parse_args()


def discover_jobs(input_root: Path, obj_glob: str) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    for obj_path in sorted(input_root.glob(obj_glob)):
        if not obj_path.is_file() or obj_path.suffix.lower() != ".obj":
            continue

        rel = obj_path.relative_to(input_root)
        if len(rel.parts) < 2:
            # Need at least class/model to match training loader assumptions.
            continue

        class_id = rel.parts[0]
        model_id = rel.parts[1]
        jobs.append((class_id, model_id, obj_path))

    return jobs


def load_template(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_config(config: dict, cfg_path: Path) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)


def run_train(preprocessing_dir: Path, cfg_path: Path) -> None:
    cmd = [sys.executable, "train.py", "--config", str(cfg_path)]
    subprocess.run(cmd, cwd=preprocessing_dir, check=True)


def append_missing_all_csv_rows(all_csv_path: Path, rows: Iterable[Tuple[str, str]]) -> int:
    all_csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_model_ids = set()
    has_file = all_csv_path.exists()
    if has_file:
        with all_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("modelId")
                if mid:
                    existing_model_ids.add(mid)

    header = ["id", "synsetId", "subSynsetId", "modelId", "split"]
    to_add = []
    for class_id, model_id in rows:
        if model_id in existing_model_ids:
            continue
        to_add.append((class_id, model_id))
        existing_model_ids.add(model_id)

    if not to_add:
        return 0

    mode = "a" if has_file else "w"
    with all_csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not has_file:
            writer.writerow(header)
        start_idx = len(existing_model_ids) - len(to_add)
        for offset, (class_id, model_id) in enumerate(to_add):
            idx = f"custom_{start_idx + offset:07d}"
            writer.writerow([idx, class_id, class_id, model_id, "train"])

    return len(to_add)


def main() -> int:
    args = parse_args()

    preprocessing_dir = Path(__file__).resolve().parent
    repo_root = preprocessing_dir.parent

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    template_path = args.config_template
    if not template_path.is_absolute():
        template_path = (preprocessing_dir / template_path).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not template_path.exists():
        raise FileNotFoundError(f"config_template not found: {template_path}")

    jobs = discover_jobs(input_root, args.obj_glob)
    if not jobs:
        print("No OBJ files found. Check --input_root and --obj_glob.")
        return 1

    print(f"Discovered {len(jobs)} OBJ files")
    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")

    if args.dry_run:
        for class_id, model_id, obj_path in jobs[:20]:
            print(f"[DRY] {class_id}/{model_id} <- {obj_path}")
        if len(jobs) > 20:
            print(f"... and {len(jobs) - 20} more")
        return 0

    base_cfg = load_template(template_path)
    success = 0
    failed: List[Tuple[str, str, str]] = []
    seen_pairs = []

    for idx, (class_id, model_id, obj_path) in enumerate(jobs, start=1):
        target_sample = output_root / class_id / model_id / "mesh_data" / "sample.pth"
        seen_pairs.append((class_id, model_id))

        if target_sample.exists() and not args.overwrite:
            print(f"[{idx}/{len(jobs)}] Skip existing: {target_sample}")
            success += 1
            continue

        rel_out_dir = Path(class_id) / model_id
        train_out_dir = preprocessing_dir / "out" / rel_out_dir
        cfg_path = preprocessing_dir / "configs" / class_id / f"{model_id}.json"

        # Optionally sanitize the OBJ in-place (creates .bak backup via sanitize_obj.py)
        if args.sanitize:
            if args.dry_run:
                print(f"[{idx}/{len(jobs)}] [DRY] Would sanitize: {obj_path}")
            else:
                try:
                    print(f"[{idx}/{len(jobs)}] Sanitizing: {obj_path}")
                    subprocess.run([sys.executable, "sanitize_obj.py", str(obj_path), "--inplace"], cwd=preprocessing_dir, check=True)
                except Exception as e:  # noqa: BLE001
                    print(f"    WARNING: sanitizer failed for {obj_path}: {e}")

        cfg = dict(base_cfg)
        cfg["ref_mesh"] = str(obj_path)
        # Only include an mtl_override if the .mtl file actually exists. If omitted or null,
        # train.py will use its default (None) and obj.load_obj() will fall back to parsing
        # mtllib inside the OBJ or using the default material.
        mtl_path = obj_path.with_suffix(".mtl")
        if mtl_path.exists():
            cfg["mtl_override"] = str(mtl_path)
        cfg["out_dir"] = str(rel_out_dir).replace("\\", "/")
        cfg["dmtet_grid"] = args.dmtet_grid
        cfg["iter"] = args.iter
        cfg["batch"] = args.batch
        cfg["train_res"] = list(args.train_res)
        cfg["texture_res"] = list(args.texture_res)

        try:
            write_config(cfg, cfg_path)
            print(f"[{idx}/{len(jobs)}] Fitting {class_id}/{model_id}")
            run_train(preprocessing_dir=preprocessing_dir, cfg_path=cfg_path)

            produced = train_out_dir / "mesh_data" / "sample.pth"
            if not produced.exists():
                raise FileNotFoundError(f"train.py finished but did not produce {produced}")

            target_sample.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, target_sample)

            for npz_name in ["deform.npz", "color.npz", "sdf.npz", "mesh.obj"]:
                src = train_out_dir / "mesh_data" / npz_name
                if src.exists():
                    shutil.copy2(src, target_sample.parent / npz_name)

            success += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((class_id, model_id, str(exc)))
            print(f"    FAILED {class_id}/{model_id}: {exc}")

    print("\nDone")
    print(f"Success: {success}/{len(jobs)}")
    print(f"Failed:  {len(failed)}")

    if args.update_all_csv is not None:
        csv_path = args.update_all_csv
        if not csv_path.is_absolute():
            csv_path = (repo_root / csv_path).resolve()
        added = append_missing_all_csv_rows(csv_path, seen_pairs)
        print(f"Updated {csv_path} with {added} new modelId row(s)")

    if failed:
        fail_log = output_root / "fit_many_failures.csv"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with fail_log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "model_id", "error"])
            writer.writerows(failed)
        print(f"Failure log: {fail_log}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

