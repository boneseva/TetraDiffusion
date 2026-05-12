#!/usr/bin/env python3
"""
register_category.py

Register all preprocessed models from a category into lib/all.csv so that
training can discover them.

Usage:
    python3 register_category.py --category Golgi
    python3 register_category.py --category Golgi --data_root data/preprocessed
    python3 register_category.py --category Golgi ER Mitochondria   # multiple at once

Assumes layout:
    <data_root>/<category>/<model_id>/mesh_data/sample.pth

Repo root is auto-detected as the directory containing this script.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register preprocessed category models into lib/all.csv"
    )
    parser.add_argument(
        "category",
        nargs="+",
        help="One or more category names (must match subfolder names under --data_root)",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/preprocessed"),
        help="Root folder containing <category>/<model_id>/mesh_data/sample.pth (default: data/preprocessed)",
    )
    parser.add_argument(
        "--all_csv",
        type=Path,
        default=Path("lib/all.csv"),
        help="Path to all.csv (default: lib/all.csv)",
    )
    parser.add_argument(
        "--require_sample_pth",
        action="store_true",
        default=True,
        help="Only register models that have mesh_data/sample.pth (default: True)",
    )
    parser.add_argument(
        "--no_require_sample_pth",
        dest="require_sample_pth",
        action="store_false",
        help="Register all subdirectories, even if sample.pth is missing",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be added without writing anything",
    )
    return parser.parse_args()


def load_existing_csv(all_csv: Path) -> tuple[set[str], int]:
    """Returns (set of existing modelIds, max numeric index found)."""
    existing: set[str] = set()
    max_idx = 0
    if not all_csv.exists():
        return existing, max_idx
    with all_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row.get("modelId", "").strip()
            if mid:
                existing.add(mid)
            raw_id = row.get("id", "")
            try:
                max_idx = max(max_idx, int(raw_id.replace("custom_", "")))
            except ValueError:
                pass
    return existing, max_idx


def discover_models(data_root: Path, category: str, require_sample_pth: bool) -> list[str]:
    cat_dir = data_root / category
    if not cat_dir.is_dir():
        print(f"  ERROR: category directory not found: {cat_dir}", file=sys.stderr)
        return []
    models = []
    for model_dir in sorted(cat_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if require_sample_pth and not (model_dir / "mesh_data" / "sample.pth").exists():
            print(f"  SKIP (no sample.pth): {model_dir.name}")
            continue
        models.append(model_dir.name)
    return models


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    args = parse_args()

    all_csv = args.all_csv
    if not all_csv.is_absolute():
        all_csv = (repo_root / all_csv).resolve()

    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()

    if not data_root.exists():
        print(f"ERROR: data_root not found: {data_root}", file=sys.stderr)
        return 1

    existing, max_idx = load_existing_csv(all_csv)
    print(f"all.csv: {all_csv}")
    print(f"Existing entries: {len(existing)}, max index: custom_{max_idx:07d}")
    print()

    to_add: list[tuple[str, str]] = []  # (category, model_id)

    for category in args.category:
        models = discover_models(data_root, category, args.require_sample_pth)
        if not models:
            print(f"[{category}] No models found.")
            continue

        new = [(category, m) for m in models if m not in existing]
        already = len(models) - len(new)
        print(f"[{category}] Found {len(models)} models, {already} already registered, {len(new)} to add:")
        for _, m in new:
            print(f"  + {m}")
        to_add.extend(new)

    print()

    if not to_add:
        print("Nothing to add. all.csv is up to date.")
        return 0

    if args.dry_run:
        print(f"DRY RUN: would add {len(to_add)} rows — not writing.")
        return 0

    # Append new rows
    write_header = not all_csv.exists()
    all_csv.parent.mkdir(parents=True, exist_ok=True)
    with all_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "synsetId", "subSynsetId", "modelId", "split"])
        for i, (category, model_id) in enumerate(to_add):
            idx = f"custom_{max_idx + 1 + i:07d}"
            writer.writerow([idx, category, category, model_id, "train"])
            print(f"  Added: {idx},{category},{category},{model_id},train")

    print()
    print(f"Done. Added {len(to_add)} row(s) to {all_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

