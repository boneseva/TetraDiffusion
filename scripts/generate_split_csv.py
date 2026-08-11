"""
scripts/generate_split_csv.py
-------------------------------
Create a subvolume-grouped train/val split and write rows to lib/all.csv.

All instances from one subvolume land in either train OR val — never both.
This prevents spatial-texture leakage between splits.

Behaviour
---------
- Reads the extraction manifest CSV produced by extract_instances_from_nifti.py.
- Shuffles subvolumes with a fixed seed and assigns the first val_ratio
  fraction to val; the rest to train.
- **Upserts** rows into lib/all.csv:
    - New modelId → appended.
    - Existing modelId with the same split → silently skipped.
    - Existing modelId with a CONFLICTING split → hard error (exit 1).
  This makes running fit_many --update_all_csv safe: it can only append
  rows that do not yet exist, and conflicts are caught here.

Usage
-----
    python scripts/generate_split_csv.py \\
        --manifest  staging/mito_extraction_manifest.csv \\
        --all_csv   lib/all.csv \\
        --val_ratio 0.2 \\
        --seed      42
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _read_manifest(path: Path) -> List[Tuple[str, str, str]]:
    """Return (subvol_id, instance_id, organelle_id) for every ok row."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                rows.append((
                    row["subvol_id"],
                    row["instance_id"],
                    row.get("organelle_id", "unknown"),
                ))
    return rows


def _read_existing_all_csv(path: Path) -> Dict[str, str]:
    """Return {modelId: split} from an existing all.csv, or {} if absent."""
    if not path.exists():
        return {}
    existing: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("modelId", "").strip()
            if mid:
                existing[mid] = row.get("split", "train").strip()
    return existing


def _append_rows(
    path: Path,
    to_add: List[Tuple[str, str, str]],  # (model_id, split, organelle_id)
    start_idx: int,
) -> None:
    """Append new rows to all.csv (creates file + header if absent)."""
    header = ["id", "synsetId", "subSynsetId", "modelId", "split"]
    needs_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(header)
        for offset, (model_id, split, organelle_id) in enumerate(to_add):
            idx = f"imgcond_{start_idx + offset:07d}"
            writer.writerow([idx, organelle_id, organelle_id, model_id, split])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate subvolume-grouped train/val split into lib/all.csv."
    )
    p.add_argument("--manifest",  type=Path, required=True,
                   help="Extraction manifest CSV from extract_instances_from_nifti.py.")
    p.add_argument("--all_csv",   type=Path, required=True,
                   help="Path to lib/all.csv (created or appended).")
    p.add_argument("--val_ratio", type=float, default=0.2,
                   help="Fraction of subvolumes assigned to val (default: 0.2).")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Random seed for reproducible subvolume shuffle.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not (0.0 < args.val_ratio < 1.0):
        print("ERROR: --val_ratio must be in (0, 1).", file=sys.stderr)
        return 1

    # ── Read manifest ──────────────────────────────────────────────────
    instances = _read_manifest(args.manifest)
    if not instances:
        print("No ok instances found in manifest; nothing to write.", file=sys.stderr)
        return 0

    # ── Subvolume-grouped split ────────────────────────────────────────
    subvol_ids = sorted({sv for sv, _, _ in instances})
    rng = random.Random(args.seed)
    rng.shuffle(subvol_ids)
    n_val = max(1, int(len(subvol_ids) * args.val_ratio))
    val_subvols = set(subvol_ids[:n_val])
    train_subvols = set(subvol_ids[n_val:])

    print(f"Subvolumes: {len(subvol_ids)} total, "
          f"{len(train_subvols)} train, {len(val_subvols)} val")
    print(f"  Val subvols : {sorted(val_subvols)}")
    print(f"  Train subvols: {sorted(train_subvols)}")

    intended: Dict[str, Tuple[str, str]] = {}  # model_id → (split, organelle_id)
    for subvol_id, instance_id, organelle_id in instances:
        split = "val" if subvol_id in val_subvols else "train"
        intended[instance_id] = (split, organelle_id)

    # ── Read existing all.csv ──────────────────────────────────────────
    existing = _read_existing_all_csv(args.all_csv)

    # ── Upsert with conflict detection ────────────────────────────────
    conflicts: List[Tuple[str, str, str]] = []
    to_add:    List[Tuple[str, str, str]] = []  # (model_id, split, organelle_id)

    for model_id, (intended_split, organelle_id) in intended.items():
        if model_id in existing:
            if existing[model_id] != intended_split:
                conflicts.append((model_id, existing[model_id], intended_split))
            # else: already correct → skip silently
        else:
            to_add.append((model_id, intended_split, organelle_id))

    if conflicts:
        lines = "\n".join(
            f"  {m}: existing={e}, proposed={n}"
            for m, e, n in conflicts
        )
        print(
            f"ERROR: Split conflicts detected for {len(conflicts)} modelId(s):\n"
            f"{lines}\n"
            f"To resolve: remove the conflicting rows from {args.all_csv} and re-run.",
            file=sys.stderr,
        )
        return 1

    if not to_add:
        print(f"All {len(intended)} instances already present in {args.all_csv}; "
              "nothing to add.")
        return 0

    _append_rows(args.all_csv, to_add, start_idx=len(existing))

    n_train = sum(1 for _, s, _ in to_add if s == "train")
    n_val_added = sum(1 for _, s, _ in to_add if s == "val")
    print(f"Appended {len(to_add)} rows to {args.all_csv} "
          f"({n_train} train, {n_val_added} val).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
