"""
scripts/link_images_to_samples.py
-----------------------------------
Copy (or symlink) extracted 2D conditioning images from the staging directory
to sit alongside each sample.pth in the final data directory.

After preprocessing/fit_many.py writes sample.pth files, this script:
  1. For every instance directory in staging/{organelle}/:
     a. Verifies that output/{organelle}/{instance}/mesh_data/sample.pth exists.
     b. Copies images/*.npy and meta.json into mesh_data/.
     c. Cross-checks copied filenames against meta.json's images_generated list.
  2. Writes output/{organelle}/link_manifest.json with SHA-256 checksums.
  3. Exits non-zero if --verify strict (default) and any mismatch is found.

Usage
-----
    python scripts/link_images_to_samples.py \\
        --staging_root  /data/staging/ \\
        --output_root   /data/organelles/ \\
        --organelle_id  mito \\
        --mode          copy \\
        --verify        strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _load_meta(meta_path: Path) -> Optional[Dict]:
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Link extracted images to their sample.pth counterparts."
    )
    p.add_argument("--staging_root",  type=Path, required=True,
                   help="Root of the staging directory (input of fit_many.py).")
    p.add_argument("--output_root",   type=Path, required=True,
                   help="Root of the data directory (output of fit_many.py).")
    p.add_argument("--organelle_id",  type=str,  required=True,
                   help="Organelle class folder name (e.g. mito).")
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                   help="copy (portable, default) or symlink (no duplication).")
    p.add_argument(
        "--verify",
        choices=["strict", "warn", "skip"],
        default="strict",
        help=(
            "strict=fail on any mismatch (default); "
            "warn=print warnings but continue; "
            "skip=no validation."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    staging_organelle = args.staging_root.resolve() / args.organelle_id
    output_organelle  = args.output_root.resolve()  / args.organelle_id

    if not staging_organelle.exists():
        print(f"ERROR: staging directory not found: {staging_organelle}",
              file=sys.stderr)
        return 1

    instance_dirs = sorted(
        d for d in staging_organelle.iterdir() if d.is_dir()
    )
    if not instance_dirs:
        print(f"No instance directories found in {staging_organelle}.")
        return 0

    print(f"Found {len(instance_dirs)} instance(s) in staging. Mode: {args.mode}.")

    manifest: Dict = {
        "linked":             [],
        "missing_sample_pth": [],
        "missing_images":     [],
        "meta_mismatch":      [],
        "checksums":          {},
    }
    issues: List[str] = []

    for inst_dir in instance_dirs:
        instance_id = inst_dir.name
        sample_pth  = output_organelle / instance_id / "mesh_data" / "sample.pth"
        mesh_data   = sample_pth.parent

        # ── Check sample.pth exists ────────────────────────────────────
        if not sample_pth.exists():
            msg = f"[{instance_id}] sample.pth not found at {sample_pth}"
            manifest["missing_sample_pth"].append(instance_id)
            issues.append(msg)
            print(f"  WARN  {msg}")
            continue

        # ── Load meta.json (optional but used for cross-check) ─────────
        meta = _load_meta(inst_dir / "meta.json")
        expected_images = set(meta["images_generated"]) if meta else None

        # ── Copy images ────────────────────────────────────────────────
        images_dir = inst_dir / "images"
        if not images_dir.exists():
            msg = f"[{instance_id}] images/ directory not found in staging"
            manifest["missing_images"].append(instance_id)
            issues.append(msg)
            print(f"  WARN  {msg}")
            continue

        copied_images: List[str] = []
        for src in sorted(images_dir.glob("image_xy_p*.npy")):
            dst = mesh_data / src.name
            _copy_file(src, dst, args.mode)
            copied_images.append(src.name)
            manifest["checksums"][f"{instance_id}/{src.name}"] = _sha256(dst)

        # Copy meta.json
        if (inst_dir / "meta.json").exists():
            dst_meta = mesh_data / "meta.json"
            _copy_file(inst_dir / "meta.json", dst_meta, args.mode)

        # ── Cross-check against meta.json ─────────────────────────────
        if expected_images is not None and args.verify != "skip":
            copied_set  = set(copied_images)
            missing_in  = expected_images - copied_set
            extra_in    = copied_set - expected_images
            if missing_in or extra_in:
                msg = (
                    f"[{instance_id}] image set mismatch: "
                    f"missing={sorted(missing_in)}, extra={sorted(extra_in)}"
                )
                manifest["meta_mismatch"].append(instance_id)
                issues.append(msg)
                print(f"  WARN  {msg}")
            else:
                manifest["linked"].append(instance_id)
                print(f"  OK    {instance_id}: {len(copied_images)} image(s) linked")
        else:
            manifest["linked"].append(instance_id)
            print(f"  OK    {instance_id}: {len(copied_images)} image(s) linked")

    # ── Write link_manifest.json ───────────────────────────────────────
    manifest_path = output_organelle / "link_manifest.json"
    output_organelle.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # ── Summary ────────────────────────────────────────────────────────
    n_linked   = len(manifest["linked"])
    n_total    = len(instance_dirs)
    n_issues   = len(issues)
    print(f"Linked {n_linked}/{n_total} instances.  Issues: {n_issues}.")

    if issues and args.verify == "strict":
        print(
            f"\nERROR: {n_issues} issue(s) found in strict mode. "
            "Fix the above warnings and re-run.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
