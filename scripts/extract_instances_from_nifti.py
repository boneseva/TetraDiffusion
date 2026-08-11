"""
scripts/extract_instances_from_nifti.py
----------------------------------------
Extract individual organelle instances from registered NIfTI microscopy
volumes and generate paired 2D conditioning images for TetraDiffusion.

For each organelle instance in a binary segmentation mask:
  1. Run connected-component labeling (configurable connectivity).
  2. Exclude instances touching the volume border (truncated shapes).
  3. Exclude instances below a minimum voxel count (noise/fragments).
  4. Crop the raw EM volume to the 3D bounding box of the instance.
  5. For each of 7 Z-depth positions, extract a tight 2D cross-section crop
     of the EM data around the visible organelle pixels and save as .npy.
  6. Export the 3D binary mask as an OBJ mesh (marching cubes) for the
     tetrahedral fitting pipeline.
  7. Write per-instance meta.json and a global extraction manifest CSV.

Usage (Option A — directory with matched prefixes):
    python scripts/extract_instances_from_nifti.py \\
        --input_dir   /data/niftis/ \\
        --em_suffix   _em.nii.gz \\
        --seg_suffix  _seg_mito.nii.gz \\
        --organelle_id mito \\
        --output_root  /data/staging/

Usage (Option B — explicit CSV manifest):
    python scripts/extract_instances_from_nifti.py \\
        --manifest     /data/pairs_mito.csv \\
        --organelle_id mito \\
        --output_root  /data/staging/

The manifest CSV must have columns: em_path, seg_path, subvol_id.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
import skimage.measure
import skimage.transform

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.image_preprocessing import normalize_image, prepare_slice

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required: pip install nibabel")


# ---------------------------------------------------------------------------
# Slice positions as fractions of the bounding-box Z extent
# ---------------------------------------------------------------------------
DEFAULT_SLICE_POSITIONS = [10, 25, 40, 50, 60, 75, 90]  # percentages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connectivity_structure(connectivity: int):
    """Map user-facing connectivity (6/18/26) to scipy binary structure."""
    rank_map = {6: 1, 18: 2, 26: 3}
    rank = rank_map.get(connectivity)
    if rank is None:
        raise ValueError(f"connectivity must be 6, 18, or 26; got {connectivity}")
    return scipy.ndimage.generate_binary_structure(3, rank)


def _load_nifti_pair(
    em_path: str, seg_path: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Load EM + segmentation NIfTI pair.

    Returns
    -------
    em_data   : (X, Y, Z) float32 array
    seg_data  : (X, Y, Z) bool array  (True = organelle)
    spacing   : (sx, sy, sz) physical voxel size from NIfTI header
    """
    em_img = nib.load(em_path)
    seg_img = nib.load(seg_path)
    em_data = em_img.get_fdata(dtype=np.float32)
    seg_data = (seg_img.get_fdata() > 0.5)
    zooms = em_img.header.get_zooms()
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    return em_data, seg_data, (sx, sy, sz)


def _touches_border(
    coords: np.ndarray, vol_shape: np.ndarray, border_mode: str = "strict"
) -> bool:
    """Check if coordinates touch volume boundary according to border_mode.

    border_mode:
        "strict"  : True if touching ANY boundary (X, Y, or Z).
        "allow_z" : True if touching X or Y boundaries (Z boundary touch is allowed).
        "none"    : Always False (no border filtering).
    """
    if border_mode == "none":
        return False

    # Check X (axis 0) and Y (axis 1) boundaries
    if (coords[:, 0] == 0).any() or (coords[:, 0] == vol_shape[0] - 1).any():
        return True
    if (coords[:, 1] == 0).any() or (coords[:, 1] == vol_shape[1] - 1).any():
        return True

    if border_mode == "allow_z":
        return False

    # strict mode: check Z (axis 2) boundary as well
    if (coords[:, 2] == 0).any() or (coords[:, 2] == vol_shape[2] - 1).any():
        return True

    return False


def _tight_2d_bbox(
    seg_slice: np.ndarray, pad: int
) -> Optional[Tuple[int, int, int, int]]:
    """Compute padded 2D bounding box of foreground pixels.

    Returns (r0, r1, c0, c1) clamped to array bounds, or None if empty.
    """
    rows = np.any(seg_slice, axis=1)
    cols = np.any(seg_slice, axis=0)
    if not rows.any():
        return None
    r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    H, W = seg_slice.shape
    r0 = max(r0 - pad, 0)
    r1 = min(r1 + pad, H - 1)
    c0 = max(c0 - pad, 0)
    c1 = min(c1 + pad, W - 1)
    return r0, r1, c0, c1


def _write_obj(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write a triangle mesh to an OBJ file (1-indexed faces)."""
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def _process_instance(
    mask: np.ndarray,
    em_data: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    args: argparse.Namespace,
    out_dir: Path,
    subvol_id: str,
    em_path: Path,
    seg_path: Path,
) -> Tuple[str, int, List[str]]:
    """Process one connected-component instance.

    Returns
    -------
    (status, n_voxels, images_generated)
    status in: "ok", "border_touching", "too_small", "no_valid_slices",
                "marching_cubes_failed:<msg>"
    """
    coords = np.argwhere(mask)
    vol_shape = np.array(em_data.shape)
    sx, sy, sz = voxel_spacing
    n_voxels = int(mask.sum())

    # ── Quality filters ────────────────────────────────────────────────
    border_mode = getattr(args, "border_mode", "strict")
    if _touches_border(coords, vol_shape, border_mode=border_mode):
        return "border_touching", n_voxels, []
    if n_voxels < args.min_voxels:
        return "too_small", n_voxels, []

    # ── 3D crop (bounding box + padding, clamped) ──────────────────────
    bbox_min = np.clip(coords.min(axis=0) - args.pad_voxels, 0, vol_shape - 1)
    bbox_max = np.clip(coords.max(axis=0) + args.pad_voxels + 1, 0, vol_shape)
    sl = tuple(slice(int(bbox_min[i]), int(bbox_max[i])) for i in range(3))

    em_crop  = em_data[sl].astype(np.float32)
    seg_crop = mask[sl]
    if args.context == "masked":
        em_crop = em_crop * seg_crop.astype(np.float32)

    Nx, Ny, Nz = em_crop.shape

    # ── 2D XY slice images ─────────────────────────────────────────────
    images_generated: List[str] = []
    images_skipped:   Dict[str, str] = {}

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    slice_fracs = [p / 100.0 for p in args.slice_positions]

    for pct_frac in slice_fracs:
        pct_int = int(round(pct_frac * 100))
        z_idx = int(round(pct_frac * (Nz - 1)))
        em_slice  = em_crop[:, :, z_idx]    # (Nx, Ny) — axis 0=x(sx), axis 1=y(sy)
        seg_slice = seg_crop[:, :, z_idx]   # (Nx, Ny) bool

        fname = f"image_xy_p{pct_int:02d}.npy"

        if seg_slice.sum() < args.min_slice_area:
            images_skipped[fname] = "too_empty"
            continue

        bbox_2d = _tight_2d_bbox(seg_slice, args.pad_pixels_2d)
        if bbox_2d is None:
            images_skipped[fname] = "too_empty"
            continue
        r0, r1, c0, c1 = bbox_2d

        # Tight EM crop around the 2D cross-section
        crop_2d = em_slice[r0 : r1 + 1, c0 : c1 + 1]

        # Anisotropy-correct + letterbox + resize (single call, shared module)
        arr = prepare_slice(crop_2d, sx=sx, sy=sy, proj_size=args.proj_size)
        arr = normalize_image(arr)

        np.save(images_dir / fname, arr)
        images_generated.append(fname)

    if not images_generated:
        return "no_valid_slices", n_voxels, []

    # ── OBJ export (marching cubes on binary seg crop) ─────────────────
    if not args.skip_obj:
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                seg_crop.astype(np.float32),
                level=0.5,
                spacing=(sx, sy, sz),
            )
            _write_obj(verts, faces, out_dir / "model.obj")
        except Exception as exc:  # noqa: BLE001
            return f"marching_cubes_failed:{exc}", n_voxels, []

    # ── meta.json ──────────────────────────────────────────────────────
    meta = {
        "instance_id":       out_dir.name,
        "subvol_id":         subvol_id,
        "em_path":           str(em_path),
        "seg_path":          str(seg_path),
        "n_voxels":          n_voxels,
        "voxel_spacing_xyz": list(voxel_spacing),
        "bbox_3d_voxels": {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
        },
        "pad_voxels_3d":     args.pad_voxels,
        "pad_pixels_2d":     args.pad_pixels_2d,
        "proj_size":         args.proj_size,
        "connectivity":      args.connectivity,
        "images_generated":  sorted(images_generated),
        "images_skipped":    images_skipped,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return "ok", n_voxels, images_generated


# ---------------------------------------------------------------------------
# Per-subvolume driver
# ---------------------------------------------------------------------------

def _process_subvolume(
    em_path: Path,
    seg_path: Path,
    subvol_id: str,
    args: argparse.Namespace,
    output_root: Path,
    manifest_rows: List[Dict],
) -> None:
    logging.info(f"[{subvol_id}] Loading {em_path.name} + {seg_path.name}")
    em_data, seg_data, voxel_spacing = _load_nifti_pair(str(em_path), str(seg_path))
    logging.info(
        f"[{subvol_id}] Volume shape {em_data.shape}, "
        f"spacing {voxel_spacing}, "
        f"foreground voxels {int(seg_data.sum())}"
    )

    struct = _connectivity_structure(args.connectivity)
    labeled, n_total = scipy.ndimage.label(seg_data, structure=struct)
    logging.info(f"[{subvol_id}] {n_total} connected components found")

    organelle_dir = output_root / args.organelle_id
    organelle_dir.mkdir(parents=True, exist_ok=True)

    counts = dict(ok=0, border_touching=0, too_small=0,
                  no_valid_slices=0, other=0)
    ok_cap = args.max_instances  # None = no cap

    for inst_idx in range(1, n_total + 1):
        if ok_cap is not None and counts["ok"] >= ok_cap:
            logging.info(f"[{subvol_id}] max_instances={ok_cap} reached; stopping.")
            break

        mask = (labeled == inst_idx)
        instance_name = f"{subvol_id}_inst_{inst_idx:06d}"
        out_dir = organelle_dir / instance_name

        if args.dry_run:
            coords     = np.argwhere(mask)
            n_voxels   = int(mask.sum())
            border_mode = getattr(args, "border_mode", "strict")
            touches    = _touches_border(coords, np.array(em_data.shape), border_mode=border_mode)
            dry_status = (
                "border_touching" if touches
                else "too_small"  if n_voxels < args.min_voxels
                else "would_extract"
            )
            logging.info(f"  [DRY] {instance_name}: {dry_status} ({n_voxels} vx)")
            manifest_rows.append(dict(
                instance_id=instance_name, subvol_id=subvol_id,
                organelle_id=args.organelle_id,
                status=dry_status, n_voxels=n_voxels, n_images=0,
                reason_skipped=dry_status if dry_status != "would_extract" else "",
                em_path=str(em_path), seg_path=str(seg_path),
            ))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        status, n_voxels, images = _process_instance(
            mask, em_data, voxel_spacing, args, out_dir,
            subvol_id, em_path, seg_path,
        )

        bucket = status if status in counts else "other"
        counts[bucket] += 1

        manifest_rows.append(dict(
            instance_id=instance_name, subvol_id=subvol_id,
            organelle_id=args.organelle_id,
            status=status, n_voxels=n_voxels, n_images=len(images),
            reason_skipped="" if status == "ok" else status,
            em_path=str(em_path), seg_path=str(seg_path),
        ))

        if status == "ok":
            logging.info(
                f"  [OK]   {instance_name}: {n_voxels} vx, {len(images)} images"
            )
        else:
            logging.info(f"  [SKIP] {instance_name}: {status} ({n_voxels} vx)")

    total_seen = sum(counts.values())
    logging.info(
        f"[{subvol_id}] Summary: {counts['ok']}/{total_seen} extracted  "
        f"(border={counts['border_touching']}, small={counts['too_small']}, "
        f"no_slices={counts['no_valid_slices']}, other={counts['other']})"
    )


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def _discover_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path, str]]:
    """Return list of (em_path, seg_path, subvol_id) tuples."""
    pairs: List[Tuple[Path, Path, str]] = []

    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((
                    Path(row["em_path"]),
                    Path(row["seg_path"]),
                    row["subvol_id"],
                ))
        return pairs

    # Option A: directory with matched filename prefixes
    input_dir = Path(args.input_dir)
    seg_dir   = Path(args.seg_dir) if args.seg_dir else input_dir
    em_suffix  = args.em_suffix
    seg_suffix = args.seg_suffix

    em_files = sorted(input_dir.glob(f"*{em_suffix}"))
    if not em_files:
        raise FileNotFoundError(
            f"No files ending with '{em_suffix}' found in {input_dir}"
        )

    for em_path in em_files:
        prefix = (
            em_path.name[: -len(em_suffix)]
            if len(em_suffix) > 0 and em_path.name.endswith(em_suffix)
            else em_path.name.split(".")[0]
        )
        seg_candidates = [
            seg_dir / f"{prefix}{seg_suffix}",
            seg_dir / "instance" / f"{prefix}{seg_suffix}",
            seg_dir / "binary" / f"{prefix}{seg_suffix}",
            seg_dir / "precise" / "binary" / f"{prefix}{seg_suffix}",
            seg_dir / "approximate" / "binary" / f"{prefix}{seg_suffix}",
            seg_dir / em_path.name,
            seg_dir / "instance" / em_path.name,
            seg_dir / "binary" / em_path.name,
            seg_dir / "precise" / "binary" / em_path.name,
            seg_dir / "approximate" / "binary" / em_path.name,
        ]
        seg_path = None
        for cand in seg_candidates:
            if cand.exists():
                seg_path = cand
                break

        if seg_path is None:
            # Fallback: search recursively inside seg_dir for any matching .nii.gz
            rglob_matches = sorted(seg_dir.rglob(f"*{prefix}*.nii.gz"))
            if rglob_matches:
                seg_path = rglob_matches[0]

        if seg_path is None:
            logging.warning(
                f"Segmentation file not found for {em_path.name} in {seg_dir}; skipping."
            )
            continue

        subvol_id = prefix.rstrip("_- ")
        pairs.append((em_path, seg_path, subvol_id))

    return pairs


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract organelle instances from NIfTI volumes."
    )

    # Input — Option A
    p.add_argument("--input_dir",  type=Path, default=None,
                   help="Directory containing EM NIfTI volumes (Option A).")
    p.add_argument("--seg_dir",    type=Path, default=None,
                   help="Directory containing segmentation NIfTI volumes (defaults to --input_dir).")
    p.add_argument("--em_suffix",  type=str, default="_em.nii.gz",
                   help="Suffix identifying EM volumes.")
    p.add_argument("--seg_suffix", type=str, default="_seg.nii.gz",
                   help="Suffix identifying segmentation volumes.")
    # Input — Option B
    p.add_argument("--manifest",   type=Path, default=None,
                   help="CSV manifest with em_path, seg_path, subvol_id columns (Option B).")

    # Required
    p.add_argument("--organelle_id", type=str, required=True,
                   help="Class name used as output folder (e.g. mito, lyso).")
    p.add_argument("--output_root",  type=Path, required=True,
                   help="Staging root folder for OBJ meshes and images.")

    # Image generation
    p.add_argument("--proj_size",      type=int,   default=64,
                   help="Output image side length in pixels.")
    p.add_argument("--pad_voxels",     type=int,   default=8,
                   help="Padding added to 3D bounding box (voxels).")
    p.add_argument("--pad_pixels_2d",  type=int,   default=4,
                   help="Padding added to tight 2D bounding box (pixels, before resize).")
    p.add_argument("--min_voxels",     type=int,   default=500,
                   help="Skip instances with fewer voxels than this.")
    p.add_argument("--min_slice_area", type=int,   default=20,
                   help="Skip 2D slices with fewer foreground pixels than this.")
    p.add_argument(
        "--slice_positions",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_SLICE_POSITIONS,
        metavar="PCT,PCT,...",
        help=(
            "Slice depths as integer percentages of bounding-box Z extent, "
            "comma-separated (default: 10,25,40,50,60,75,90)."
        ),
    )
    p.add_argument("--image_mode", choices=["xy_slices", "all"], default="xy_slices",
                   help="xy_slices=XY slices only (baseline); all=XY+XZ+YZ (experimental).")
    p.add_argument("--context", choices=["full", "masked"], default="full",
                   help="full=raw EM; masked=zero non-organelle voxels before slicing.")

    # Instance filtering
    p.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6,
                   help="Voxel connectivity for connected-component labeling.")
    p.add_argument("--border_mode", choices=["strict", "allow_z", "none"], default="strict",
                   help=(
                       "strict=exclude touching any boundary (default); "
                       "allow_z=allow touching top/bottom Z faces (filter XY side cuts only); "
                       "none=allow all border-touching instances."
                   ))
    p.add_argument("--max_instances", type=int, default=None,
                   help="Cap on extracted instances per subvolume (for debugging).")

    # Modes
    p.add_argument("--skip_obj", action="store_true",
                   help="Generate images only; do not run marching cubes or write OBJ.")
    p.add_argument("--dry_run",  action="store_true",
                   help="Print plan without writing any files.")

    args = p.parse_args()

    if args.manifest is None and args.input_dir is None:
        p.error("Provide either --manifest or --input_dir.")

    for pct in args.slice_positions:
        if not (0 <= pct <= 100):
            p.error(f"slice_positions must be in [0,100]; got {pct}")

    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(args)
    if not pairs:
        logging.error("No valid NIfTI pairs found.")
        return 1

    logging.info(
        f"Found {len(pairs)} subvolume pair(s) to process. "
        f"Organelle: {args.organelle_id}. "
        f"Output: {output_root}. "
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"{'[SKIP OBJ] ' if args.skip_obj else ''}"
    )

    manifest_rows: List[Dict] = []
    for em_path, seg_path, subvol_id in pairs:
        if not em_path.exists():
            logging.error(f"EM file not found: {em_path}")
            continue
        if not seg_path.exists():
            logging.error(f"Segmentation file not found: {seg_path}")
            continue
        _process_subvolume(
            em_path, seg_path, subvol_id, args, output_root, manifest_rows
        )

    # Write extraction manifest CSV
    manifest_path = output_root / f"{args.organelle_id}_extraction_manifest.csv"
    fieldnames = [
        "instance_id", "subvol_id", "organelle_id", "status", "n_voxels",
        "n_images", "reason_skipped", "em_path", "seg_path",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    logging.info(f"Extraction manifest written to {manifest_path}")

    n_ok = sum(1 for r in manifest_rows if r["status"] == "ok")
    logging.info(
        f"Total: {n_ok}/{len(manifest_rows)} instances extracted successfully."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
