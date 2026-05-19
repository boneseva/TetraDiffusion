"""
augment_objs.py — Generate augmented copies of OBJ meshes for TetraDiffusion preprocessing.

For organelles (and most organic shapes), the object has no fixed orientation in 3D space,
so axis-aligned reflections and 90° rotations are all *genuinely different* shapes that are
valid training samples.

Augmentation catalogue
----------------------
  reflections  : flip along X, Y, Z axes independently → up to 8 variants (2^3 including identity)
  rotations    : 90° rotations around each axis          → 24 unique rigid-body orientations
  combined     : all 48 elements of the octahedral group (Oh)

Default (--mode reflections) gives a simple 8× multiplier with no quality loss.

Output layout
-------------
The output mirrors the input layout so the result folder can be passed directly to fit_many.py:

  <output_root>/<class_id>/<aug_model_id>/<same_relative_path>.obj

where aug_model_id = "<original_model_id>__aug_<tag>".

Usage
-----
  python augment_objs.py \\
      --input_root  data/organelles_obj \\
      --output_root data/organelles_obj_aug \\
      --mode reflections        # or: rotations | combined | custom
      --copy_original           # also copy the un-augmented shape into output_root
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Transform catalogue
# ---------------------------------------------------------------------------

def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    """Return a 3×3 rotation matrix for a rotation around 'x', 'y', or 'z'."""
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    if axis == 'x':
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]], dtype=float)
    elif axis == 'y':
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=float)
    else:  # z
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], dtype=float)


def _build_reflection_transforms() -> List[Tuple[str, np.ndarray]]:
    """8 transforms: all combinations of sign flips on x/y/z (identity included)."""
    result = []
    for fx, fy, fz in itertools.product([-1, 1], repeat=3):
        if fx == fy == fz == 1:
            continue  # identity — caller decides whether to copy original
        tag = ("" if fx == 1 else "fx") + ("" if fy == 1 else "fy") + ("" if fz == 1 else "fz")
        M = np.diag([fx, fy, fz]).astype(float)
        result.append((tag, M))
    return result


def _build_rotation_transforms() -> List[Tuple[str, np.ndarray]]:
    """24 unique orientations of a cube / rigid-body rotation group."""
    angles = [0, 90, 180, 270]
    seen: dict[bytes, str] = {}
    result = []

    for ax, ay, az in itertools.product(angles, repeat=3):
        M = (_rotation_matrix('x', ax)
             @ _rotation_matrix('y', ay)
             @ _rotation_matrix('z', az))
        key = np.round(M, 4).tobytes()
        if key in seen:
            continue
        seen[key] = f"rx{ax}ry{ay}rz{az}"
        if ax == ay == az == 0:
            continue  # identity
        tag = f"rx{ax}ry{ay}rz{az}"
        result.append((tag, M))

    return result


def _build_combined_transforms() -> List[Tuple[str, np.ndarray]]:
    """48 elements of the full octahedral group (Oh)."""
    seen: dict[bytes, str] = {}
    result = []
    for _, R in _build_rotation_transforms() + [("id", np.eye(3))]:
        for _, F in _build_reflection_transforms() + [("id", np.eye(3))]:
            M = F @ R
            key = np.round(M, 4).tobytes()
            if key in seen:
                continue
            seen[key] = "ok"
            if np.allclose(M, np.eye(3)):
                continue  # identity
            tag = "oh_" + np.array2string(M.flatten().round(0).astype(int),
                                           separator='').replace(' ', '').strip('[]').replace('-1', 'n').replace('1', 'p').replace('0', '_')
            result.append((tag, M))
    return result


TRANSFORM_CATALOGUES = {
    "reflections": _build_reflection_transforms,
    "rotations":   _build_rotation_transforms,
    "combined":    _build_combined_transforms,
}


# ---------------------------------------------------------------------------
# Minimal OBJ reader / writer (no extra deps)
# ---------------------------------------------------------------------------

def _load_obj_vertices_and_lines(path: Path):
    """Read an OBJ file; return (vertices Nx3, all_lines, vertex_line_indices)."""
    vertices = []
    all_lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    vertex_line_indices = []

    for i, line in enumerate(all_lines):
        stripped = line.strip()
        if stripped.startswith('v '):
            parts = stripped.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                vertex_line_indices.append(i)

    return np.array(vertices, dtype=float), all_lines, vertex_line_indices


def _save_obj_with_new_vertices(path: Path, all_lines: list, vertex_line_indices: list,
                                 new_verts: np.ndarray) -> None:
    """Write a new OBJ file, replacing only the vertex positions."""
    lines = list(all_lines)
    for raw_idx, line_idx in enumerate(vertex_line_indices):
        parts = lines[line_idx].strip().split()
        # Preserve any w component or vertex colour suffix
        suffix = ' '.join(parts[4:]) if len(parts) > 4 else ''
        x, y, z = new_verts[raw_idx]
        new_line = f"v {x:.8f} {y:.8f} {z:.8f}"
        if suffix:
            new_line += ' ' + suffix
        lines[line_idx] = new_line

    # Repair face winding if the transform has negative determinant (mirroring)
    # A negative-determinant transform reverses face orientation; flip f lines.
    yield_lines = lines
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(yield_lines) + '\n', encoding='utf-8')


def _fix_winding_if_needed(all_lines: list, matrix: np.ndarray) -> list:
    """If matrix has det < 0 (mirror), reverse face vertex order so normals stay outward."""
    if np.linalg.det(matrix) >= 0:
        return all_lines

    fixed = []
    for line in all_lines:
        stripped = line.strip()
        if stripped.startswith('f '):
            tokens = stripped.split()
            # Reverse the face vertex tokens (keep 'f' prefix)
            fixed.append('f ' + ' '.join(tokens[1:][::-1]))
        else:
            fixed.append(line)
    return fixed


# ---------------------------------------------------------------------------
# Core augmentation logic
# ---------------------------------------------------------------------------

def discover_obj_jobs(input_root: Path, obj_glob: str) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    for obj_path in sorted(input_root.glob(obj_glob)):
        if not obj_path.is_file() or obj_path.suffix.lower() != '.obj':
            continue
        rel = obj_path.relative_to(input_root)
        if len(rel.parts) < 2:
            continue
        class_id  = rel.parts[0]
        model_id  = rel.parts[1]
        jobs.append((class_id, model_id, obj_path))
    return jobs


def augment_one(obj_path: Path, input_root: Path, output_root: Path,
                transforms: List[Tuple[str, np.ndarray]],
                copy_original: bool, dry_run: bool, overwrite: bool) -> int:
    rel = obj_path.relative_to(input_root)
    class_id  = rel.parts[0]
    model_id  = rel.parts[1]
    rel_inner = Path(*rel.parts[2:]) if len(rel.parts) > 2 else Path(obj_path.name)

    written = 0

    if copy_original:
        dst = output_root / class_id / model_id / rel_inner
        if overwrite or not dst.exists():
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(obj_path, dst)
            written += 1

    verts, lines, vert_indices = _load_obj_vertices_and_lines(obj_path)
    if len(verts) == 0:
        print(f"  WARNING: no vertices found in {obj_path}")
        return written

    # Centre the mesh before applying transforms, then restore
    centre = verts.mean(axis=0)
    verts_centred = verts - centre

    for tag, M in transforms:
        aug_model_id = f"{model_id}__aug_{tag}"
        dst = output_root / class_id / aug_model_id / rel_inner

        if dst.exists() and not overwrite:
            written += 1
            continue

        new_verts = (M @ verts_centred.T).T + centre
        fixed_lines = _fix_winding_if_needed(lines, M)

        if dry_run:
            print(f"  [DRY] {class_id}/{aug_model_id}/{rel_inner}")
        else:
            _save_obj_with_new_vertices(dst, fixed_lines, vert_indices, new_verts)

        written += 1

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate augmented OBJ copies for TetraDiffusion")
    p.add_argument("--input_root",     type=Path, required=True,
                   help="Root folder with class/model OBJ files")
    p.add_argument("--output_root",    type=Path, required=True,
                   help="Destination root (mirrors input layout with aug_ model IDs)")
    p.add_argument("--obj_glob",       type=str,  default="*/*/**/*.obj",
                   help="Glob relative to input_root (default: */*/**/*.obj)")
    p.add_argument("--mode",           type=str,  default="reflections",
                   choices=list(TRANSFORM_CATALOGUES.keys()) + ["custom"],
                   help="Which transforms to apply (default: reflections → 7 variants per shape)")
    p.add_argument("--copy_original",  action="store_true",
                   help="Also copy the un-augmented OBJ into output_root")
    p.add_argument("--overwrite",      action="store_true",
                   help="Re-generate even if output already exists")
    p.add_argument("--dry_run",        action="store_true",
                   help="Print planned jobs without writing files")
    p.add_argument("--category",       type=str,  default=None,
                   help="Only process this category (class subfolder name)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_root  = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        print(f"ERROR: input_root not found: {input_root}")
        return 1

    transforms = TRANSFORM_CATALOGUES[args.mode]()
    print(f"Mode: {args.mode}  →  {len(transforms)} transform(s) per shape")
    if args.copy_original:
        print("  + copying original (un-augmented) shapes")

    jobs = discover_obj_jobs(input_root, args.obj_glob)
    if args.category:
        jobs = [(c, m, p) for c, m, p in jobs if c == args.category]
        print(f"Category filter: '{args.category}'  →  {len(jobs)} OBJ(s)")
    else:
        print(f"Discovered {len(jobs)} OBJ file(s) in {input_root}")

    if not jobs:
        print("No OBJ files found. Check --input_root and --obj_glob.")
        return 1

    if args.dry_run:
        print("\n[DRY RUN — first 10 jobs]")
        for c, m, p in jobs[:10]:
            print(f"  {c}/{m}  ({p.name})")
            for tag, _ in transforms[:3]:
                print(f"    → {c}/{m}__aug_{tag}/")
            if len(transforms) > 3:
                print(f"    … and {len(transforms) - 3} more variants")
        return 0

    total_written = 0
    for idx, (class_id, model_id, obj_path) in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {class_id}/{model_id}")
        n = augment_one(obj_path, input_root, output_root, transforms,
                        copy_original=args.copy_original,
                        dry_run=args.dry_run,
                        overwrite=args.overwrite)
        total_written += n

    print(f"\nDone — {total_written} file(s) written to {output_root}")
    print("\nNext step: run fit_many.py on the augmented output:")
    print(f"  python preprocessing/fit_many.py \\")
    print(f"      --input_root  {output_root} \\")
    print(f"      --output_root <your_data_path>/<class> \\")
    print(f"      --dmtet_grid 128 --iter 3000 \\")
    print(f"      --update_all_csv lib/all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

