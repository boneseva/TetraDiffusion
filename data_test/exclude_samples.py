"""
Scan all .obj files under `organelles/` and copy those that satisfy mesh
quality requirements to a mirrored `organelles_raw/` folder.

Current requirements
--------------------
- Watertight  (mesh.is_watertight via trimesh)

Normal repair
-------------
Before writing each passing mesh to the destination, face windings are
corrected so that vertex normals point **outward** (away from the mesh
interior).  This is done via `trimesh.repair.fix_normals`, which
works reliably on watertight meshes and is guaranteed to run because
watertightness is already a hard requirement above.

Usage
-----
# Dry-run (only prints what would be moved, no files are copied)
python exclude_samples.py --dry-run

# Copy passing files to organelles_raw/ (normals fixed on the fly)
python exclude_samples.py

# Move instead of copy
python exclude_samples.py --move

# Override the input / output root dirs
python exclude_samples.py --src organelles --dst organelles_raw
"""

import argparse
import sys
from pathlib import Path

try:
    import trimesh
except ImportError:
    sys.exit("trimesh is required: pip install trimesh")


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def is_watertight(mesh_path: Path) -> bool:
    """Return True if the mesh at *mesh_path* is watertight."""
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    except Exception as exc:
        print(f"  [LOAD ERROR] {mesh_path.name}: {exc}")
        return False

    if not isinstance(mesh, trimesh.Trimesh):
        # Could be a Scene with multiple geometries – merge and check
        try:
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        except Exception:
            return False

    return bool(mesh.is_watertight)


def passes_requirements(mesh_path: Path) -> list:
    """Return a list of failed check names (empty = all passed).
    Extend this function for future criteria."""
    checks = {
        "watertight": is_watertight,
        # Add more checks here, e.g.:
        # "positive_volume": has_positive_volume,
    }
    failed = [name for name, fn in checks.items() if not fn(mesh_path)]
    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Filter .obj files by mesh quality.")
    parser.add_argument(
        "--src",
        type=Path,
        default=here / "organelles",
        help="Root folder containing organelle sub-directories (default: ./organelles)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=here / "organelles_raw",
        help="Destination root folder (default: ./organelles_raw)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without touching any files.",
    )
    args = parser.parse_args()

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()

    if not src_root.exists():
        sys.exit(f"Source directory does not exist: {src_root}")

    obj_files = sorted(src_root.rglob("*.obj"))
    if not obj_files:
        sys.exit(f"No .obj files found under {src_root}")

    print(f"Source : {src_root}")
    print(f"Dest   : {dst_root}")
    print(f"Mode   : {'DRY RUN' if args.dry_run else ('move' if args.move else 'copy')}")
    print(f"Found  : {len(obj_files)} .obj files\n")

    n_passed = n_failed = n_error = 0

    for obj_path in obj_files:
        rel = obj_path.relative_to(src_root)
        failed_checks = passes_requirements(obj_path)

        if failed_checks:
            print(f"  SKIP  [{', '.join(failed_checks)}]  {rel}")
            n_failed += 1
        else:
            dst_path = dst_root / rel

            if args.dry_run:
                print(f"  [DRY-RUN] PASS  {rel}  ->  {dst_path}")
                n_passed += 1
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Load, fix normals so they face outward, then export.
                    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
                    if not isinstance(mesh, trimesh.Trimesh):
                        mesh = trimesh.util.concatenate(
                            [g for g in mesh.geometry.values()
                             if isinstance(g, trimesh.Trimesh)]
                        )
                    trimesh.repair.fix_normals(mesh)
                    mesh.export(str(dst_path))
                    if args.move:
                        obj_path.unlink()
                    print(f"  PASS (normals fixed)  {rel}")
                    n_passed += 1
                except Exception as exc:
                    print(f"  [IO ERROR] {rel}: {exc}")
                    n_error += 1

    print(
        f"\nDone. Passed: {n_passed}  |  Failed: {n_failed}  |  Errors: {n_error}"
        f"  |  Total: {len(obj_files)}"
    )


if __name__ == "__main__":
    main()

