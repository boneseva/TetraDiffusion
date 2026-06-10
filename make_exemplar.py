"""
make_exemplar.py — Convert a surface mesh or binary voxel segmentation into a
TetraDiffusion exemplar `.pt` file that can be passed to training via
``--exemplar_path``.

Output format
-------------
The saved file is a 4-element list mirroring the ``sample_*.pt`` layout used
throughout MeshLoader:

    [sdfs, displacements, colors, source_path]

where
  * ``sdfs``          — float32 Tensor [N_full], one SDF value per tetrahedral
                        vertex (negative = interior, positive = exterior).
  * ``displacements`` — float32 Tensor [N_full, 4], all zeros (weight=0);
                        the training normalisation statistics are derived
                        from real training data so zeros are safe here.
  * ``colors``        — float32 Tensor [N_full, 3], all zeros (same reason).
  * ``source_path``   — str, the original ``--input`` path for provenance.

Supported input formats
-----------------------
Mesh  : .obj  .stl  .ply  (and anything else trimesh can read)
Volume: .npy  — expected to contain a 3-D boolean / uint8 occupancy array
                (True / 1 = interior voxel).

Signed-distance convention
--------------------------
* Mesh path  : trimesh proximity query, then negated so that the *interior*
               of the mesh (where ``contains()`` is True) has SDF < 0.
* Volume path: scipy EDT on the interior, minus EDT on the exterior, giving
               a proper signed distance on the regular voxel grid.  Values
               are then interpolated (nearest-neighbour) onto the
               tetrahedral vertex positions that lie inside the volume
               bounding box.

Usage example
-------------
    python make_exemplar.py \\
        --input  /data/mito_gt.obj \\
        --output /data/mito_exemplar.pt \\
        --grid_res 128

    python make_exemplar.py \\
        --input  /data/lyso_mask.npy \\
        --output /data/lyso_exemplar.pt \\
        --grid_res 128
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a surface mesh or binary voxel segmentation into a "
            "TetraDiffusion exemplar .pt file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help=(
            "Path to a 3-D surface mesh (.obj, .stl, .ply, …) "
            "or a binary 3-D voxel occupancy array (.npy)."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the output .pt exemplar file.",
    )
    p.add_argument(
        "--grid_res",
        type=int,
        default=128,
        metavar="N",
        help=(
            "Tetrahedral grid resolution.  Must match the resolution used "
            "during training.  Determines which "
            "tetrahedra/{grid_res}/0_tets.npz file is loaded."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Grid loading — mirrors MeshLoader.__init__ exactly
# ---------------------------------------------------------------------------

def _load_tet_vertices(grid_res: int) -> np.ndarray:
    """Load and zero-centre the coarsest-resolution tetrahedral vertices.

    Replicates the MeshLoader initialisation:
        vertices[i] = vertices[i] - mean(vertices[i], axis=0)

    Returns
    -------
    np.ndarray  [N, 3]  float32  — zero-centred vertex positions.
    """
    tet_path = os.path.join("tetrahedra", str(grid_res), "0_tets.npz")
    if not os.path.isfile(tet_path):
        sys.exit(
            f"[make_exemplar] ERROR: tetrahedral grid file not found: {tet_path}\n"
            f"Make sure you are running from the TetraDiffusion project root "
            f"and that the tetrahedra/{grid_res}/ directory is present."
        )
    data = np.load(tet_path)
    verts = data["vertices"].astype(np.float32)  # [N, 3]
    verts -= verts.mean(axis=0)                   # zero-centre (MeshLoader convention)
    return verts


# ---------------------------------------------------------------------------
# SDF from mesh
# ---------------------------------------------------------------------------

def _sdf_from_mesh(mesh_path: str, verts: np.ndarray) -> np.ndarray:
    """Compute signed distances from a surface mesh at tetrahedral vertices.

    Uses trimesh proximity queries.  Sign convention: interior → SDF < 0.

    Parameters
    ----------
    mesh_path : str
    verts     : np.ndarray [N, 3]

    Returns
    -------
    np.ndarray [N]  float32
    """
    try:
        import trimesh
    except ImportError:
        sys.exit(
            "[make_exemplar] ERROR: 'trimesh' is not installed.\n"
            "Run:  pip install trimesh rtree"
        )

    print(f"[make_exemplar] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=True)

    if not isinstance(mesh, trimesh.Trimesh):
        # Some loaders return a Scene; extract the geometry
        geom = list(mesh.geometry.values())
        if not geom:
            sys.exit("[make_exemplar] ERROR: mesh file contained no geometry.")
        mesh = trimesh.util.concatenate(geom)

    print(f"[make_exemplar] Mesh loaded: {len(mesh.vertices)} verts, "
          f"{len(mesh.faces)} faces")

    # Scale mesh to fit inside the tetrahedral bounding box so that the SDF
    # is meaningful in the same coordinate space.
    mesh_bb   = mesh.bounds                # [2, 3]
    mesh_span = mesh_bb[1] - mesh_bb[0]
    tet_bb    = np.stack([verts.min(0), verts.max(0)])
    tet_span  = tet_bb[1] - tet_bb[0]

    scale = (tet_span / mesh_span.clip(min=1e-8)).min()

    mesh_centre  = mesh_bb.mean(0)
    tet_centre   = tet_bb.mean(0)
    mesh.apply_translation(-mesh_centre)
    mesh.apply_scale(scale)
    mesh.apply_translation(tet_centre)

    print(f"[make_exemplar] Running proximity query on {len(verts):,} vertices …")

    # unsigned distances + containment
    _, unsigned_dists, _ = trimesh.proximity.closest_point(mesh, verts)
    inside = mesh.contains(verts)          # [N] bool

    # SDF: negative inside, positive outside
    sdf = np.where(inside, -unsigned_dists, unsigned_dists).astype(np.float32)

    print(f"[make_exemplar] SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]  "
          f"interior vertices: {inside.sum():,}")
    return sdf


# ---------------------------------------------------------------------------
# SDF from voxel volume
# ---------------------------------------------------------------------------

def _sdf_from_volume(npy_path: str, verts: np.ndarray) -> np.ndarray:
    """Build a signed distance field from a binary voxel occupancy array.

    1. Run EDT on the interior mask  → distance-to-surface from inside.
    2. Run EDT on the exterior mask  → distance-to-surface from outside.
    3. SDF = exterior_edt − interior_edt  (negative inside, positive outside).
    4. Interpolate (nearest-neighbour) onto tetrahedral vertex positions.

    Parameters
    ----------
    npy_path : str
    verts    : np.ndarray [N, 3]

    Returns
    -------
    np.ndarray [N]  float32
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        sys.exit(
            "[make_exemplar] ERROR: 'scipy' is not installed.\n"
            "Run:  pip install scipy"
        )

    print(f"[make_exemplar] Loading voxel volume: {npy_path}")
    vol = np.load(npy_path)

    if vol.ndim != 3:
        sys.exit(
            f"[make_exemplar] ERROR: expected a 3-D numpy array, got shape {vol.shape}."
        )

    interior = vol.astype(bool)
    exterior = ~interior

    print(f"[make_exemplar] Volume shape: {vol.shape}, "
          f"interior voxels: {interior.sum():,}")

    # EDT distances (in voxel units)
    print("[make_exemplar] Running distance transform (interior) …")
    edt_in  = distance_transform_edt(interior).astype(np.float32)  # 0 outside
    print("[make_exemplar] Running distance transform (exterior) …")
    edt_out = distance_transform_edt(exterior).astype(np.float32)  # 0 inside

    # Signed distance field on the voxel grid (negative inside)
    sdf_vol = edt_out - edt_in   # [D, H, W]  float32

    # ── Map tet vertex positions to voxel indices ──────────────────────────
    D, H, W = vol.shape

    # Normalise tet coords to [0, D/H/W - 1] voxel index space using the
    # tet bounding box, then clamp to valid voxel indices.
    tet_min = verts.min(0)   # [3]
    tet_max = verts.max(0)   # [3]
    tet_span = (tet_max - tet_min).clip(min=1e-8)

    # verts[:, 0] → depth axis (D), [1] → height (H), [2] → width (W)
    idx_d = ((verts[:, 0] - tet_min[0]) / tet_span[0] * (D - 1)).round().astype(int)
    idx_h = ((verts[:, 1] - tet_min[1]) / tet_span[1] * (H - 1)).round().astype(int)
    idx_w = ((verts[:, 2] - tet_min[2]) / tet_span[2] * (W - 1)).round().astype(int)

    # Clamp to grid bounds
    idx_d = idx_d.clip(0, D - 1)
    idx_h = idx_h.clip(0, H - 1)
    idx_w = idx_w.clip(0, W - 1)

    sdf = sdf_vol[idx_d, idx_h, idx_w]  # [N]  float32

    print(f"[make_exemplar] SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]  "
          f"interior vertices: {(sdf < 0).sum():,}")
    return sdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # ── Validate input ──────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        sys.exit(f"[make_exemplar] ERROR: input file not found: {args.input}")

    ext = os.path.splitext(args.input)[1].lower()
    volume_exts = {".npy"}
    mesh_exts   = {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}

    if ext not in volume_exts and ext not in mesh_exts:
        print(
            f"[make_exemplar] WARNING: unrecognised extension '{ext}'. "
            f"Attempting to load as a mesh via trimesh."
        )

    # ── Load tetrahedral grid ───────────────────────────────────────────────
    print(f"[make_exemplar] Loading tetrahedral grid (res={args.grid_res}) …")
    verts = _load_tet_vertices(args.grid_res)   # [N, 3]  float32
    N = len(verts)
    print(f"[make_exemplar] Grid loaded: {N:,} vertices")

    # ── Compute SDF ─────────────────────────────────────────────────────────
    if ext in volume_exts:
        sdf_np = _sdf_from_volume(args.input, verts)
    else:
        sdf_np = _sdf_from_mesh(args.input, verts)

    # ── Pack into TetraDiffusion sample format ───────────────────────────────
    # Layout: [sdfs [N], displacements [N, 4], colors [N, 3], source_path str]
    # Displacements channel 4 is the surface-weight mask; zeros are safe.
    sdfs          = torch.from_numpy(sdf_np)                     # [N]   float32
    displacements = torch.zeros(N, 4, dtype=torch.float32)       # [N,4] zeros
    colors        = torch.zeros(N, 3, dtype=torch.float32)       # [N,3] zeros

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save([sdfs, displacements, colors, args.input], args.output)
    print(f"[make_exemplar] Saved exemplar to: {args.output}")
    print(
        f"[make_exemplar] Done.  Pass this file to training with:\n"
        f"    --exemplar_path {args.output}"
    )


if __name__ == "__main__":
    main()
