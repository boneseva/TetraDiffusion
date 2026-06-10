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
    """Load and zero-centre the finest-resolution tetrahedral vertices.

    Files inside ``tetrahedra/{grid_res}/`` are named ``{res}_tets.npz``
    (e.g. ``128_tets.npz``), matching the ``MeshLoader`` convention where
    ``cube_range`` lists the actual resolution integers and the last entry
    is the finest grid.

    Replicates the MeshLoader initialisation:
        vertices[i] = vertices[i] - mean(vertices[i], axis=0)

    Returns
    -------
    np.ndarray  [N, 3]  float32  — zero-centred vertex positions.
    """
    tet_path = os.path.join("tetrahedra", str(grid_res), f"{grid_res}_tets.npz")
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
# Shared EDT core — used by both mesh and volume paths
# ---------------------------------------------------------------------------

def _sdf_from_edt(interior: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Convert a binary occupancy volume to SDF values at tet vertex positions.

    Runs two distance transforms (interior + exterior), combines them into a
    signed field, then samples at the tetrahedral vertex coordinates via
    nearest-neighbour lookup.

    Parameters
    ----------
    interior : np.ndarray [D, H, W]  bool  — True = inside the organelle.
    verts    : np.ndarray [N, 3]    float32 — zero-centred tet vertex positions.

    Returns
    -------
    np.ndarray [N]  float32  — SDF (negative inside, positive outside).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        sys.exit(
            "[make_exemplar] ERROR: 'scipy' is not installed.\n"
            "Run:  pip install scipy"
        )

    exterior = ~interior
    D, H, W  = interior.shape

    print("[make_exemplar] Running distance transform (interior) …")
    edt_in  = distance_transform_edt(interior).astype(np.float32)
    print("[make_exemplar] Running distance transform (exterior) …")
    edt_out = distance_transform_edt(exterior).astype(np.float32)

    sdf_vol = edt_out - edt_in   # negative inside, positive outside

    # ── Nearest-neighbour sampling at tet vertex positions ─────────────────
    tet_min  = verts.min(0)
    tet_span = (verts.max(0) - tet_min).clip(min=1e-8)

    idx_d = ((verts[:, 0] - tet_min[0]) / tet_span[0] * (D - 1)).round().astype(int).clip(0, D - 1)
    idx_h = ((verts[:, 1] - tet_min[1]) / tet_span[1] * (H - 1)).round().astype(int).clip(0, H - 1)
    idx_w = ((verts[:, 2] - tet_min[2]) / tet_span[2] * (W - 1)).round().astype(int).clip(0, W - 1)

    sdf = sdf_vol[idx_d, idx_h, idx_w]
    print(f"[make_exemplar] SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]  "
          f"interior vertices: {(sdf < 0).sum():,}")
    return sdf


# ---------------------------------------------------------------------------
# SDF from mesh  (voxelise -> EDT -- avoids per-vertex ray casting entirely)
# ---------------------------------------------------------------------------

def _sdf_from_mesh(mesh_path: str, verts: np.ndarray,
                   voxel_resolution: int = 256) -> np.ndarray:
    """Compute signed distances from a surface mesh at tetrahedral vertices.

    Strategy
    --------
    Instead of querying every tet vertex directly against the mesh (which
    requires one ray-cast per point and is O(N * F) at 277K+ vertices), the
    mesh is first **voxelised** at ``voxel_resolution`` along the longest
    axis, then a pair of Euclidean Distance Transforms are run on the
    resulting binary volume.  The total runtime is O(D*H*W) regardless of
    tet vertex count -- typically 3-15 seconds on a CPU.

    Sign convention: interior -> SDF < 0.

    Parameters
    ----------
    mesh_path        : str
    verts            : np.ndarray [N, 3]
    voxel_resolution : int   number of voxels along the longest axis
                             (default 256; increase for finer meshes).

    Returns
    -------
    np.ndarray [N]  float32
    """
    try:
        import trimesh
    except ImportError:
        sys.exit(
            "[make_exemplar] ERROR: 'trimesh' is not installed.\n"
            "Run:  pip install trimesh"
        )

    print(f"[make_exemplar] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=True)

    if not isinstance(mesh, trimesh.Trimesh):
        geom = list(mesh.geometry.values())
        if not geom:
            sys.exit("[make_exemplar] ERROR: mesh file contained no geometry.")
        mesh = trimesh.util.concatenate(geom)

    print(f"[make_exemplar] Mesh loaded: {len(mesh.vertices):,} verts, "
          f"{len(mesh.faces):,} faces")

    # ── Scale + centre mesh into the tet bounding box ──────────────────────
    mesh_bb    = mesh.bounds
    mesh_span  = (mesh_bb[1] - mesh_bb[0]).clip(min=1e-8)
    tet_bb     = np.stack([verts.min(0), verts.max(0)])
    tet_span   = tet_bb[1] - tet_bb[0]

    scale = (tet_span / mesh_span).min()
    mesh.apply_translation(-mesh_bb.mean(0))
    mesh.apply_scale(scale)
    mesh.apply_translation(tet_bb.mean(0))

    # ── Voxelise ───────────────────────────────────────────────────────────
    pitch = mesh.extents.max() / voxel_resolution
    print(f"[make_exemplar] Voxelising at pitch={pitch:.5f} "
          f"(target {voxel_resolution} voxels on longest axis) …")

    voxgrid  = trimesh.voxel.creation.voxelize(mesh, pitch=pitch)
    interior = voxgrid.matrix.astype(bool)   # [D, H, W]  True = inside

    print(f"[make_exemplar] Voxel grid shape: {interior.shape}, "
          f"interior voxels: {interior.sum():,}")

    return _sdf_from_edt(interior, verts)


# ---------------------------------------------------------------------------
# SDF from voxel volume
# ---------------------------------------------------------------------------

def _sdf_from_volume(npy_path: str, verts: np.ndarray) -> np.ndarray:
    """Build a signed distance field from a binary voxel occupancy array.

    Parameters
    ----------
    npy_path : str
    verts    : np.ndarray [N, 3]

    Returns
    -------
    np.ndarray [N]  float32
    """
    print(f"[make_exemplar] Loading voxel volume: {npy_path}")
    vol = np.load(npy_path)

    if vol.ndim != 3:
        sys.exit(
            f"[make_exemplar] ERROR: expected a 3-D numpy array, got shape {vol.shape}."
        )

    interior = vol.astype(bool)
    print(f"[make_exemplar] Volume shape: {vol.shape}, "
          f"interior voxels: {interior.sum():,}")

    return _sdf_from_edt(interior, verts)



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
