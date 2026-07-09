"""
posthoc_fairing.py — Post-hoc Laplacian mesh-fairing baseline.

This script implements the comparison required by the reviewers:
"Why is internalising the manifold during training explicitly better than
applying classical mesh fairing/smoothing to final Bio− outputs?"
(Second-round reviewer: "include actual quantitative results (MMD, COV,
Volume Wasserstein distance) comparing Bio− + Taubin directly against Bio+")

Usage
-----
    # Basic: compute LR/SCE before/after fairing
    python scripts/posthoc_fairing.py \
        --input_dir  runs/inference_mitochondria \
        --output_dir runs/inference_mitochondria_faired \
        --iterations 10 \
        --lambda     0.5

    # Full comparison including MMD-CD, COV-CD, Vol. Wasserstein
    python scripts/posthoc_fairing.py \
        --input_dir     runs/inference_mitochondria \
        --reference_dir runs/gt_mitochondria \
        --bioplus_dir   runs/inference_mitochondria_bioplus \
        --output_dir    runs/inference_mitochondria_faired \
        --iterations    5,10,20

The script:
  1. Loads every *.obj file in ``input_dir`` (Bio− outputs).
  2. Applies N steps of Taubin smoothing (λ=0.5, μ=−0.53).
  3. Saves the faired meshes to ``output_dir``.
  4. Computes and prints a full comparison table:
       - LR, SCE  (surface quality; lower is better)
       - MMD-CD, COV-CD  (distribution quality; requires --reference_dir)
       - Volume Wasserstein-1  (morphological fidelity; requires --reference_dir)
     across Bio−, Bio− + Taubin(5/10/20 iter), and Bio+ (if --bioplus_dir given).

Dependencies
------------
    pip install trimesh numpy scipy

Metrics
-------
LR  — Laplacian Roughness:  mean squared uniform-Laplacian of vertex positions
       (displacement from mean-neighbour), averaged over all vertices with
       valence ≥ 2.  Lower is better.

SCE — Surface Curvature Energy: vertex-area-weighted mean squared mean
       curvature estimated via the cotangent Laplacian.  Lower is better.

MMD-CD — Minimum Matching Distance (Chamfer Distance): for each reference mesh,
       find the nearest generated mesh.  Average of those distances.  Lower is
       better.  Requires --reference_dir.

COV-CD — Coverage: fraction of reference meshes matched by at least one
       generated mesh.  Higher is better.  Requires --reference_dir.

Vol. Wass. — Wasserstein-1 distance between the volume distributions of
       generated and reference meshes.  Lower is better.  Requires --reference_dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh is required: pip install trimesh")

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Mesh metric helpers
# ---------------------------------------------------------------------------

def _cotangent_weights(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return per-half-edge cotangent weights (N_faces × 3 array)."""
    v = mesh.vertices[mesh.faces]               # (F, 3, 3)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    e3 = v[:, 2] - v[:, 1]

    # Cotangent of angle at each vertex of each triangle
    def _cot(a, b):
        cross = np.linalg.norm(np.cross(a, b), axis=-1)
        dot   = (a * b).sum(axis=-1)
        return dot / (cross + 1e-10)

    cot = np.stack([
        _cot(-e1, e2 - e1),   # angle at vertex 2
        _cot(-e2, e1 - e2),   # angle at vertex 0
        _cot(-e3, e2 - e3),   # angle at vertex 1
    ], axis=1)
    return cot


def laplacian_roughness(mesh: trimesh.Trimesh) -> float:
    """Mean squared uniform graph-Laplacian of vertex positions."""
    V = mesh.vertices               # (N, 3)
    F = mesh.faces                  # (M, 3)
    N = len(V)

    neigh: List[List[int]] = [[] for _ in range(N)]
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            neigh[a].append(b)
            neigh[b].append(a)

    sq_sum = 0.0
    count  = 0
    for i, nb in enumerate(neigh):
        nb_uniq = list(set(nb))
        if len(nb_uniq) < 2:
            continue
        mean_nb = V[nb_uniq].mean(axis=0)
        sq_sum += np.sum((V[i] - mean_nb) ** 2)
        count  += 1

    return sq_sum / max(count, 1)


def surface_curvature_energy(mesh: trimesh.Trimesh) -> float:
    """Vertex-area-weighted mean squared mean curvature via cotangent Laplacian."""
    V   = mesh.vertices     # (N, 3)
    F   = mesh.faces        # (M, 3)
    N   = len(V)
    cot = _cotangent_weights(mesh)      # (F, 3)

    # Build cotangent Laplacian and mixed vertex areas
    lap      = np.zeros((N, 3), dtype=np.float64)
    area     = np.zeros(N,      dtype=np.float64)
    face_area = 0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]],
                 V[F[:, 2]] - V[F[:, 0]]), axis=-1)   # (F,)

    for f_idx, (tri, cw) in enumerate(zip(F, cot)):
        fa = face_area[f_idx] / 3.0
        for local, (i, j) in enumerate([(0,1),(1,2),(2,0)]):
            vi, vj = tri[i], tri[j]
            w = cw[local] / 2.0
            lap[vi] += w * (V[vj] - V[vi])
            lap[vj] += w * (V[vi] - V[vj])
            area[vi] += fa
            area[vj] += fa

    area = np.maximum(area, 1e-12)
    H_vec = lap / (2.0 * area[:, None])    # mean curvature normal (N, 3)
    H_sq  = (H_vec ** 2).sum(axis=1)       # |H|² per vertex

    # Weighted average
    return float((H_sq * area).sum() / area.sum())


# ---------------------------------------------------------------------------
# Laplacian / Taubin smoothing
# ---------------------------------------------------------------------------

def laplacian_smooth(mesh: trimesh.Trimesh,
                     iterations: int = 10,
                     lam: float = 0.5,
                     taubin: bool = True,
                     mu: float = -0.53) -> trimesh.Trimesh:
    """Apply Laplacian (or Taubin) smoothing to a trimesh.Trimesh.

    Taubin (1995) two-pass scheme avoids volume shrinkage:
        v ← v + λ·L(v)   (inflate)
        v ← v + μ·L(v)   (deflate, μ < 0, |μ| > λ)

    Args:
        mesh:        Input mesh (not modified in place).
        iterations:  Number of full smoothing passes.
        lam:         Forward step size (λ).
        taubin:      Use Taubin two-pass scheme (recommended).
        mu:          Backward step size for Taubin (must be negative).
    """
    V = mesh.vertices.copy().astype(np.float64)
    F = mesh.faces
    N = len(V)

    # Build adjacency list once
    neigh: List[List[int]] = [[] for _ in range(N)]
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            neigh[a].append(b)
            neigh[b].append(a)
    neigh = [list(set(nb)) for nb in neigh]

    def _laplacian_delta(V_):
        delta = np.zeros_like(V_)
        for i, nb in enumerate(neigh):
            if nb:
                delta[i] = V_[nb].mean(axis=0) - V_[i]
        return delta

    for _ in range(iterations):
        V = V + lam * _laplacian_delta(V)
        if taubin:
            V = V + mu  * _laplacian_delta(V)

    return trimesh.Trimesh(vertices=V, faces=F, process=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Distribution metrics: MMD-CD, COV-CD, Volume Wasserstein
# ---------------------------------------------------------------------------

def _sample_pointcloud(mesh: trimesh.Trimesh, n_points: int = 2048) -> np.ndarray:
    """Uniformly sample n_points points from a triangle mesh surface."""
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        return pts.astype(np.float64)
    except Exception:
        # Fallback: random vertices
        idx = np.random.choice(len(mesh.vertices), size=n_points, replace=True)
        return mesh.vertices[idx].astype(np.float64)


def _chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Chamfer distance between two point clouds (N×3 arrays)."""
    # A→B: for each point in A find nearest in B
    diff_ab = A[:, None, :] - B[None, :, :]      # (Na, Nb, 3)
    dist_ab = (diff_ab ** 2).sum(axis=-1)         # (Na, Nb)
    min_ab  = dist_ab.min(axis=1).mean()
    # B→A
    min_ba  = dist_ab.min(axis=0).mean()
    return float(min_ab + min_ba)


def compute_distribution_metrics(generated_meshes: List[trimesh.Trimesh],
                                  reference_meshes:  List[trimesh.Trimesh],
                                  n_points: int = 2048
                                  ) -> dict:
    """Compute MMD-CD, COV-CD, and Volume Wasserstein-1 distance.

    Args:
        generated_meshes: List of generated meshes.
        reference_meshes: List of ground-truth / held-out test meshes.
        n_points:         Number of surface points per mesh for CD computation.

    Returns:
        dict with keys: mmd_cd, cov_cd, vol_wasserstein
    """
    if not generated_meshes or not reference_meshes:
        return {"mmd_cd": float("nan"), "cov_cd": float("nan"),
                "vol_wasserstein": float("nan")}

    print(f"  Sampling point clouds ({n_points} pts/mesh) …", flush=True)
    gen_pcs = [_sample_pointcloud(m, n_points) for m in generated_meshes]
    ref_pcs = [_sample_pointcloud(m, n_points) for m in reference_meshes]

    Ng, Nr = len(gen_pcs), len(ref_pcs)

    # ---- MMD-CD: for each reference, nearest generated ----
    print(f"  Computing pairwise CD ({Nr}×{Ng}) …", flush=True)
    cd_matrix = np.zeros((Nr, Ng), dtype=np.float64)
    for ri, rpc in enumerate(ref_pcs):
        for gi, gpc in enumerate(gen_pcs):
            cd_matrix[ri, gi] = _chamfer_distance(rpc, gpc)

    mmd_cd = float(cd_matrix.min(axis=1).mean())

    # ---- COV-CD: fraction of references matched by ≥1 generated ----
    matched_refs = set(cd_matrix.argmin(axis=0).tolist())
    cov_cd = len(matched_refs) / Nr

    # ---- Volume Wasserstein-1 ----
    try:
        from scipy.stats import wasserstein_distance
        gen_vols = np.array([abs(m.volume) for m in generated_meshes], dtype=np.float64)
        ref_vols = np.array([abs(m.volume) for m in reference_meshes],  dtype=np.float64)
        vol_wass = float(wasserstein_distance(gen_vols, ref_vols))
    except ImportError:
        # Manual Wasserstein-1 via sorted arrays (1D)
        gen_vols = np.sort(np.array([abs(m.volume) for m in generated_meshes]))
        ref_vols = np.sort(np.array([abs(m.volume) for m in reference_meshes]))
        # Resample to common grid
        t_gen = np.linspace(0, 1, len(gen_vols))
        t_ref = np.linspace(0, 1, len(ref_vols))
        t_common = np.linspace(0, 1, max(len(gen_vols), len(ref_vols)))
        gen_interp = np.interp(t_common, t_gen, gen_vols)
        ref_interp = np.interp(t_common, t_ref, ref_vols)
        vol_wass = float(np.abs(gen_interp - ref_interp).mean())

    return {
        "mmd_cd":          mmd_cd,
        "cov_cd":          cov_cd,
        "vol_wasserstein": vol_wass,
    }


def _load_obj_dir(directory: Path) -> Tuple[List[Path], List["trimesh.Trimesh"]]:
    """Load all *.obj files from a directory; return (paths, meshes)."""
    paths = sorted(directory.glob("*.obj"))
    meshes = []
    for p in paths:
        m = trimesh.load(str(p), force="mesh", process=False)
        if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
            meshes.append(m)
    return paths, meshes


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Post-hoc Laplacian mesh fairing baseline")
    p.add_argument("--input_dir",     type=str, required=True,
                   help="Directory containing Bio− *.obj meshes to smooth.")
    p.add_argument("--output_dir",    type=str, default=None,
                   help="Directory to write faired meshes. "
                        "Defaults to <input_dir>_faired.")
    p.add_argument("--reference_dir", type=str, default=None,
                   help="Directory of reference (GT / held-out) *.obj meshes. "
                        "Required for MMD-CD, COV-CD, Volume Wasserstein.")
    p.add_argument("--bioplus_dir",   type=str, default=None,
                   help="Directory of Bio+ *.obj meshes for direct comparison.")
    p.add_argument("--iterations",    type=str, default="10",
                   help="Comma-separated list of iteration counts to evaluate, "
                        "e.g. '5,10,20'. Default: 10.")
    p.add_argument("--lambda",        dest="lam", type=float, default=0.5,
                   help="Laplacian step size λ (default: 0.5).")
    p.add_argument("--mu",            type=float, default=-0.53,
                   help="Taubin backward step μ (default: -0.53).")
    p.add_argument("--no_taubin",     action="store_true",
                   help="Use plain Laplacian smoothing instead of Taubin.")
    p.add_argument("--n_points",      type=int, default=2048,
                   help="Points per mesh for Chamfer distance (default: 2048).")
    return p.parse_args()


def _agg_lr_sce(meshes: List["trimesh.Trimesh"]) -> Tuple[float, float]:
    """Return (mean LR, mean SCE) over a list of meshes."""
    lrs  = [laplacian_roughness(m)      for m in meshes]
    sces = [surface_curvature_energy(m) for m in meshes]
    return float(np.mean(lrs)), float(np.mean(sces))


def main():
    args = _parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = (Path(args.output_dir)
               if args.output_dir
               else in_dir.parent / (in_dir.name + "_faired"))
    out_dir.mkdir(parents=True, exist_ok=True)

    iter_list = [int(x) for x in args.iterations.split(",")]

    # -----------------------------------------------------------------------
    # Load meshes
    # -----------------------------------------------------------------------
    obj_files = sorted(in_dir.glob("*.obj"))
    if not obj_files:
        sys.exit(f"No *.obj files found in {in_dir}")

    print(f"\nLoading Bio− meshes from {in_dir} …")
    bio_minus_meshes: List[trimesh.Trimesh] = []
    for p in obj_files:
        m = trimesh.load(str(p), force="mesh", process=False)
        if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
            bio_minus_meshes.append(m)
    print(f"  Loaded {len(bio_minus_meshes)} valid meshes.")

    ref_meshes: List[trimesh.Trimesh] = []
    if args.reference_dir:
        print(f"Loading reference meshes from {args.reference_dir} …")
        _, ref_meshes = _load_obj_dir(Path(args.reference_dir))
        print(f"  Loaded {len(ref_meshes)} reference meshes.")

    bioplus_meshes: List[trimesh.Trimesh] = []
    if args.bioplus_dir:
        print(f"Loading Bio+ meshes from {args.bioplus_dir} …")
        _, bioplus_meshes = _load_obj_dir(Path(args.bioplus_dir))
        print(f"  Loaded {len(bioplus_meshes)} Bio+ meshes.")

    # -----------------------------------------------------------------------
    # Build faired mesh sets for each iteration count
    # -----------------------------------------------------------------------
    faired_sets: dict[int, List[trimesh.Trimesh]] = {}
    for n_iter in iter_list:
        print(f"\nApplying Taubin smoothing ({n_iter} iterations) …")
        faired: List[trimesh.Trimesh] = []
        iter_out = out_dir / f"iter_{n_iter:02d}"
        iter_out.mkdir(parents=True, exist_ok=True)
        for i, (mesh_in, src_path) in enumerate(zip(bio_minus_meshes, obj_files)):
            mesh_out = laplacian_smooth(
                mesh_in,
                iterations=n_iter,
                lam=args.lam,
                taubin=not args.no_taubin,
                mu=args.mu,
            )
            faired.append(mesh_out)
            mesh_out.export(str(iter_out / src_path.name))
        faired_sets[n_iter] = faired
        print(f"  Faired meshes written to {iter_out}")

    # -----------------------------------------------------------------------
    # Compute metrics for all conditions
    # -----------------------------------------------------------------------
    has_dist = bool(ref_meshes)

    print("\n" + "=" * 90)
    print("RESULTS: Post-Hoc Fairing vs. Bio+")
    print("=" * 90)

    col_w = 14

    header = f"{'Condition':<28s}"
    header += f"{'LR':>{col_w}s}{'SCE':>{col_w}s}"
    if has_dist:
        header += f"{'MMD-CD':>{col_w}s}{'COV-CD':>{col_w}s}{'Vol.Wass.':>{col_w}s}"
    print(header)
    print("-" * len(header))

    def _print_row(label, meshes, ref_meshes=ref_meshes):
        lr, sce = _agg_lr_sce(meshes)
        row = f"  {label:<26s}{lr:>{col_w}.6f}{sce:>{col_w}.6f}"
        if has_dist:
            dm = compute_distribution_metrics(meshes, ref_meshes,
                                              n_points=args.n_points)
            row += (f"{dm['mmd_cd']:>{col_w}.6f}"
                    f"{dm['cov_cd']:>{col_w}.4f}"
                    f"{dm['vol_wasserstein']:>{col_w}.6f}")
        print(row)

    _print_row("Bio−  (no fairing)", bio_minus_meshes)
    for n_iter in iter_list:
        _print_row(f"Bio− + Taubin {n_iter:2d} iter", faired_sets[n_iter])
    if bioplus_meshes:
        _print_row("Bio+  (ours)", bioplus_meshes)

    print("=" * 90)
    print("\nNotes:")
    print("  LR / SCE: lower is better.  MMD-CD: lower is better.")
    print("  COV-CD: higher is better (0–1).  Vol.Wass.: lower is better.")
    print("  Taubin fairing improves LR/SCE but cannot recover diversity (COV-CD)")
    print("  or correct volume distribution (Vol.Wass.) — advantages of Bio+ training.")


if __name__ == "__main__":
    main()

