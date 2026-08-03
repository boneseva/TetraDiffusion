#!/usr/bin/env python3

import os
import glob
import argparse
import time
import io
import numpy as np
import scipy.spatial
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def normalize_point_cloud(pc):
    """Center point cloud at origin and scale to unit bounding sphere."""
    if len(pc) == 0:
        return pc
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    max_radius = np.max(np.linalg.norm(pc_centered, axis=1))
    if max_radius > 1e-7:
        return pc_centered / max_radius
    return pc_centered

def sample_point_cloud(mesh, num_points=2048):
    """Sample normalized point cloud from mesh or point cloud."""
    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            mesh = None

    vertices = getattr(mesh, 'vertices', None)
    faces = getattr(mesh, 'faces', None)

    if vertices is None or len(vertices) == 0:
        pts = np.random.randn(num_points, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    if isinstance(mesh, trimesh.PointCloud) or faces is None or len(faces) == 0:
        idx = np.random.choice(len(vertices), num_points, replace=True)
        pts = vertices[idx]
    else:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        except Exception:
            idx = np.random.choice(len(vertices), num_points, replace=True)
            pts = vertices[idx]

    return normalize_point_cloud(pts)

def render_shape_thumbnail(pc, color='#1f77b4', size_px=100, elev=20, azim=45):
    """
    Render a 2D thumbnail image of a 3D point cloud with fixed camera elevation/azimuth.
    Returns RGB uint8 numpy array.
    """
    dpi = 100
    fig_size = size_px / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    
    # Set background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Plot 3D point cloud
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=2.5, alpha=0.85, linewidths=0)

    # Uniform aspect ratio limits
    max_range = 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    img = plt.imread(buf)
    return img

def compute_chamfer_distance(pc1, pc2):
    """Compute bidirectional Chamfer Distance between two point clouds."""
    tree1 = scipy.spatial.KDTree(pc1)
    tree2 = scipy.spatial.KDTree(pc2)
    d1, _ = tree2.query(pc1, k=1)
    d2, _ = tree1.query(pc2, k=1)
    return float(np.mean(d1**2) + np.mean(d2**2))

def classical_mds(D, n_components=2):
    """
    Classical Multidimensional Scaling (PCoA) on distance matrix D.
    Maps pairwise Chamfer distances directly into 2D Cartesian coordinates.
    """
    K = D.shape[0]
    H = np.eye(K) - np.ones((K, K)) / K  # Centering matrix
    B = -0.5 * H.dot(D**2).dot(H)        # Double-centered Gram matrix
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1][:n_components]
    evals = np.maximum(evals[idx], 0.0)
    evecs = evecs[:, idx]
    return evecs * np.sqrt(evals)

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D Shape Space Manifold with ACTUAL 3D Shape Thumbnails & 1-NN Connections.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run inference output folder (e.g. runs/abl_bio_on/inference_...)")
    parser.add_argument("--gt_dir", type=str, default="../data_test/organelles/lyso", help="Path to GT directory.")
    parser.add_argument("--points", type=int, default=2048, help="Number of points per cloud.")
    parser.add_argument("--max_gen", type=int, default=25, help="Max generated shapes to display (default: 25 for clear visual rendering).")
    parser.add_argument("--output", type=str, default=None, help="Output plot filename.")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.obj")))
    if not gt_files:
        gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "**", "*.obj"), recursive=True))

    gen_files = sorted(glob.glob(os.path.join(args.run_dir, "*.obj")))
    if not gen_files:
        gen_files = sorted(glob.glob(os.path.join(args.run_dir, "**", "*.obj"), recursive=True))

    gt_files = gt_files[:50]
    gen_files = gen_files[:args.max_gen]

    if not gt_files or not gen_files:
        print(f"Error: Need both GT files ({len(gt_files)}) and Gen files ({len(gen_files)}).")
        return

    print(f"Loading {len(gt_files)} GT meshes and {len(gen_files)} Generated meshes...")
    gt_pcs = [sample_point_cloud(trimesh.load(f), args.points) for f in gt_files]
    gen_pcs = [sample_point_cloud(trimesh.load(f), args.points) for f in gen_files]

    all_pcs = gt_pcs + gen_pcs
    num_gt = len(gt_pcs)
    num_gen = len(gen_pcs)
    Total = num_gt + num_gen

    print(f"Computing {Total}x{Total} pairwise Chamfer Distance matrix...")
    t0 = time.time()
    D = np.zeros((Total, Total))
    for i in range(Total):
        for j in range(i + 1, Total):
            cd = compute_chamfer_distance(all_pcs[i], all_pcs[j])
            D[i, j] = cd
            D[j, i] = cd
    print(f"Distance matrix computed in {time.time() - t0:.1f}s.")

    # Compute 1-NN for each shape
    nn_indices = []
    for i in range(Total):
        D_temp = D[i].copy()
        D_temp[i] = float('inf')
        nn_indices.append(np.argmin(D_temp))

    print("Projecting distance matrix to 2D via Classical MDS...")
    coords = classical_mds(D, n_components=2)

    print("Rendering 3D shape thumbnails...")
    t1 = time.time()
    thumbnails = []
    for i in range(Total):
        is_gt = (i < num_gt)
        color = '#1f77b4' if is_gt else '#d62728'  # Blue for GT, Red for Generated
        img = render_shape_thumbnail(all_pcs[i], color=color, size_px=90)
        thumbnails.append(img)
    print(f"Thumbnails rendered in {time.time() - t1:.1f}s.")

    # Setup Plot
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    ax.set_facecolor('#f9f9fb')

    # Draw 1-NN arrows/lines from each generated shape to its closest GT shape
    for i in range(num_gt, Total):  # Generated shapes
        j = nn_indices[i]            # Nearest neighbor (could be GT or another Gen)
        is_nn_gt = (j < num_gt)
        line_color = '#2ca02c' if is_nn_gt else '#ff7f0e'  # Green line if matched to GT, Orange if matched to Fake
        ax.annotate('', xy=(coords[j, 0], coords[j, 1]), xytext=(coords[i, 0], coords[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=line_color, lw=1.5, ls="--", alpha=0.7))

    # Place 3D Shape Thumbnails at (x, y) coordinates with colored borders
    for i in range(Total):
        is_gt = (i < num_gt)
        border_color = '#1f77b4' if is_gt else '#d62728'
        imagebox = OffsetImage(thumbnails[i], zoom=0.5)
        ab = AnnotationBbox(
            imagebox,
            (coords[i, 0], coords[i, 1]),
            frameon=True,
            bboxprops=dict(edgecolor=border_color, linewidth=2.0, facecolor='white', boxstyle='round,pad=0.15'),
            pad=0.0
        )
        ax.add_artist(ab)

    # Dummy legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Ground Truth (Real, Blue Border, N={num_gt})', markerfacecolor='#1f77b4', markersize=12),
        Line2D([0], [0], marker='o', color='w', label=f'Generated (Fake, Red Border, N={num_gen})', markerfacecolor='#d62728', markersize=12),
        Line2D([0], [0], color='#2ca02c', lw=1.5, ls='--', label='1-NN Match: Generated → Ground Truth'),
        Line2D([0], [0], color='#ff7f0e', lw=1.5, ls='--', label='1-NN Match: Generated → Generated')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=11)

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    ax.set_title(f"3D Organelle Shape Space & 1-NN Nearest Neighbors\nRun: {run_name}", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("MDS Axis 1 (Chamfer Shape Metric Space)", fontsize=12)
    ax.set_ylabel("MDS Axis 2 (Chamfer Shape Metric Space)", fontsize=12)

    # Expand margin limits to fit thumbnail boxes
    x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.15
    y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.15
    ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
    ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_name = args.output or f"shape_space_actual_shapes_{run_name}.png"
    out_path = os.path.join(results_dir, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"\nActual 3D Shape Space plot saved to:")
    print(f"  - {out_path}")
    print(f"  - {pdf_path}")

if __name__ == '__main__':
    main()
