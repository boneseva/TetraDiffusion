import os
import sys
import time
import glob
import argparse
import numpy as np
import pandas as pd
import trimesh
from metrics import (
    sample_point_cloud,
    compute_chamfer_and_fscore,
    compute_sphericity,
    compute_mesh_quality,
    compute_coverage,
    compute_1nn_accuracy,
    normalize_point_cloud
)

def log(msg):
    """Print immediately without line buffering for SLURM tail -f tracking."""
    print(msg, flush=True)

def load_gt_point_clouds(gt_dir, num_points=2048):
    """Load all ground truth meshes and sample normalized point clouds from them."""
    gt_files = glob.glob(os.path.join(gt_dir, "*.obj"))
    if not gt_files:
        log(f"WARNING: No ground truth OBJ files found in {gt_dir}")
        return []
        
    log(f"Loading {len(gt_files)} ground truth meshes from {gt_dir}...")
    gt_pcs = []
    t0 = time.time()
    for i, f in enumerate(gt_files, 1):
        try:
            mesh = trimesh.load(f)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            pc = sample_point_cloud(mesh, num_points, normalize=True)
            gt_pcs.append(pc)
        except Exception as e:
            log(f"  Error loading GT file {f}: {e}")
        if i % 10 == 0 or i == len(gt_files):
            log(f"  GT Progress: {i}/{len(gt_files)} loaded ({time.time() - t0:.1f}s)")
    return gt_pcs

def evaluate_run(run_dir, gt_pcs, num_points=2048, fscore_threshold=0.05):
    """Calculate average metrics for all generated meshes in a directory."""
    gen_files = sorted(glob.glob(os.path.join(run_dir, "*.obj")))
    if not gen_files:
        return None
        
    total_files = len(gen_files)
    t0 = time.time()
    
    gen_pcs = []
    cds = []
    fscores = []
    sphericities = []
    watertight_count = 0
    cc_counts = []
    degen_fractions = []
    
    for i, f in enumerate(gen_files, 1):
        try:
            mesh = trimesh.load(f)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # 1. Geometry Metrics
            sph = compute_sphericity(mesh)
            sphericities.append(sph)
            
            qual = compute_mesh_quality(mesh)
            if qual["watertight"]:
                watertight_count += 1
            cc_counts.append(qual["connected_components"])
            degen_fractions.append(qual["degenerate_faces_fraction"])
            
            # 2. Distribution / Chamfer Metrics (Normalized to Unit Sphere)
            gen_pc = sample_point_cloud(mesh, num_points, normalize=True)
            gen_pcs.append(gen_pc)
            
            best_cd = float('inf')
            best_f = 0.0
            
            for gt_pc in gt_pcs:
                cd, fscore = compute_chamfer_and_fscore(gen_pc, gt_pc, fscore_threshold)
                if cd < best_cd:
                    best_cd = cd
                if fscore > best_f:
                    best_f = fscore
                    
            if best_cd != float('inf'):
                cds.append(best_cd)
                fscores.append(best_f)
                
        except Exception as e:
            log(f"  Warning: Error evaluating mesh {os.path.basename(f)}: {e}")
            
        if i % 20 == 0 or i == total_files:
            elapsed = time.time() - t0
            speed = i / max(elapsed, 0.001)
            eta = (total_files - i) / max(speed, 0.001)
            log(f"    - Processed {i}/{total_files} meshes ({elapsed:.1f}s, ETA: {eta:.1f}s)")
            
    if not cds:
        return None

    # Compute dataset-level 3D generative benchmarks (Coverage & 1-NN Accuracy)
    log("    - Computing Coverage (COV) and 1-NN Accuracy...")
    cov = compute_coverage(gen_pcs, gt_pcs, fscore_threshold)
    onn_acc = compute_1nn_accuracy(gen_pcs, gt_pcs, fscore_threshold)
        
    return {
        "CD_MMD": np.mean(cds),
        "FScore_MMD": np.mean(fscores),
        "COV (%)": cov,
        "1NN_Acc (%)": onn_acc,
        "Sphericity": np.mean(sphericities),
        "Watertight_Ratio": watertight_count / total_files,
        "Connected_Components": np.mean(cc_counts),
        "Degenerate_Faces": np.mean(degen_fractions),
        "Mesh_Count": total_files
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare generated 3D meshes to GT database.")
    parser.add_argument("--runs_dir", type=str, default="../runs", help="Directory where training runs are stored.")
    parser.add_argument("--gt_dir", type=str, default="../data_test/organelles/lyso", help="Directory with GT meshes.")
    parser.add_argument("--fscore_thresh", type=float, default=0.05, help="Threshold distance for F-Score (unit sphere scale).")
    parser.add_argument("--points", type=int, default=2048, help="Number of points to sample from each mesh.")
    parser.add_argument("--filter", type=str, default=None, help="Pattern/prefix to filter run directories (e.g., 'abl_' or '*bio*').")
    args = parser.parse_args()
    
    # 1. Load Ground Truth datasets
    gt_pcs = load_gt_point_clouds(args.gt_dir, num_points=args.points)
    if not gt_pcs:
        log("Error: Ground truth meshes are required for evaluation. Exiting.")
        return
        
    # 2. Scan for evaluation folders containing OBJ files
    filter_msg = f" (filtered by: '{args.filter}')" if args.filter else ""
    log(f"\nScanning for directories containing generated meshes in {args.runs_dir}{filter_msg}...")
    
    candidate_dirs = []
    for root, dirs, files in os.walk(args.runs_dir):
        obj_files = [f for f in files if f.endswith('.obj')]
        if obj_files:
            rel_path = os.path.relpath(root, args.runs_dir)
            if rel_path == ".":
                continue
                
            # Only evaluate dedicated inference directories (skip training validation samples)
            if "inference_" not in rel_path:
                continue

            if args.filter:
                import fnmatch
                top_dir = rel_path.split(os.sep)[0]
                pattern = args.filter
                if not ('*' in pattern or '?' in pattern):
                    pattern = f"{pattern}*"
                
                if not fnmatch.fnmatch(top_dir.lower(), pattern.lower()):
                    continue
            candidate_dirs.append((root, rel_path, len(obj_files)))

    if not candidate_dirs:
        log("No generated meshes (.obj) found to evaluate in any run subdirectories.")
        return

    log(f"Found {len(candidate_dirs)} directory(ies) with meshes to evaluate.")
    log("-" * 80)
    for idx, (root, rel_path, count) in enumerate(candidate_dirs, 1):
        log(f"  [{idx}/{len(candidate_dirs)}] {rel_path} ({count} meshes)")
    log("-" * 80 + "\n")

    run_results = {}
    total_start = time.time()

    for idx, (root, rel_path, count) in enumerate(candidate_dirs, 1):
        log(f"[{idx}/{len(candidate_dirs)}] Evaluating '{rel_path}' ({count} meshes)...")
        run_start = time.time()
        metrics = evaluate_run(root, gt_pcs, num_points=args.points, fscore_threshold=args.fscore_thresh)
        run_time = time.time() - run_start
        
        if metrics:
            run_results[rel_path] = metrics
            log(f"  ✓ Finished '{rel_path}' in {run_time:.1f}s | CD_MMD: {metrics['CD_MMD']:.6f} | FScore: {metrics['FScore_MMD']:.4f} | COV: {metrics['COV (%)']:.1f}% | 1NN: {metrics['1NN_Acc (%)']:.1f}%\n")
        else:
            log(f"  ✗ Failed/Empty metrics for '{rel_path}' in {run_time:.1f}s\n")
            
    total_time = time.time() - total_start
    log(f"Evaluation complete for {len(run_results)} run(s) in {total_time:.1f} seconds.")

    # 3. Format & print results
    df = pd.DataFrame.from_dict(run_results, orient='index')
    df = df.sort_values(by="CD_MMD", ascending=True)
    
    log("\n" + "="*80)
    log(" EVALUATION COMPARISON RESULTS (Sorted by CD MMD - Lower is Better)")
    log("="*80)
    log(df.to_string())
    log("="*80)
    
    # Save results to CSV & Markdown
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/evaluation_summary.csv")
    
    with open("results/evaluation_summary.md", "w") as f:
        f.write("# TetraDiffusion Ablation Evaluation Summary\n\n")
        f.write("Generated on: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("## Runs Comparison Table\n\n")
        f.write(df.to_markdown() + "\n\n")
        f.write("### Metrics Reference:\n")
        f.write("* **CD_MMD**: Minimum Modified Chamfer Distance on unit-sphere normalized point clouds (lower = closer to GT shape distribution).\n")
        f.write("* **FScore_MMD**: F-Score at threshold 0.05 (higher = better surface coverage accuracy).\n")
        f.write("* **COV (%)**: Coverage percentage (higher = better diversity, max 100%).\n")
        f.write("* **1NN_Acc (%)**: 1-Nearest Neighbor classifier accuracy (ideal = 50.0%).\n")
        f.write("* **Sphericity**: Volume-to-surface compactness ratio (1.0 = perfect sphere, ideal for Lysosomes).\n")
        f.write("* **Watertight_Ratio**: Fraction of meshes that are closed/watertight (higher = cleaner geometry).\n")
        f.write("* **Connected_Components**: Average number of disconnected mesh parts (ideal = 1.0; >1.0 indicates background noise/floaters).\n")
        f.write("* **Degenerate_Faces**: Fraction of faces with near-zero area (lower = better mesh quality).\n")

    log("\nResults successfully saved to:")
    log("  - evaluation/results/evaluation_summary.csv")
    log("  - evaluation/results/evaluation_summary.md")

if __name__ == '__main__':
    main()
