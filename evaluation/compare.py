import os
import glob
import argparse
import numpy as np
import pandas as pd
import trimesh
from metrics import sample_point_cloud, compute_chamfer_and_fscore, compute_sphericity, compute_mesh_quality

def load_gt_point_clouds(gt_dir, num_points=2048):
    """Load all ground truth meshes and sample point clouds from them."""
    gt_files = glob.glob(os.path.join(gt_dir, "*.obj"))
    if not gt_files:
        print(f"WARNING: No ground truth OBJ files found in {gt_dir}")
        return []
        
    print(f"Loading {len(gt_files)} ground truth meshes from {gt_dir}...")
    gt_pcs = []
    for f in gt_files:
        try:
            mesh = trimesh.load(f)
            pc = sample_point_cloud(mesh, num_points)
            gt_pcs.append(pc)
        except Exception as e:
            print(f"Error loading GT file {f}: {e}")
    return gt_pcs

def evaluate_run(run_dir, gt_pcs, num_points=2048, fscore_threshold=0.02):
    """Calculate average metrics for all generated meshes in a directory."""
    gen_files = glob.glob(os.path.join(run_dir, "*.obj"))
    if not gen_files:
        return None
        
    print(f"Evaluating {len(gen_files)} generated meshes in {run_dir}...")
    
    cds = []
    fscores = []
    sphericities = []
    watertight_count = 0
    cc_counts = []
    degen_fractions = []
    
    for f in gen_files:
        try:
            mesh = trimesh.load(f)
            
            # 1. Geometry Metrics
            sph = compute_sphericity(mesh)
            sphericities.append(sph)
            
            qual = compute_mesh_quality(mesh)
            if qual["watertight"]:
                watertight_count += 1
            cc_counts.append(qual["connected_components"])
            degen_fractions.append(qual["degenerate_faces_fraction"])
            
            # 2. Distribution / Chamfer Metrics
            # Sample point cloud from generated shape
            gen_pc = sample_point_cloud(mesh, num_points)
            
            # Find nearest neighbor in GT set (Minimum Modified Distance)
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
            print(f"Error evaluating mesh {f}: {e}")
            
    if not cds:
        return None
        
    return {
        "CD_MMD": np.mean(cds),
        "FScore_MMD": np.mean(fscores),
        "Sphericity": np.mean(sphericities),
        "Watertight_Ratio": watertight_count / len(gen_files),
        "Connected_Components": np.mean(cc_counts),
        "Degenerate_Faces": np.mean(degen_fractions),
        "Mesh_Count": len(gen_files)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare generated 3D meshes to GT database.")
    parser.add_argument("--runs_dir", type=str, default="../runs", help="Directory where training runs are stored.")
    parser.add_argument("--gt_dir", type=str, default="../data_test/organelles/lyso", help="Directory with GT meshes.")
    parser.add_argument("--fscore_thresh", type=float, default=0.02, help="Threshold distance for F-Score.")
    parser.add_argument("--points", type=int, default=2048, help="Number of points to sample from each mesh.")
    parser.add_argument("--filter", type=str, default=None, help="Pattern/prefix to filter run directories (e.g., 'abl_' or '*bio*').")
    args = parser.parse_args()
    
    # 1. Load Ground Truth datasets
    gt_pcs = load_gt_point_clouds(args.gt_dir, num_points=args.points)
    if not gt_pcs:
        print("Error: Ground truth meshes are required for evaluation. Exiting.")
        return
        
    # 2. Scan for evaluation folders containing OBJ files
    filter_msg = f" (filtered by: '{args.filter}')" if args.filter else ""
    print(f"Scanning for directories containing generated meshes in {args.runs_dir}{filter_msg}...")
    run_results = {}
    
    # Walk through subdirectories to find folders containing OBJ files
    for root, dirs, files in os.walk(args.runs_dir):
        # Skip top-level runs directory itself, look for leaf directories with OBJs
        obj_files = [f for f in files if f.endswith('.obj')]
        if obj_files:
            # Determine a friendly name for this run output
            rel_path = os.path.relpath(root, args.runs_dir)
            
            # Avoid picking up top-level runs folder directly
            if rel_path == ".":
                continue
                
            # Filter runs if a pattern is provided
            if args.filter:
                import fnmatch
                top_dir = rel_path.split(os.sep)[0]
                # Auto-append '*' if user passes a simple prefix (like 'abl_')
                pattern = args.filter
                if not ('*' in pattern or '?' in pattern):
                    pattern = f"{pattern}*"
                
                # Check case-insensitive match
                if not fnmatch.fnmatch(top_dir.lower(), pattern.lower()):
                    continue
                
            metrics = evaluate_run(root, gt_pcs, num_points=args.points, fscore_threshold=args.fscore_thresh)
            if metrics:
                run_results[rel_path] = metrics
                
    if not run_results:
        print("No generated meshes (.obj) found to evaluate in any run subdirectories.")
        return
        
    # 3. Format & print results
    df = pd.DataFrame.from_dict(run_results, orient='index')
    
    # Sort by Chamfer Distance MMD (lower is better)
    df = df.sort_values(by="CD_MMD", ascending=True)
    
    print("\n" + "="*80)
    print(" EVALUATION COMPARISON RESULTS (Sorted by CD MMD - Lower is Better)")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Save results to CSV & Markdown
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/evaluation_summary.csv")
    
    with open("results/evaluation_summary.md", "w") as f:
        f.write("# TetraDiffusion Ablation Evaluation Summary\n\n")
        f.write("Generated on: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("## Runs Comparison Table\n\n")
        f.write(df.to_markdown() + "\n\n")
        f.write("### Metrics Reference:\n")
        f.write("* **CD_MMD**: Minimum Modified Chamfer Distance (lower = closer to GT shape distribution).\n")
        f.write("* **FScore_MMD**: F-Score at threshold (higher = better surface coverage accuracy).\n")
        f.write("* **Sphericity**: Volume-to-surface compactness ratio (1.0 = perfect sphere, ideal for Lysosomes).\n")
        f.write("* **Watertight_Ratio**: Fraction of meshes that are closed/watertight (higher = cleaner geometry).\n")
        f.write("* **Connected_Components**: Average number of disconnected mesh parts (ideal = 1.0; >1.0 indicates background noise/floaters).\n")
        f.write("* **Degenerate_Faces**: Fraction of faces with near-zero area (lower = better mesh quality).\n")

    print("\nResults successfully saved to:")
    print("  - evaluation/results/evaluation_summary.csv")
    print("  - evaluation/results/evaluation_summary.md")

if __name__ == "__main__":
    main()
