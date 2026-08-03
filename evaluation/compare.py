import os
import sys
import time
import glob
import json
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
    compute_1nn_accuracy_decomposed,
    compute_morphological_features,
    compute_wasserstein_distances,
    normalize_point_cloud
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def log(msg):
    """Print immediately without line buffering for SLURM tail -f tracking."""
    print(msg, flush=True)

def save_intermediate_results(run_results, cache_file=None):
    """
    Save accumulated metrics to JSON cache, CSV, and Markdown live after each run.
    Ensures work is never lost if a job is cancelled or interrupted.
    All outputs strictly land in evaluation/results/.
    """
    if not run_results:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if cache_file is None:
        cache_file = os.path.join(RESULTS_DIR, "cache_evaluation.json")

    # 1. Save JSON cache for instant resumption
    try:
        with open(cache_file, "w") as f:
            json.dump(run_results, f, indent=2)
    except Exception as e:
        log(f"Warning: Could not update JSON cache: {e}")

    # 2. Save CSV spreadsheet
    df = pd.DataFrame.from_dict(run_results, orient='index')
    df = df.sort_values(by="CD_MMD", ascending=True)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    df.to_csv(csv_path)

    # 3. Save Markdown summary table
    md_path = os.path.join(RESULTS_DIR, "evaluation_summary.md")
    with open(md_path, "w") as f:
        f.write("# TetraDiffusion Ablation Evaluation Summary\n\n")
        f.write("Last updated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write(f"Evaluated runs: {len(run_results)}\n\n")
        f.write("## Runs Comparison Table\n\n")
        f.write(df.to_markdown() + "\n\n")
        f.write("### Metrics Reference:\n")
        f.write("* **CD_MMD**: Minimum Modified Chamfer Distance on unit-sphere normalized point clouds (lower = closer to GT shape distribution).\n")
        f.write("* **FScore_MMD**: F-Score at threshold 0.05 (higher = better surface coverage accuracy).\n")
        f.write("* **COV (%)**: Coverage percentage (higher = better diversity, max 100%).\n")
        f.write("* **1NN_Total (%)**: Overall 1-Nearest Neighbor classifier accuracy (ideal = 50.0%).\n")
        f.write("* **1NN_Fake (%)**: % of generated shapes whose 1-NN is also generated (ideal = 50.0%, precision indicator).\n")
        f.write("* **1NN_Real (%)**: % of real GT shapes whose 1-NN is also real (ideal = 50.0%, recall/coverage indicator).\n")
        f.write("* **W1_Volume**: Wasserstein distance between GT and generated volume distributions (lower = better physical size calibration).\n")
        f.write("* **W1_Area**: Wasserstein distance between GT and generated surface area distributions (lower = better surface scaling).\n")
        f.write("* **W1_Aspect**: Wasserstein distance between GT and generated aspect ratio distributions (lower = better elongation alignment).\n")
        f.write("* **Sphericity**: Volume-to-surface compactness ratio (1.0 = perfect sphere, ideal for Lysosomes).\n")
        f.write("* **Watertight_Ratio**: Fraction of meshes that are closed/watertight (higher = cleaner geometry).\n")
        f.write("* **Connected_Components**: Average number of disconnected mesh parts (ideal = 1.0; >1.0 indicates background noise/floaters).\n")
        f.write("* **Degenerate_Faces**: Fraction of faces with near-zero area (lower = better mesh quality).\n")

def load_gt_data(gt_dir, num_points=2048):
    """Load all ground truth meshes, extract morphological features, and sample normalized point clouds."""
    gt_files = glob.glob(os.path.join(gt_dir, "*.obj"))
    if not gt_files:
        log(f"WARNING: No ground truth OBJ files found in {gt_dir}")
        return [], []
        
    log(f"Loading {len(gt_files)} ground truth meshes from {gt_dir}...")
    gt_pcs = []
    gt_features = []
    t0 = time.time()
    for i, f in enumerate(gt_files, 1):
        try:
            mesh = trimesh.load(f)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            pc = sample_point_cloud(mesh, num_points, normalize=True)
            feat = compute_morphological_features(mesh)
            gt_pcs.append(pc)
            gt_features.append(feat)
        except Exception as e:
            log(f"  Error loading GT file {f}: {e}")
        if i % 10 == 0 or i == len(gt_files):
            log(f"  GT Progress: {i}/{len(gt_files)} loaded ({time.time() - t0:.1f}s)")
    return gt_pcs, gt_features

def evaluate_run(run_dir, gt_pcs, gt_features, num_points=2048, fscore_threshold=0.05):
    """Calculate average metrics for all generated meshes in a directory."""
    gen_files = sorted(glob.glob(os.path.join(run_dir, "*.obj")))
    if not gen_files:
        return None
        
    total_files = len(gen_files)
    t0 = time.time()
    
    gen_pcs = []
    gen_features = []
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
            
            # 1. Geometry & Morphological Features
            sph = compute_sphericity(mesh)
            sphericities.append(sph)
            
            feat = compute_morphological_features(mesh)
            gen_features.append(feat)
            
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

    # Compute dataset-level 3D generative benchmarks
    log("    - Computing Coverage (COV), Decomposed 1-NN, and Morphological Wasserstein W1 distances...")
    cov = compute_coverage(gen_pcs, gt_pcs, fscore_threshold)
    onn_dict = compute_1nn_accuracy_decomposed(gen_pcs, gt_pcs, fscore_threshold)
    w1_dict = compute_wasserstein_distances(gen_features, gt_features)
        
    res = {
        "CD_MMD": np.mean(cds),
        "FScore_MMD": np.mean(fscores),
        "COV (%)": cov,
        "1NN_Total (%)": onn_dict["1NN_Total (%)"],
        "1NN_Fake (%)": onn_dict["1NN_Fake (%)"],
        "1NN_Real (%)": onn_dict["1NN_Real (%)"],
        "W1_Volume": w1_dict["W1_Volume"],
        "W1_Area": w1_dict["W1_Area"],
        "W1_Aspect": w1_dict["W1_Aspect"],
        "Sphericity": np.mean(sphericities),
        "Watertight_Ratio": watertight_count / total_files,
        "Connected_Components": np.mean(cc_counts),
        "Degenerate_Faces": np.mean(degen_fractions),
        "Mesh_Count": total_files
    }
    return res

def generate_run_visualizations(root_dir, gt_dir, force=False):
    """Automatically generate static plots (.png/.pdf) & interactive web HTML app for a run in evaluation/results/."""
    try:
        run_name = os.path.basename(os.path.normpath(root_dir))
        out_png = os.path.join(RESULTS_DIR, f"shape_space_actual_shapes_{run_name}.png")
        out_html = os.path.join(RESULTS_DIR, f"shape_space_interactive_{run_name}.html")

        plot_script = os.path.join(SCRIPT_DIR, "plot_shape_space.py")
        html_script = os.path.join(SCRIPT_DIR, "plot_shape_space_html.py")

        # Static plot
        if not os.path.exists(out_png) or force:
            log(f"    - Generating static plot: {out_png}")
            os.system(f"python3 '{plot_script}' --run_dir '{root_dir}' --gt_dir '{gt_dir}' >/dev/null 2>&1")

        # Interactive HTML app
        if not os.path.exists(out_html) or force:
            log(f"    - Generating interactive HTML: {out_html}")
            os.system(f"python3 '{html_script}' --run_dir '{root_dir}' --gt_dir '{gt_dir}' >/dev/null 2>&1")
    except Exception as e:
        log(f"    Warning: Plot generation failed for '{root_dir}': {e}")

def _dir_has_objs(directory):
    if not os.path.isdir(directory):
        return False
    objs = glob.glob(os.path.join(directory, "*.obj"))
    if objs:
        return True
    objs = glob.glob(os.path.join(directory, "**", "*.obj"), recursive=True)
    return len(objs) > 0

def resolve_gt_dir(run_dir, default_gt_dir=None):
    """
    Automatically resolve the ground truth OBJ directory for a given run folder.
    Inspects config.yaml, inference_manifest.json, or folder name.
    """
    run_parent = run_dir
    while os.path.basename(run_parent) and "inference_" in os.path.basename(run_parent):
        run_parent = os.path.dirname(run_parent)

    config_path = os.path.join(run_parent, "config.yaml")
    category = None
    data_path = None

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                cfg_data = yaml.safe_load(f)
            if isinstance(cfg_data, dict):
                dataset_cfg = cfg_data.get('dataset', {})
                if isinstance(dataset_cfg, dict):
                    category = dataset_cfg.get('category')
                    data_path = dataset_cfg.get('data_path')
                category = category or cfg_data.get('category')
                data_path = data_path or cfg_data.get('data_path')
        except Exception:
            pass

    manifest_path = os.path.join(run_dir, "inference_manifest.json")
    if not category and os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            category = manifest.get('organelle') or manifest.get('category')
        except Exception:
            pass

    folder_name = os.path.basename(run_parent).lower()
    if not category:
        if "er" in folder_name:
            category = "er"
        elif "golgi" in folder_name:
            category = "golgi"
        elif "mito" in folder_name:
            category = "mito"
        elif "lyso" in folder_name:
            category = "lyso"
        elif "fv" in folder_name:
            category = "fv"

    category_variations = []
    if category:
        cat_str = str(category).strip()
        cat_lower = cat_str.lower()
        if "er" in cat_lower or "endoplasmic" in cat_lower:
            category_variations = ["ER", "er", "endoplasmic_reticulum"]
        elif "golgi" in cat_lower:
            category_variations = ["Golgi", "golgi"]
        elif "mito" in cat_lower:
            category_variations = ["Mitochondria", "mito", "mitochondria"]
        elif "lyso" in cat_lower:
            category_variations = ["Lysosome", "lyso", "lysosome"]
        elif "fv" in cat_lower or "vacuole" in cat_lower:
            category_variations = ["fv", "vacuole", "vacuoles"]
        else:
            category_variations = [cat_str, cat_str.upper(), cat_str.lower(), cat_str.capitalize()]

    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    is_urocell = "urocell" in folder_name or (data_path and "urocell" in str(data_path).lower())

    if is_urocell:
        search_roots = [
            os.path.join(repo_root, "data_urocell", "organelles"),
            os.path.join(repo_root, "data_urocell", "organelles_raw"),
            os.path.join(repo_root, "data", "organelles"),
            os.path.join(repo_root, "data", "organelles_raw"),
            os.path.join(repo_root, "data_test", "organelles"),
        ]
    else:
        search_roots = [
            os.path.join(repo_root, "data", "organelles"),
            os.path.join(repo_root, "data", "organelles_raw"),
            os.path.join(repo_root, "data_urocell", "organelles"),
            os.path.join(repo_root, "data_urocell", "organelles_raw"),
            os.path.join(repo_root, "data_test", "organelles"),
        ]

    for sroot in search_roots:
        for cvar in category_variations:
            cand = os.path.join(sroot, cvar)
            if _dir_has_objs(cand):
                return os.path.abspath(cand)

    if default_gt_dir and _dir_has_objs(default_gt_dir):
        return os.path.abspath(default_gt_dir)

    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare generated 3D meshes to GT database.")
    parser.add_argument("--runs_dir", type=str, default="../runs", help="Directory where training runs are stored.")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory with GT meshes (default: auto-detect per run).")
    parser.add_argument("--fscore_thresh", type=float, default=0.05, help="Threshold distance for F-Score (unit sphere scale).")
    parser.add_argument("--points", type=int, default=2048, help="Number of points to sample from each mesh.")
    parser.add_argument("--filter", type=str, default=None, help="Pattern/prefix to filter run directories (e.g., 'abl_' or '*bio*').")
    parser.add_argument("--force", action="store_true", help="Force re-evaluating runs even if cached in evaluation/results/cache_evaluation.json.")
    parser.add_argument("--no_plots", action="store_true", help="Disable automatic plot & HTML explorer generation.")
    args = parser.parse_args()
    
    # Cache GT datasets by path: gt_dir -> (pcs, features)
    gt_cache = {}
    
    # 1. Scan for evaluation folders containing OBJ files
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
    
    # Load cache if available
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache_file = os.path.join(RESULTS_DIR, "cache_evaluation.json")
    run_results = {}
    
    if os.path.exists(cache_file) and not args.force:
        try:
            with open(cache_file, "r") as f:
                run_results = json.load(f)
            log(f"Loaded {len(run_results)} previously cached run evaluation(s) from {cache_file}.")
        except Exception as e:
            log(f"Warning: Failed to load cache file {cache_file}: {e}")
            run_results = {}

    log("-" * 80)
    for idx, (root, rel_path, count) in enumerate(candidate_dirs, 1):
        status = " (CACHED)" if rel_path in run_results else ""
        log(f"  [{idx}/{len(candidate_dirs)}] {rel_path} ({count} meshes){status}")
    log("-" * 80 + "\n")

    total_start = time.time()
    evaluated_count = 0

    for idx, (root, rel_path, count) in enumerate(candidate_dirs, 1):
        target_gt_dir = resolve_gt_dir(root, default_gt_dir=args.gt_dir)
        
        if rel_path in run_results and not args.force:
            log(f"[{idx}/{len(candidate_dirs)}] Skipping cached '{rel_path}'")
            if not args.no_plots and target_gt_dir:
                generate_run_visualizations(root, target_gt_dir, force=args.force)
            continue

        if not target_gt_dir:
            log(f"[{idx}/{len(candidate_dirs)}] Skipping '{rel_path}': Could not resolve GT directory.")
            continue

        if target_gt_dir not in gt_cache:
            log(f"  [GT] Loading ground truth dataset from: {target_gt_dir}")
            pcs, feats = load_gt_data(target_gt_dir, num_points=args.points)
            gt_cache[target_gt_dir] = (pcs, feats)
        else:
            pcs, feats = gt_cache[target_gt_dir]

        gt_pcs, gt_features = pcs, feats
        if not gt_pcs:
            log(f"[{idx}/{len(candidate_dirs)}] Skipping '{rel_path}': No valid GT meshes in {target_gt_dir}.")
            continue

        log(f"[{idx}/{len(candidate_dirs)}] Evaluating '{rel_path}' ({count} meshes) against GT '{os.path.basename(target_gt_dir)}'...")
        run_start = time.time()
        metrics = evaluate_run(root, gt_pcs, gt_features, num_points=args.points, fscore_threshold=args.fscore_thresh)
        run_time = time.time() - run_start
        
        if metrics:
            run_results[rel_path] = metrics
            evaluated_count += 1
            log(f"  ✓ Finished '{rel_path}' in {run_time:.1f}s | CD_MMD: {metrics['CD_MMD']:.6f} | FScore: {metrics['FScore_MMD']:.4f} | 1NN_Total: {metrics['1NN_Total (%)']:.1f}% (Fake: {metrics['1NN_Fake (%)']:.1f}%, Real: {metrics['1NN_Real (%)']:.1f}%)\n")
            save_intermediate_results(run_results, cache_file=cache_file)
            
            if not args.no_plots:
                generate_run_visualizations(root, target_gt_dir, force=args.force)
                
            log(f"    [Saved live checkpoint to evaluation/results/evaluation_summary.md and .csv]")
        else:
            log(f"  ✗ Failed/Empty metrics for '{rel_path}' in {run_time:.1f}s\n")
            
    total_time = time.time() - total_start
    log(f"Evaluation complete for {len(run_results)} run(s) ({evaluated_count} new) in {total_time:.1f} seconds.")

    # 3. Final format & print results table
    df = pd.DataFrame.from_dict(run_results, orient='index')
    df = df.sort_values(by="CD_MMD", ascending=True)
    
    log("\n" + "="*80)
    log(" EVALUATION COMPARISON RESULTS (Sorted by CD MMD - Lower is Better)")
    log("="*80)
    log(df.to_string())
    log("="*80)

    csv_summary = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    md_summary = os.path.join(RESULTS_DIR, "evaluation_summary.md")
    log("\nResults successfully saved and up to date at:")
    log(f"  - {csv_summary}")
    log(f"  - {md_summary}")
    log(f"  - {cache_file}")
    if not args.no_plots:
        log(f"  - {RESULTS_DIR}/shape_space_actual_shapes_*.png & .pdf")
        log(f"  - {RESULTS_DIR}/shape_space_interactive_*.html")

if __name__ == '__main__':
    main()
