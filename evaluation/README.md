# Evaluation Pipeline — TetraDiffusion Ablation Studies

This folder contains a standalone, CPU-friendly evaluation pipeline to compute shape similarity and mesh quality metrics for your generated models compared to the ground-truth database.

## Metrics Used

The comparison evaluates the generated shape distribution against the ground-truth database using:
1. **CD_MMD (Chamfer Distance Minimum Modified Distance)**: Average distance from each generated mesh's point cloud to its nearest neighbor in the ground truth mesh set. **Lower is better.**
2. **FScore_MMD**: Average paired F-Score to the nearest ground-truth neighbor (measuring precision and recall of surface coverage). **Higher is better.**
3. **Sphericity**: Volume-to-surface compactness ratio ($36\pi V^2 / A^3$). Perfect sphere is `1.0`. **Ideal for lysosomes.**
4. **Connected_Components**: The average count of disconnected parts in each mesh. **Ideal is `1.0`.** Higher numbers indicate background noise or floaters (a key indicator if the background loss ablation worked!).
5. **Watertight_Ratio**: The fraction of meshes that are completely closed/watertight. **Higher is better.**
6. **Degenerate_Faces**: Fraction of faces with near-zero area. **Lower is better.**

---

## How to Run

After generating meshes via `inference.py` for one or more runs, do the following:

### 1. Run the comparison script
Run `compare.py` from within this `evaluation/` folder. It will scan your `runs/` directory, find all subfolders containing OBJ files, compare them to the lysosome GT dataset, and print a sorted comparison table.

```bash
python compare.py
```

### 2. View Summarized Outputs
The script automatically generates summary reports inside `evaluation/results/`:
* **`results/evaluation_summary.md`**: A clean, formatted markdown table ready to copy into reports or papers.
* **`results/evaluation_summary.csv`**: A spreadsheet of the metrics for programmatic analysis or plotting.

### Custom Options (Command Line Flags)

You can customize the directories and parameters by passing flags:
```bash
python compare.py \
    --runs_dir "../runs" \
    --gt_dir "../data_test/organelles/lyso" \
    --points 2048 \
    --fscore_thresh 0.02 \
    --filter "abl_"
```
* `--filter`: Pattern or prefix to match against the top-level run directory name. Matches are case-insensitive.
  * Passing `abl_` automatically matches any directory starting with `abl_` (translates to `abl_*`).
  * You can use wildcard characters, e.g., `*bio*` to only compare runs containing "bio".
* `--points`: Number of points sampled from each mesh surface for Chamfer/F-score comparison (default: 2048).
* `--fscore_thresh`: Threshold distance mapping precision and recall for F-Score (default: 0.02).
