# Ablation Studies — TetraDiffusion

Everything runs via `submit_train.sh`.  
The general pattern is always:

```bash
sbatch submit_train.sh --category <Category> --name <descriptive_run_name> [flags]
```

Results land in `runs/<name>/`, checkpoints are `model-0.pt` / `model-1.pt` (alternating), and
all metrics are logged to the **TetraDiffusion** WandB project under the run name.

---

## 1  Biological Constraint Loss  *(dissertation section 2.2 / 3.1)*

The core contribution.  The biological loss adds a Laplacian smoothness term and/or a
Helfrich bending-energy proxy to the diffusion objective, weighted by `SNR(t)/(SNR(t)+1)`
so it only matters at low noise levels.

Default config: `bio_loss_weight=0.005`, `bio_loss_type=both`, `bio_snr_weighting=True`.

### 1a  With vs Without bio loss  (primary ablation)

```bash
# --- Mitochondria ---
sbatch submit_train.sh --category Mitochondria --name mito_bio_on
sbatch submit_train.sh --category Mitochondria --name mito_bio_off   --no_bio_loss

# --- Golgi ---
sbatch submit_train.sh --category Golgi --name golgi_bio_on
sbatch submit_train.sh --category Golgi --name golgi_bio_off          --no_bio_loss

# --- ER ---
sbatch submit_train.sh --category ER --name er_bio_on
sbatch submit_train.sh --category ER --name er_bio_off                --no_bio_loss
```

### 1b  Loss type decomposition

Which term matters more — smoothness (Laplacian) or bending energy (curvature)?

```bash
sbatch submit_train.sh --category Mitochondria --name mito_bio_both        --bio_loss_type both
sbatch submit_train.sh --category Mitochondria --name mito_bio_laplacian   --bio_loss_type laplacian
sbatch submit_train.sh --category Mitochondria --name mito_bio_curvature   --bio_loss_type curvature
```

### 1c  Loss weight sensitivity

How sensitive are results to the weight λ?

```bash
sbatch submit_train.sh --category Mitochondria --name mito_bio_w1e-3   --bio_loss_weight 0.001
sbatch submit_train.sh --category Mitochondria --name mito_bio_w5e-3   --bio_loss_weight 0.005   # default
sbatch submit_train.sh --category Mitochondria --name mito_bio_w1e-2   --bio_loss_weight 0.01
sbatch submit_train.sh --category Mitochondria --name mito_bio_w5e-2   --bio_loss_weight 0.05
```

### 1d  SNR weighting on/off

Without SNR weighting the bio loss is applied uniformly at all noise levels.

```bash
sbatch submit_train.sh --category Mitochondria --name mito_bio_snrw_on   # default
sbatch submit_train.sh --category Mitochondria --name mito_bio_snrw_off  --no_snr_weighting
```

---

## 2  LR Schedule

Does the warmup actually help, and does cosine decay hurt?

```bash
sbatch submit_train.sh --category Golgi --name golgi_lr_warmup_const   --lr_schedule warmup_constant  # default
sbatch submit_train.sh --category Golgi --name golgi_lr_constant        --lr_schedule constant
sbatch submit_train.sh --category Golgi --name golgi_lr_warmup_cos      --lr_schedule warmup_cosine
sbatch submit_train.sh --category Golgi --name golgi_lr_cosine          --lr_schedule cosine
```

Tip: run these with `--no_bio_loss` to isolate LR from bio-loss effects:

```bash
sbatch submit_train.sh --category Golgi --name golgi_lr_warmup_const_plain --lr_schedule warmup_constant --no_bio_loss
sbatch submit_train.sh --category Golgi --name golgi_lr_cosine_plain        --lr_schedule cosine          --no_bio_loss
```

---

## 3  Offset Noise

Offset noise prevents mean-collapse on small datasets (currently 0.1).
Edit `offset_noise` in `config/config.yaml` before submitting:

```yaml
# config/config.yaml  →  diffusion section
offset_noise: 0.0   # baseline — original DDPM, no offset noise
offset_noise: 0.1   # default
offset_noise: 0.2   # stronger
```

```bash
sbatch submit_train.sh --category Mitochondria --name mito_offset0    # set offset_noise: 0.0 first
sbatch submit_train.sh --category Mitochondria --name mito_offset01   # default (0.1)
sbatch submit_train.sh --category Mitochondria --name mito_offset02   # set offset_noise: 0.2 first
```

---

## 4  Cross-Category (Joint) Training

> ⚠️ **Important — read before running these experiments.**
>
> The model is **unconditional** — there is no class label input anywhere in the
> architecture (UVIT only takes `x` and `log_snr`).  
> A jointly trained model therefore samples from the **entire learned mixture
> distribution**.  You cannot say "generate a mitochondrion" — inference will
> produce whichever organelle type the sampling happens to land on.
>
> Adding class conditioning would require a new `class_embedding` layer in UVIT
> and **retraining from scratch** (existing checkpoints would be incompatible).
>
> **What this experiment CAN answer:**  
> *Does shared representation learning across organelle types improve generation
> quality for each type, compared to training each type separately?*
>
> **How to evaluate it:**  
> Train jointly, then generate a large batch (e.g. 100 samples).  Because the
> organelles look morphologically very different (ER = tubular, mitochondria =
> oval, Golgi = stacked cisternae), you can sort/filter the output manually or
> with a simple shape classifier.  Compare Chamfer distance / F-score of the
> per-type subsets against the single-category baselines.

Use `--categories` (plural) to pass a space-separated list.

```bash
# Single-category baselines (needed for comparison)
sbatch submit_train.sh --category Mitochondria --name single_mito
sbatch submit_train.sh --category Golgi        --name single_golgi
sbatch submit_train.sh --category ER           --name single_er
sbatch submit_train.sh --category Lysosome     --name single_lyso

# Jointly trained models (unconditional — output organelle type is random)
sbatch submit_train.sh --categories Mitochondria Golgi   --name joint_mito_golgi
sbatch submit_train.sh --categories ER Lysosome          --name joint_er_lyso
sbatch submit_train.sh --categories Mitochondria Golgi ER Lysosome --name joint_all
```

> Each combination gets its own preprocessed-data cache keyed by the sorted
> category list, so the first run of a new combination is slower; re-runs are fast.

---

## 5  Full Factorial: Bio-loss × Category  *(recommended for paper table)*

Run all four organelles with and without bio loss for the main results table.

```bash
for CAT in Mitochondria Golgi ER Lysosome; do
    sbatch submit_train.sh --category $CAT --name ${CAT,,}_bio_on
    sbatch submit_train.sh --category $CAT --name ${CAT,,}_bio_off --no_bio_loss
done
```

---

## 6  Computational Overhead  *(cost-justification table)*

The core claim is that the biological prior comes at essentially zero cost.  This
section provides the three numbers needed to back that up in a single compact table:
**ms/step overhead**, **peak VRAM delta**, and **FLOP%**.

### 6a  Wall-clock time per step

The Trainer already logs `steps_per_sec` to WandB for every run.  No new runs are
needed — read the metric from matching `bio_on` / `bio_off` runs on the same organelle,
batch size, and GPU.  Report mean ± std over the last 10 k steps.

Expected result: <5% overhead.  The bio path is two O(N·K) scatter/gather operations;
the dominant cost is the UVIT forward pass.

### 6b  Peak GPU memory

Add a one-off profiling script (`profile_bio_cost.py`) that:
1. Loads a real preprocessed batch from disk.
2. Calls `p_losses` 100× with `bio_loss_weight=0` → record
   `torch.cuda.max_memory_allocated()`.
3. Repeats with `bio_loss_weight=0.005` → record delta.

```python
# profile_bio_cost.py  (sketch)
import torch
from lib.DDPM import GaussianDiffusion

model = ...                       # load from checkpoint
batch = torch.load("sample_0.pt").cuda().unsqueeze(0)

torch.cuda.reset_peak_memory_stats()
for _ in range(100):
    loss = model.p_losses(batch)   # bio OFF
base_mem = torch.cuda.max_memory_allocated() / 1e9

torch.cuda.reset_peak_memory_stats()
for _ in range(100):
    loss = model.p_losses(batch)   # bio ON
bio_mem  = torch.cuda.max_memory_allocated() / 1e9

print(f"Delta VRAM: {bio_mem - base_mem:.3f} GB")
```

### 6c  FLOPs breakdown

Use `torch.profiler` (or `fvcore.nn.FlopCountAnalysis`) on a single forward pass
with and without the bio path.  Report bio FLOPs as a fraction of total step FLOPs.

Expected result: bio loss ≈ 1–3% of total (dominated by UVIT attention ops).

---

## 7  Curvature Kernel Width (σ) Sensitivity

`bio_curvature_softness` (σ) controls how tightly `L_curv` is concentrated near the
zero-level set via `exp(-|φ|/σ)`.  Currently config-only; edit `config/config.yaml`
before each run.

> **TODO:** add a `--bio_curvature_softness` CLI flag to `main.py` to avoid editing YAML.

```bash
# Set bio_curvature_softness in config/config.yaml, then:
sbatch submit_train.sh --category Mitochondria --name mito_sigma005  --bio_loss_type curvature  # σ=0.05
sbatch submit_train.sh --category Mitochondria --name mito_sigma010  --bio_loss_type curvature  # σ=0.10
sbatch submit_train.sh --category Mitochondria --name mito_sigma015  --bio_loss_type curvature  # σ=0.15 (default)
sbatch submit_train.sh --category Mitochondria --name mito_sigma020  --bio_loss_type curvature  # σ=0.20
sbatch submit_train.sh --category Mitochondria --name mito_sigma030  --bio_loss_type curvature  # σ=0.30
```

Expected result: too-small σ makes the loss noisy (near-zero weight on most vertices);
too-large σ dilutes the surface signal and degrades toward a uniform smoothness penalty.
Performance should peak near the default σ=0.15, validating the hyperparameter choice.

---

## 8  Sample Efficiency

Does the bio prior provide larger gains when training data is scarce?
This is the single most important argument for the biological domain (EM datasets
are inherently small).

Add a `--train_fraction` flag to `main.py` that sub-samples the training CSV before
creating `MeshLoader`.  Run at 25%, 50%, 75%, 100% of each organelle's training
split, both `bio_on` and `bio_off`.  Use **at least 3 random seeds** per condition
and report mean ± std to account for high variance on small splits.

```bash
for FRAC in 0.25 0.50 0.75 1.00; do
    sbatch submit_train.sh --category Mitochondria --name mito_frac${FRAC}_bio_on  \
        --train_fraction $FRAC
    sbatch submit_train.sh --category Mitochondria --name mito_frac${FRAC}_bio_off \
        --train_fraction $FRAC --no_bio_loss
done
```

Expected result: at 25% data the bio loss should produce a larger *relative* improvement
in MMD-CD and LR than at 100% — the prior fills in the information gap.
Plotting improvement (bio_on − bio_off) as a function of training set size
makes a compelling figure.

---

## 9  Training Dynamics Analysis

All of these analyses use **existing WandB logs** from `bio_on` / `bio_off` runs —
no new training runs are required unless explicitly noted.

### 9a  Loss curve overlay

Plot `diffusion_loss` and `bio_loss` on the same axis over training steps.
Does `bio_loss` stabilise before `diffusion_loss`?  Is there a phase where the two
terms pull in opposite directions?

### 9b  Convergence speed

Define a threshold (e.g. 90% of final MMD-CD) and measure steps-to-threshold for
`bio_on` vs `bio_off`.  Requires periodic evaluation checkpoints — lower `test_every`
for a short diagnostic run if not already available.

### 9c  Gradient magnitude decomposition

Add logging of the bio-component gradient norm vs total gradient norm to `p_losses`:

```python
# In DDPM.p_losses, after computing the combined loss:
bio_grad = torch.autograd.grad(bio_term, model_out, retain_graph=True)[0].norm()
accelerator.log({"bio_grad_norm": bio_grad, "total_grad_norm": total_grad_norm})
```

Report `bio_grad_norm / total_grad_norm` over training.  Expected: <5% at high noise
(SNR weighting suppresses it) rising to ~20–30% near t≈0.

### 9d  SNR-conditioned bio loss profile

Log `bio_loss` bucketed by noise level `t` (10 bins) to confirm SNR weighting works
as intended: bio loss contribution should be near-zero at high t and peak near t≈0.
This empirically validates the design rationale for SNR weighting.

---

## 10  Inference Step Count × Bio Loss Interaction

Does a bio-constrained model degrade more gracefully as inference steps decrease?
If the model has internalised a smoothness prior, it may partially substitute for
denoising steps — enabling a faster inference budget.

Use the existing `sampling_steps` support in `DDPM.py` to sample from both
`bio_on` and `bio_off` checkpoints at multiple step counts:

```bash
# After training completes, run inference at 16 / 32 / 64 / 128 steps:
python inference.py --config_path runs/mito_bio_on   --sampling_steps 16
python inference.py --config_path runs/mito_bio_on   --sampling_steps 32
python inference.py --config_path runs/mito_bio_on   --sampling_steps 64
python inference.py --config_path runs/mito_bio_on   --sampling_steps 128
python inference.py --config_path runs/mito_bio_off  --sampling_steps 16
python inference.py --config_path runs/mito_bio_off  --sampling_steps 32
python inference.py --config_path runs/mito_bio_off  --sampling_steps 64
python inference.py --config_path runs/mito_bio_off  --sampling_steps 128
```

Report LR, SCE, and MMD-CD at each step count.  Plot as a 2×4 grid (bio_on/off × step count).

---

## 11  Morphological Distribution Analysis

Chamfer distance captures reconstruction accuracy but not biological plausibility of
the *full shape distribution*.  Compute the following on generated and GT mesh sets
using `trimesh`:

| Metric | Biological relevance | Expected bio_on improvement |
|---|---|---|
| Volume distribution (KL div, Wasserstein-1) | Organelle size control | Tighter, better-calibrated |
| Surface area distribution | Membrane area, scales with function | Closer to GT |
| Aspect ratio (bbox eigenvalue ratios) | Elongation — mito vs lyso | Reduced spread |
| Sphericity `36π·V²/A³` | Compactness — lysosome ≈ 1.0 | Higher for lysosome |
| Mean absolute curvature histogram | Directly reflects L_curv target | Lower tails |

> Run Priority 6 on **Lysosome** as a sanity check in addition to Mitochondria:
> lysosomes should be near-spherical, giving a clear ground-truth expectation for
> sphericity improvement.

Compute statistics across all generated samples per run and compare distributions
visually (violin plots or histograms) as well as numerically (KL / Wasserstein).

---

## 12  Topology Quality Metrics

Chamfer distance is blind to degenerate geometry.  Count the following from each
generated OBJ using `trimesh`:

- Watertightness: `mesh.is_watertight` (boolean)
- Fraction of near-zero-area faces: `face_areas < 1e-6`
- Self-intersections: `trimesh.repair.broken_faces`

Expected: bio loss reduces degenerate triangles by penalising high-curvature regions.
Report as a table of mean rates across 50 generated samples per run.

---

## 13  Inference-Time Guidance vs Training-Time Constraint

A conceptually important comparison: does the bio prior need to be *learned* (training-
time), or can it be injected at inference without any retraining via classifier-like
guidance?

Implement an inference-time version by adding the bio loss gradient as a correction at
each DDPM reverse step (akin to classifier guidance / manifold-constrained diffusion,
Chung et al. 2022):

```
x_{t-1} ← DDPM_step(x_t) − η · ∇_{x_t} bio_loss(x_start_pred(x_t))
```

Compare three conditions:
- **Training only** (current method)
- **Inference only** (bio loss gradient at each step, vanilla trained model)
- **Both** (trained with bio loss + inference guidance)

Expected: training-time is more effective (the model learns a smooth manifold);
inference-time alone is noisier but requires no retraining.  Directly addresses the
comparison with Chung et al. 2022 cited in the paper.

---

## Resuming a Run

If a job is interrupted add `--resume` together with the original `--name`:

```bash
sbatch submit_train.sh --category Mitochondria --name mito_bio_on --resume
```

The trainer will load the latest checkpoint and continue the same WandB run.

---

## Evaluation

After training, generate samples with `submit_inference.sh`:

```bash
sbatch submit_inference.sh --config_path runs/mito_bio_on
sbatch submit_inference.sh --config_path runs/mito_bio_off
```

OBJ files are saved to `runs/<name>/` and can be compared visually or with mesh-quality metrics
(Chamfer distance, F-score, etc.) using the scripts in `preprocessing/`.

---

## Quick Reference — All CLI Flags

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--category` | any organelle name | Golgi | single category |
| `--categories` | space-separated list | — | **multi-organelle joint training** |
| `--name` | string | auto-generated | run dir + WandB display name |
| `--no_bio_loss` | flag | off | sets `bio_loss_weight=0` |
| `--bio_loss_weight` | float | 0.005 | λ for bio constraint term |
| `--bio_loss_type` | laplacian / curvature / both | both | which bio term(s) to use |
| `--no_snr_weighting` | flag | off | apply bio loss at ALL noise levels |
| `--lr_schedule` | warmup_constant / constant / warmup_cosine / cosine | warmup_constant | LR schedule |
| `--num_steps` | int | 400000 | total training steps |
| `--batch_size` | int | 4 | per-GPU batch size |
| `--resume` | flag | off | continue from latest checkpoint |
| `--train_fraction` | float 0–1 | 1.0 | sub-sample training set (sample efficiency ablation) *(to be implemented)* |

Config keys that still require editing `config/config.yaml` directly:

| Key | Section | Default | Notes |
|---|---|---|---|
| `bio_curvature_softness` | diffusion | 0.15 | surface kernel width σ — add `--bio_curvature_softness` CLI flag to avoid YAML edits |
| `offset_noise` | diffusion | 0.1 | offset noise strength |
| `warmup_steps` | training | 2000 | warmup length in steps |
| `lr_min` | training | 1e-6 | cosine schedule floor |

---

## Priority Summary

| # | Study | Effort | Impact | New runs? |
|---|---|---|---|---|
| 1a/5 | Bio ON vs OFF × all categories | High | ★★★★★ | Yes (8 runs) |
| 6 | Computational overhead | Low | ★★★★☆ | No (profile existing) |
| 1b | Loss decomposition (smooth vs curv) | Medium | ★★★★☆ | Yes (2 runs) |
| 1c | Weight sensitivity | Medium | ★★★☆☆ | Yes (3 runs) |
| 1d | SNR weighting on/off | Low | ★★★★☆ | Yes (1 run) |
| 7 | σ sensitivity | Medium | ★★★☆☆ | Yes (4 runs) |
| 8 | Sample efficiency | High | ★★★★★ | Yes (8+ runs) |
| 9 | Training dynamics | Low | ★★★☆☆ | No (from logs) |
| 10 | Inference step × bio | Low | ★★★☆☆ | No (post-training) |
| 11 | Morphological distributions | Medium | ★★★★☆ | No (post-training) |
| 12 | Topology quality | Low | ★★☆☆☆ | No (post-training) |
| 13 | Inference-time guidance | High | ★★★☆☆ | Partial |
