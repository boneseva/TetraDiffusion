# Surface-Weighted SDF Background Loss

> **Implementation**: `lib/ops/BioConstraints.py` → `sdf_background_loss`
> **Config keys**: `diffusion.sdf_bg_loss_weight`, `diffusion.sdf_bg_threshold`
> **W&B metric**: `sdf_bg_loss`

---

## Problem: Multi-Component Mesh Generation

### What was observed
Certain organelle runs produce generated meshes with multiple **disconnected components** (e.g. a nucleus plus several ghost blobs floating nearby), even though:
- The training data was prepared with a single connected component per sample.
- Other organelle categories generate clean single-component meshes.
- The problem is stochastic — some generated samples from the same model are clean.

### Root cause

Mesh extraction in `Tetradata.py` (`get_mesh`) uses a **hard binary threshold**:

```python
sdf = torch.sign(sdf)   # → {-1, 0, +1}
```

Marching tets then finds every tetrahedral edge that crosses the `sdf = 0` boundary and extracts a surface triangle there. This means:

> **Any vertex where the predicted SDF is spuriously negative — however small the error, however far from the real organelle — becomes part of the extracted mesh.**

The diffusion model is trained with an MSE loss that is **uniform across all vertices**. The organelle interior (true `sdf < 0`) is a small fraction of the total vertex count — perhaps 5–15%. The vast majority of vertices are background (always `sdf > 0`). The MSE loss therefore provides a weak gradient signal for background correctness: a handful of spuriously negative predictions make almost no dent in the average loss, yet each one creates a real mesh component at extraction time.

### Why it is stochastic

The diffusion reverse process is stochastic. Different initial noise samples take different trajectories through the learned data manifold. Some trajectories produce a clean SDF field where every background vertex ends up confidently positive; others accumulate small sign errors that survive to the final denoised output. The probability of a bad trajectory is higher for:

- **Complex or non-convex organelles** (more surface area relative to volume → more borderline vertices).
- **Partially trained models** where the SDF prediction is noisier.
- **High offset noise** (`offset_noise > 0`) which shifts the mean of the noise distribution and can cause a DC bias in background SDF predictions.

---

## The Fix: Surface-Weighted Background Hinge Loss

### Mathematical formulation

For each vertex `i` in the tetrahedral graph, let:
- `φ_pred,i` = model's predicted normalised SDF (channel 0 of the recovered `x_start_pred`)
- `τ` = `sdf_bg_threshold` — normalised cutoff defining "background" (default `0.3`)

The **background weight** grows linearly with distance from the surface:

```
bg_weight_i = ReLU( φ_pred,i.detach() − τ )
```

This is **zero** for vertices near or inside the surface (`φ_pred < τ`) and grows linearly with distance for background vertices. Using `.detach()` means gradients only flow through the hinge term, not the weight.

The **hinge penalty** for spurious negativity:

```
hinge_i = ReLU( −φ_pred,i )
```

Zero when the prediction is non-negative (correct), penalises negative predictions proportionally.

The combined loss:

```
L_bg = mean_i( ReLU(φ_pred,i − τ) · ReLU(−φ_pred,i) )
```

Only background vertices that are *also* predicted negative contribute non-zero loss — exactly the scenario that produces ghost components.

### SNR gating

Like all biological constraint losses in this codebase, `L_bg` is multiplied by the soft SNR weight:

```
w(t) = SNR(t) / (SNR(t) + 1)
```

This ensures the background penalty is only active when the model's predicted `x_start` is already a reasonable estimate (low noise levels, late in the denoising chain). At high noise the prediction is dominated by noise and the background signal would be uninformative.

### Total training loss

```
L_total = L_diffusion  +  λ_bg · w(t) · L_bg   [+  λ_bio · w(t) · L_bio  if enabled]
```

The background loss is **independent** of `bio_loss_weight` — it can be used alone (`bio_loss_weight: 0`, `sdf_bg_loss_weight: 0.05`) or combined with the existing curvature / Laplacian losses.

---

## Configuration

In `config/config.yaml` (or as a CLI override for ablations):

```yaml
diffusion:
  sdf_bg_loss_weight: 0.05   # λ_bg — set to 0 to disable (default)
  sdf_bg_threshold: 0.3      # τ — normalised SDF cutoff for "background"
```

### Parameter guide

| Parameter | Typical range | Effect |
|---|---|---|
| `sdf_bg_loss_weight` | 0 → 0.01 (gentle) → 0.1 (strong) | Overall weight λ_bg. Start at `0.05`. Increase if ghost components persist. |
| `sdf_bg_threshold` | 0.2 → 0.5 | Defines how far from the surface "background" starts. Lower = tighter band, more vertices exempt. |

> **Note on threshold units**: `τ` is in the **normalised** SDF space (after min-max normalisation to `[0, 1]`), not physical voxel units. `τ = 0.3` means 30% of the full `sdfs_max − sdfs_min` range away from the organelle surface.

---

## W&B Monitoring

When `sdf_bg_loss_weight > 0`, the following metric is logged per step:

| Metric | Description |
|---|---|
| `sdf_bg_loss` | Raw (unweighted, ungated) background hinge loss |
| `diffusion_loss` | Pure MSE diffusion component |
| `bio_loss` | Laplacian + curvature losses (if `bio_loss_weight > 0`) |

A healthy run should show `sdf_bg_loss` **decreasing over training**. If it plateaus at a non-zero value, increase `sdf_bg_loss_weight` or lower `sdf_bg_threshold`.

---

## Ablation Study Design

The loss is designed for systematic ablation. Suggested conditions:

| Run name | `bio_loss_weight` | `sdf_bg_loss_weight` | `sdf_bg_threshold` | Purpose |
|---|---|---|---|---|
| `baseline` | `0` | `0` | — | No geometric regularisation |
| `bio_only` | `0.005` | `0` | — | Curvature + Laplacian only |
| `bg_w005_t03` | `0` | `0.05` | `0.3` | Background loss only, default params |
| `bg_w01_t03` | `0` | `0.1` | `0.3` | Stronger weight |
| `bg_w005_t02` | `0` | `0.05` | `0.2` | Tighter surface band |
| `bg_w005_t05` | `0` | `0.05` | `0.5` | Wider surface band |
| `combined` | `0.005` | `0.05` | `0.3` | Bio + background (full regularisation) |

**Primary metric**: fraction of generated meshes with exactly 1 connected component (computed post-hoc with trimesh on the OBJ outputs).

**Secondary metrics**: FID / coverage / MMD on surface point clouds, `sdf_bg_loss` curve in W&B.

### CLI commands for ablation

```bash
# Baseline (no regularisation)
accelerate launch main.py --data_path /data --grid_res 128 --name baseline

# Background loss only, default params
accelerate launch main.py --data_path /data --grid_res 128 --name bg_w005_t03 \
  --diffusion.sdf_bg_loss_weight 0.05 \
  --diffusion.sdf_bg_threshold 0.3

# Stronger weight
accelerate launch main.py --data_path /data --grid_res 128 --name bg_w01_t03 \
  --diffusion.sdf_bg_loss_weight 0.1 \
  --diffusion.sdf_bg_threshold 0.3

# Combined with bio loss
accelerate launch main.py --data_path /data --grid_res 128 --name combined \
  --diffusion.bio_loss_weight 0.005 \
  --diffusion.sdf_bg_loss_weight 0.05
```

---

## Implementation Notes

- `sdf_background_loss` is in `lib/ops/BioConstraints.py`.
- It is called inside `GaussianDiffusion.p_losses` in `lib/DDPM.py`, **separately** from the `organelle_loss_registry` path, so it can be logged and weighted independently from the curvature/Laplacian bio losses.
- The weight uses `sdf_pred.detach()` to avoid double-counting the gradient — only `ReLU(-φ_pred)` receives gradients.
- The `_any_bio_active` flag in `GaussianDiffusion.__init__` ensures the `x_start_pred` recovery block is entered when *either* `bio_loss_weight > 0` **or** `sdf_bg_loss_weight > 0`, so the two loss families can be combined or used independently.
- **Backward compatible**: `sdf_bg_loss_weight` defaults to `0.0` — the loss is a strict no-op in all existing configs unless explicitly enabled.
- The loss does **not** affect inference — it is training-only.
