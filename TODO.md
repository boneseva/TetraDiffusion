# TODO — Reviewer Response Checklist

Derived from the reviewer comments on _Biologically-Informed Tetrahedral Diffusion for 3D Organelle Generation_.
Paper changes have already been applied to `paper/main.tex`. The items below are the **code / experiment / verification** tasks that still need to be done to back those claims.

---

## 1. Method / Code

- [ ] **On-the-fly augmentation in the dataloader**
  `lib/Tetradata.py` — add random 90° rotations (around all three axes) and axis reflections applied per batch during training. The paper (§4 step 2) now claims this is implemented. Verify it exists or add it.
  - Augmentation should be applied _after_ loading cached `.pt` samples, _before_ passing to the model.
  - Do **not** cache augmented variants; apply on-the-fly so each epoch sees different orientations.

- [ ] **Hard-threshold SNR gating variants in `lib/DDPM.py` / `lib/Trainer.py`**
  Add a config flag `snr_gate: str = "soft"` with options `"soft"` | `"hard_0.3"` | `"hard_0.5"` | `"hard_0.7"` | `"none"`.
  - `"soft"` → existing `SNR(t) / (SNR(t) + 1)` (default, unchanged)
  - `"hard_X"` → `1.0 if t < X else 0.0`
  - `"none"` → constant `1.0`
  - This allows the §7.3 SNR ablation runs to be launched without any code changes at experiment time.

- [ ] **1-NNA-CD validation monitoring**
  Add periodic evaluation (every N steps, configurable) of 1-NNA-CD on a held-out validation split.
  - Needed to confirm the model is not memorising the training set (claimed in §6).
  - Log to W&B as `val/1nna_cd`.

---

## 2. Experiments to Run

### 2a. Curvature correlation verification (§5.1)
- [ ] Write a short script (≈30 lines, `scripts/verify_laplacian_correlation.py`) that:
  1. Loads 50 held-out preprocessed `.pt` samples for each organelle class.
  2. Computes the **uniform graph Laplacian** of the SDF channel using the tetrahedral adjacency.
  3. Extracts the triangular surface mesh via marching tetrahedra and computes **cotangent-weight mean curvature** using `trimesh` or `igl`.
  4. Reports the **Pearson r** between the two at surface-adjacent vertices.
- [ ] Verify that `r > 0.92` for all four classes. If not, update the claim in §5.1 to match the actual numbers.

### 2b. SNR gating ablation (§7.3)
- [ ] Train **5 models** on Mitochondria, differing only in `snr_gate`:
  - `none` (no gating)
  - `soft` (default — already done if Bio+ run exists)
  - `hard_0.3`
  - `hard_0.5`
  - `hard_0.7`
- [ ] Evaluate all five on MMD-CD, COV-CD, 1-NNA-CD, LR, SCE.
- [ ] Produce a bucketed-by-noise-level plot of bio-loss magnitude for the `soft` run to visually confirm suppression at high `t`.

### 2c. Full Bio+ / Bio− comparison (§8, all organelle classes)
- [ ] Run Bio+ and Bio− training on all four classes: **Mitochondria, Lysosome, ER, Golgi**.
- [ ] Evaluate with MMD-CD, COV-CD, 1-NNA-CD, LR, SCE.
- [ ] Report results in a table in the paper.

### 2d. Loss component decomposition (§7.1)
- [ ] Train **4 models** on Mitochondria:
  - Bio− (baseline, no bio loss)
  - smooth only (`L_smooth`, no `L_curv`)
  - curv only (`L_curv`, no `L_smooth`)
  - Bio+ (both, default)
- [ ] Evaluate all four on all five metrics.

### 2e. Constraint weight sweep (§7.2)
- [ ] Train 4 models on Mitochondria with `lambda` ∈ {0.001, 0.005, 0.010, 0.050}.
- [ ] Evaluate on all five metrics to confirm 0.005 is optimal.

### 2f. Curvature kernel width sweep (§7.4)
- [ ] Train 5 models on Mitochondria with `sigma` ∈ {0.05, 0.10, 0.15, 0.20, 0.30} (`L_curv` only).
- [ ] Evaluate on LR, SCE, MMD-CD, COV-CD.

### 2g. Computational overhead measurement (§7.5)
- [ ] Profile 100 forward passes with Bio+ and Bio− on a single GPU.
- [ ] Record: wall-clock time per step (`steps_per_sec` from W&B), peak GPU memory (`torch.cuda.max_memory_allocated()`), FLOPs (`torch.profiler`).
- [ ] Confirm overhead < 5% on all three axes (or update the claim).

### 2h. Sample efficiency ablation (§7.6)
- [ ] Train Bio+ and Bio− on {25%, 50%, 75%, 100%} of the Mitochondria training split, **3 seeds each** → 24 runs total.
- [ ] Plot improvement gap (ΔMMD-CD, ΔLR) vs. training set fraction.

### 2i. Morphological distributions (§7.7)
- [ ] Compute volume, surface area, aspect ratio, sphericity, mean absolute curvature on generated and GT meshes using `trimesh`.
- [ ] Compare distributions via Wasserstein-1 distance.
- [ ] Lysosome sphericity is the headline sanity check.

### 2j. Inference-time guidance vs. training-time constraint (§7.8)
- [ ] Implement inference-time bio-loss guidance in `inference.py`:
  `x_{t-1} ← DDPM_step(x_t) − η ∇_{x_t} L_bio(X̂_0(x_t))`
- [ ] Evaluate 3 conditions: training-time only, inference-time only (on Bio− checkpoint), both combined.

---

## 3. Paper / Writing

All the following are **already done** in `paper/main.tex` but depend on the experimental results above to be filled in with actual numbers:

- [ ] **§5.1** — replace `r > 0.92` placeholder with actual measured Pearson r values per class (from task 2a).
- [ ] **§5.4** — replace "within 2% on LR and SCE" with actual measured numbers (from task 2b).
- [ ] **§7.3** — fill in the SNR ablation table (from task 2b).
- [ ] **§8** — fill in the main Bio+/Bio− results table across all four classes (from task 2c).
- [ ] **§7.1–7.8** — fill in all ablation tables (from tasks 2d–2j).
- [ ] **§7.5** — fill in actual overhead percentages (from task 2g).
- [ ] Check whether the on-the-fly augmentation in §4 step 2 ("tripling the effective dataset size") is accurate given the actual number of rotation/reflection symmetries used. Adjust wording if needed.

---

## 4. Infrastructure / Misc

- [ ] Add `scripts/verify_laplacian_correlation.py` (see task 2a).
- [ ] Add `scripts/eval_metrics.py` (or confirm existing evaluation script covers MMD-CD, COV-CD, 1-NNA-CD, LR, SCE, Wasserstein morphology).
- [ ] Confirm `inference.py` supports the `--snr_gate` flag once added to the config.
- [ ] Set `WANDB_MODE=offline` in CI / local runs to avoid accidental W&B uploads during testing.

