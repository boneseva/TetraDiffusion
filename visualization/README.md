# TetraDiffusion — Visualization & Diagnostic Dashboard

A **no-code Gradio interface** for structural biologists and paper reviewers to interactively test shape variations, evaluate exemplar style matching, and inspect geometric distributions — without writing a single line of Python.

---

## Quick Start

### 1. Install visualization dependencies

```bash
# From the repo root:
pip install -r visualization/requirements.txt
```

### 2. Launch (demo mode — no model required)

```bash
python visualization/app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser. The app
will immediately generate a synthetic organelle mesh so you can explore the
interface.

### 3. Launch with a trained model

```bash
python visualization/app.py --config_path results/<run_name>
```

The run directory must contain at least:
- `config.yaml`
- `ds.pth`
- at least one `model-*.pt` checkpoint

### 4. Full launch options

```
--config_path PATH    Path to trained run directory (omit for demo mode)
--cuda_device INT     CUDA device index (default: 0)
--server_port INT     Gradio server port (default: 7860)
--share               Create a public Gradio share link (for remote reviewers)
```

---

## Interface Overview

| Column | Purpose |
|--------|---------|
| **1 · Control Desk** | Upload exemplar, choose organelle profile, tune CFG & λ_bio sliders |
| **2 · Visual Verification Deck** | 2D EM-simulation projection + interactive 3D WebGL mesh viewport |
| **3 · Morphological Metrics** | Surface curvature histogram vs. reference, S/V ratio scorecard |

### Exemplar-driven mode

Upload a preprocessed `.pt` file (any `sample_*.pt` from your
`preprocessed_data/` directory) to:
1. Auto-compute target curvature statistics from the exemplar.
2. Inject an `ExemplarLossProfile` into the global `organelle_loss_registry`
   under the key `"exemplar-driven"`.
3. Switch the **Organelle Profile Target** dropdown to `exemplar-driven`.

The curvature histogram panel will then show a side-by-side comparison of the
generated shape vs. the exemplar reference.

---

## Architecture Notes

The app degrades gracefully in **three tiers**:

| Tier | Condition | Behaviour |
|------|-----------|-----------|
| **Full model** | `--config_path` supplied, all deps present | Real `GaussianDiffusion.p_sample_loop()` |
| **Demo / synthetic** | No config, or model load failed | Signed-distance ellipsoid + `scikit-image` marching cubes |
| **Exemplar overlay** | `.pt` file uploaded | Curvature stats extracted; histogram comparison activated |

---

## File Layout

```
visualization/
├── app.py            ← Main Gradio dashboard (this file)
├── requirements.txt  ← Extra pip dependencies
└── README.md         ← This document
```
