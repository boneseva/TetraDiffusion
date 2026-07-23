# 🚀 TetraDiffusion — End-to-End Pipeline & Visual Studio Guide

This directory (`pipeline/`) provides a unified, intuitive workflow for managing the complete lifecycle of 3D organelle diffusion models—from raw segmentation masks/OBJ inputs to preprocessed DMTet grids, CSV dataset registration, multi-GPU SLURM training, mesh inference sampling, shape-space evaluation, and interactive 3D web visualization.

---

## 🏗 Pipeline Architecture Overview

```text
┌─────────────────────────┐
│  1. Raw Data Ingestion  │ ──► Converts flat/raw OBJs into organized subfolders
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 2. DMTet Preprocessing  │ ──► Fits 128³ tetrahedral grid SDFs (sample.pth)
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  3. CSV Registration    │ ──► Registers preprocessed samples in lib/all_urocell.csv
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 4. Production Training  │ ──► Trains 400k-step U-ViT diffusion models with bio-losses
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  5. 3D Mesh Inference   │ ──► Samples novel 3D organelle OBJ meshes from checkpoints
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  6. Evaluation & Studio │ ──► Computes metrics (Chamfer/EMD) & interactive WebGL dashboard
└─────────────────────────┘
```

---

## ⚡ Quick Start: 2 Ways to Run

### Method 1: The Unified CLI (`python pipeline/cli.py` or `bash pipeline/run.sh`)

#### 1. Check Full Pipeline & Dataset Status
Print an instant matrix overview of raw files, preprocessed samples, CSV registrations, checkpoints, and generated meshes:
```bash
python pipeline/cli.py status
```

#### 2. Register Preprocessed Samples into CSV
Audit `data_urocell/preprocessed/` and automatically append newly finished samples to `lib/all_urocell.csv`:
```bash
python pipeline/cli.py register --dataset urocell --category fv
```

#### 3. Launch or Resume Production Training
Launch 400,000-step training runs with biological constraint losses and offset noise:
```bash
# Preview sbatch commands (dry run):
python pipeline/cli.py train --dataset urocell --dry_run

# Submit to SLURM:
python pipeline/cli.py train --dataset urocell --category fv

# Resume training from latest checkpoint:
python pipeline/cli.py train --dataset urocell --resume
```

#### 4. Generate 3D Meshes (Inference)
Sample 3D meshes from trained model checkpoints:
```bash
python pipeline/cli.py infer --run_name urocell_fv_final_prod --num_images 8
```

#### 5. Run Evaluation & Generate Interactive Shape-Space Plots
Compute morphological metrics (Chamfer distance, EMD, sphericity, curvature stats) and generate interactive HTML scatter plots:
```bash
python pipeline/cli.py evaluate --dataset urocell
```

---

### Method 2: The Visual Web Studio (`pipeline/dashboard.py`)

Launch the visual web dashboard to interactively view 3D meshes, trigger pipeline actions, and explore shape-space plots right in your browser:

```bash
python pipeline/cli.py dashboard
# OR
python pipeline/dashboard.py --port 7860
```
Open **[http://localhost:7860](http://localhost:7860)** in your browser.

#### Features inside the Web Studio:
- **📊 Health & Status Matrix**: Live progress tracking across OpenOrganelle and UroCell categories.
- **🧊 3D WebGL Inspector**: Rotate, zoom, and inspect raw, preprocessed, or generated 3D organelle OBJ meshes interactively.
- **🚀 Action Control Desk**: Trigger dataset registration, dry-run training launches, inference, and shape-space plot creation via buttons.
- **🌐 Interactive Shape Space Plot**: Inspect PCA/t-SNE morphological feature distribution plots directly in the dashboard.

---

## 📁 File Structure in `pipeline/`

| File | Purpose |
| :--- | :--- |
| [cli.py](file:///c:/Users/evabo/Documents/Projekti/3D%20Generation/Repos/TetraDiffusion/pipeline/cli.py) | Master command-line orchestrator handling all 6 pipeline stages |
| [dashboard.py](file:///c:/Users/evabo/Documents/Projekti/3D%20Generation/Repos/TetraDiffusion/pipeline/dashboard.py) | Interactive Gradio visual studio with 3D viewport & control deck |
| [run.sh](file:///c:/Users/evabo/Documents/Projekti/3D%20Generation/Repos/TetraDiffusion/pipeline/run.sh) | Convenience bash wrapper script |
| [README.md](file:///c:/Users/evabo/Documents/Projekti/3D%20Generation/Repos/TetraDiffusion/pipeline/README.md) | This guide |

---

## 💡 Summary of Dataset Categories

- **UroCell Dataset** (`--dataset urocell`):
  - `lyso`: Lysosomes
  - `mito`: Mitochondria
  - `fv`: Fusiform Vesicles
- **OpenOrganelle Dataset** (`--dataset openorganelle`):
  - `Lysosome`: Lysosomes
  - `Mitochondria`: Mitochondria
  - `Golgi`: Golgi Apparatus
  - `ER`: Endoplasmic Reticulum
