# AGENTS.md

## Project at a glance
- `main.py` is the training entry point: it merges `config/config.yaml` + `config/path.yaml`, applies CLI overrides, writes `results_folder/config.yaml`, then builds `lib/Trainer.py`.
- `inference.py` reuses the same `Trainer` in `inference=True` mode and samples meshes through `lib/DDPM.py` + `lib/ops/Utils.py`.
- Core model stack is `lib/UVIT.py` (tetrahedral U-Net + ViT bottleneck) wrapped by `lib/DDPM.py` (`GaussianDiffusion`).
- Data contract is owned by `lib/Tetradata.py` (`MeshLoader`), which loads tetrahedral assets from `tetrahedra/{grid_res}` and dataset samples from `data_path`.

## Data and artifact flow you must preserve
- Raw samples are discovered as `"{data_path}/{shapenet_id}/*/*/sample.pth"` in `MeshLoader._init_gt_iterative`.
- On first train run, processed samples are cached under `"{data_path}/preprocessed_data/samples/sample_{i}.pt"`.
- `Trainer` saves `ds.pth` into `config_folder` (training path), and inference requires that file at `--config_path/ds.pth`.
- Checkpoints are `model-*.pt` in `config_folder`; loading picks the newest file by ctime (`glob("*.pt")` + `max`).
- Mesh exports are OBJ files named like `"{results_folder}/{k}_stepsize_{sampling_step}.obj"`.

## Canonical workflows
- Train (single/multi GPU) uses Accelerate, not direct `python main.py` for production runs:
  - `accelerate launch --multi_gpu --num_processes <N> --gpu_ids all main.py --data_path <root> --grid_res 128 --name <run>`
- Inference expects a run directory with at least `config.yaml`, `ds.pth`, and checkpoint(s):
  - `python inference.py --config_path results/<run_name>`
- Preprocessing is a separate pipeline under `preprocessing/`; `fit_single.py` writes a JSON config and shells into `preprocessing/train.py`.

## Repo-specific conventions
- Config precedence is strict: base YAMLs -> CLI overrides in `main.py` via `OmegaConf.update` -> saved run-local `config.yaml`.
- `dataset.color` toggles channel count (`4` vs `7`) across model + diffusion; keep this consistent when editing I/O shapes.
- Grid pruning and masking are intertwined in `MeshLoader.mask_sdfs_or_disps_it` and `GridPruning.mask_cube`; changes here affect stats, normalization, and mesh extraction.
- `MeshLoader.get_statistics()` scans cached training samples at startup; startup cost is expected and not a hang.

## Integration points and external deps
- Training always initializes Weights & Biases in `Trainer.__init__` (`wandb.init(project="debug_accumulate")`), so offline/CI runs should set `WANDB_MODE=offline` if needed.
- Distributed behavior uses `accelerate.Accelerator` and optional `LocalSGD`; respect `accelerator.is_main_process` guards for saves/logging.
- `preprocessing/train.py` depends on `nvdiffrast`, `xatlas`, and CUDA rendering code in `preprocessing/render/renderutils/c_src`.

## Known sharp edges (important for agent edits)
- `job_cars.sh` passes flags (`--threshold`, `--load_weights`) that `main.py` does not parse; do not assume that script is up to date.
- Inference does not force `cfg.load_weights=True`; sampling can run with random weights if config/checkpoints are mismatched.
- `lib/ops/Utils.py` references `dataset.get_mesh_wo_color` when `dataset.color=False`, but `MeshLoader` only defines `get_mesh`; color-off path appears incomplete.
- Paths in `config/path.yaml` are cluster-oriented by default (`/scratch...`); local runs almost always need `--data_path` and `--name` overrides.

## High-value files to read before major changes
- `main.py`, `inference.py`, `lib/Trainer.py`, `lib/Tetradata.py`, `lib/DDPM.py`, `lib/UVIT.py`
- `config/config.yaml`, `config/path.yaml`, `job_cars.sh`, `preprocessing/README.md`

