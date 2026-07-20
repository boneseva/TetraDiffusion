#!/usr/bin/env python3
"""
probe_batch_size.py — find the largest batch size that fits in VRAM.

Runs 3 forward+backward passes at each candidate batch size.
Exits cleanly with a summary — no checkpoints, no WandB.

Usage (submit via submit_batch_probe.sh, or run directly):
    python scripts/probe_batch_size.py \
        --data_path data_test/preprocessed \
        --csv_path  lib/all_urocell.csv \
        --category  lyso \
        --sizes     16 32 48 64 96 128
"""
import argparse
import sys
import gc
import torch
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--csv_path',  required=True)
parser.add_argument('--category',  default='lyso')
parser.add_argument('--grid_res',  type=int, default=128)
parser.add_argument('--sizes',     type=int, nargs='+',
                    default=[16, 24, 32, 48, 64, 96, 128])
parser.add_argument('--steps',     type=int, default=3,
                    help='Forward+backward passes per batch size (default 3)')
args = parser.parse_args()

# ── Build config (same as main.py) ────────────────────────────────────────────
cfg = OmegaConf.merge(OmegaConf.load('config/config.yaml'),
                      OmegaConf.load('config/path.yaml'))
OmegaConf.update(cfg, 'data_path',              args.data_path)
OmegaConf.update(cfg, 'dataset.grid_res',       args.grid_res)
OmegaConf.update(cfg, 'dataset.shapenet_ids',   [args.category])
OmegaConf.update(cfg, 'splits_csv',             args.csv_path)
OmegaConf.update(cfg, 'load_weights',           False)
OmegaConf.update(cfg, 'training.mixed_precision', True)
OmegaConf.update(cfg, 'name',                   'batch_probe')
OmegaConf.update(cfg, 'results_folder',         '/tmp/batch_probe')
OmegaConf.update(cfg, 'num_workers',            4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

print('=' * 60)
print(f'  GPU  : {gpu_name}')
print(f'  VRAM : {total_vram:.1f} GB')
print(f'  Sizes: {args.sizes}')
print('=' * 60)

# ── Load dataset once ─────────────────────────────────────────────────────────
import os
os.makedirs('/tmp/batch_probe', exist_ok=True)

# Suppress WandB — we don't want a real run
os.environ['WANDB_MODE'] = 'disabled'

from torch.utils.data import DataLoader
from lib.Tetradata import MeshLoader
from lib.UVIT import UVIT
from lib.DDPM import GaussianDiffusion
from torch.cuda.amp import autocast, GradScaler

print('[probe] Loading dataset...')
ds = MeshLoader(config=cfg, device='cpu', cuda_device=device, accelerator=None)
print(f'[probe] Dataset loaded — {len(ds)} samples.')

num_verts = len(ds.tet_verts)
channels  = 4 + (3 if cfg.dataset.color else 0)

# ── Probe each batch size ─────────────────────────────────────────────────────
results = []

for bs in args.sizes:
    torch.cuda.empty_cache()
    gc.collect()

    print(f'\n[probe] Trying batch_size={bs} ...', flush=True)
    try:
        model = UVIT(cfg, rank=device, ds=ds).to(device)
        diffusion = GaussianDiffusion(
            model,
            num_verts=num_verts,
            channels=channels,
            image_size=cfg.dataset.grid_res,
            noise_d=cfg.diffusion.noise_d,
            cfg=cfg,
            pred_objective=cfg.diffusion.pred_objective,
            num_sample_steps=cfg.diffusion.sampling_steps,
            offset_noise_strength=cfg.diffusion.offset_noise,
            ds=ds,
        ).to(device)
        diffusion.mask = ds.mask_verts

        opt    = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)
        scaler = GradScaler()
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2,
                            pin_memory=True)

        peak_mb = 0
        for step, batch in enumerate(loader):
            if step >= args.steps:
                break
            opt.zero_grad()
            with autocast():
                loss = diffusion(batch)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()

        print(f'  ✓  batch_size={bs:4d}  peak VRAM: {peak_mb/1e3:.2f} GB')
        results.append((bs, True, peak_mb))

    except torch.cuda.OutOfMemoryError:
        print(f'  ✗  batch_size={bs:4d}  → OOM')
        results.append((bs, False, None))
        break   # no point trying larger sizes

    finally:
        # Free GPU memory before next iteration
        try:
            del diffusion, model, opt, scaler, loader
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  BATCH SIZE PROBE — SUMMARY')
print(f'  GPU : {gpu_name}  ({total_vram:.1f} GB)')
print('=' * 60)
passing = [bs for bs, ok, _ in results if ok]
if passing:
    best = max(passing)
    print(f'  Maximum fitting batch size: {best}')
    print()
    for bs, ok, mb in results:
        status = f'OK  ({mb/1e3:.2f} GB peak)' if ok else 'OOM'
        print(f'    bs={bs:4d}  {status}')
    print()
    print(f'  → Use --batch_size {best} in ablation_fast_sweep.sh')
else:
    print('  All batch sizes OOM — try reducing grid_res or enabling ga.')
print('=' * 60)
