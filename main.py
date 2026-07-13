import argparse
from omegaconf import OmegaConf
from lib.Trainer import Trainer
from lib.ops.BioConstraints import ExemplarLossProfile, organelle_loss_registry
import torch
import warnings
# Suppress specific warnings - it doesnt matter in inference
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None", category=UserWarning)

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.set_float32_matmul_precision('high')
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.suppress_errors = True  # suppress inductor/nvcc errors
torch._dynamo.disable()  # disable TorchDynamo JIT compilation to avoid nvcc permission issues
cfg = OmegaConf.merge(OmegaConf.load('config/config.yaml'), OmegaConf.load('config/path.yaml'))

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Root directory of datasets.')
parser.add_argument('--grid_res', type=int)
parser.add_argument('--shapenet_id', type=str, nargs='+',
                    help='One or more category names (space-separated) to train on.')
parser.add_argument('--name', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--ga', type=int)
parser.add_argument('--num_steps', type=int, help='Total training steps (overrides config).')
parser.add_argument('--resume', action='store_true',
                    help='Resume training: load latest checkpoint and continue the same wandb run.')
parser.add_argument('--wandb_project', type=str, help='Weights & Biases project name.')
parser.add_argument('--bio_loss_weight', type=float,
                    help='Weight for biological constraint loss (overrides config).')
parser.add_argument('--no_bio_loss', action='store_true',
                    help='Disable biological constraint loss (sets bio_loss_weight=0).')
parser.add_argument('--bio_loss_type', type=str, choices=['laplacian', 'curvature', 'both'],
                    help='Which biological loss term to use (overrides config).')
parser.add_argument('--no_snr_weighting', action='store_true',
                    help='Disable SNR-based weighting of bio loss (applies it uniformly across all noise levels).')
parser.add_argument('--lr_schedule', type=str,
                    choices=['warmup_constant', 'constant', 'warmup_cosine', 'cosine'],
                    help='LR schedule to use (overrides config). Default: warmup_constant.')
parser.add_argument('--image_cond', action='store_true',
                    help='Enable 2D image conditioning (sets image_cond.enabled=True). '
                         'Requires a fresh training run — incompatible with unconditional checkpoints '
                         'unless fine-tuning with strict=False (image encoder starts random).')
parser.add_argument('--cfg_scale', type=float, default=None,
                    help='CFG guidance scale for image-conditioned inference (overrides config).')
parser.add_argument('--csv_path', type=str, default=None,
                    help='Path to the train/test splits CSV (default: lib/all.csv). '
                         'Use lib/all_urocell.csv for UroCell data.')
parser.add_argument('--exemplar_path', type=str, default=None,
                    help='Path to a preprocessed .pt exemplar sample file. '
                         'When provided, registers an ExemplarLossProfile on the '
                         '"exemplar" organelle slot and sets bio_loss_type=exemplar.')

# ── Ablation-sweep flags (previously config-only) ─────────────────────────
parser.add_argument('--offset_noise', type=float,
                    help='Offset noise strength (overrides diffusion.offset_noise). '
                         'Typical sweep: 0.0 / 0.05 / 0.1 / 0.2.')
parser.add_argument('--sampling_steps', type=int, nargs='+',
                    help='Inference sampling step counts as a space-separated list '
                         '(overrides diffusion.sampling_steps). '
                         'e.g. --sampling_steps 50 100')
parser.add_argument('--sdf_bg_loss_weight', type=float,
                    help='Weight for SDF background loss (overrides diffusion.sdf_bg_loss_weight). '
                         'Set >0 to enable; suggested starting value: 0.05.')
parser.add_argument('--sdf_bg_threshold', type=float,
                    help='Normalised SDF cutoff defining "background" '
                         '(overrides diffusion.sdf_bg_threshold). Range 0.2–0.5.')
parser.add_argument('--snr_gate', type=str,
                    choices=['soft', 'hard_0.3', 'hard_0.5', 'hard_0.7', 'none'],
                    help='SNR gate mode for bio loss (overrides diffusion.snr_gate). '
                         'soft=continuous weighting (default), none=no gating.')
parser.add_argument('--bio_curvature_softness', type=float,
                    help='Surface kernel width σ for curvature bio loss '
                         '(overrides diffusion.bio_curvature_softness). '
                         'Typical sweep: 0.05 / 0.10 / 0.15 / 0.20 / 0.30.')
parser.add_argument('--mixed_precision', action='store_true',
                    help='Enable FP16 mixed precision training to speed up runs.')
parser.add_argument('--test_every', type=int,
                    help='Number of training steps between saving checkpoints and sampling meshes.')
args = parser.parse_args()

if args.name is not None:
    OmegaConf.update(cfg, 'name', args.name)                        # short name → wandb display name
    OmegaConf.update(cfg, 'results_folder', f"runs/{args.name}")   # checkpoints → runs/<name>/

if args.batch_size is not None:
    OmegaConf.update(cfg, 'training.batch_size', args.batch_size)

if args.ga is not None:
    OmegaConf.update(cfg, 'training.ga', args.ga)

if args.shapenet_id is not None:
    OmegaConf.update(cfg, 'dataset.shapenet_ids', list(args.shapenet_id))

if args.data_path is not None:
    OmegaConf.update(cfg, 'data_path', args.data_path)

if args.grid_res is not None:
    OmegaConf.update(cfg, 'dataset.grid_res', args.grid_res)

if args.num_steps is not None:
    OmegaConf.update(cfg, 'training.num_steps', args.num_steps)

if args.resume:
    OmegaConf.update(cfg, 'load_weights', True)

if args.wandb_project is not None:
    OmegaConf.update(cfg, 'wandb_project', args.wandb_project)

# ── Biological constraints overrides ───────────────────────────────────────
if args.no_bio_loss:
    OmegaConf.update(cfg, 'diffusion.bio_loss_weight', 0.0)
elif args.bio_loss_weight is not None:
    OmegaConf.update(cfg, 'diffusion.bio_loss_weight', args.bio_loss_weight)

if args.bio_loss_type is not None:
    OmegaConf.update(cfg, 'diffusion.bio_loss_type', args.bio_loss_type)

if args.no_snr_weighting:
    OmegaConf.update(cfg, 'diffusion.bio_snr_weighting', False)

if args.lr_schedule is not None:
    OmegaConf.update(cfg, 'training.lr_schedule', args.lr_schedule)

if args.image_cond:
    OmegaConf.update(cfg, 'image_cond.enabled', True)

if args.cfg_scale is not None:
    OmegaConf.update(cfg, 'image_cond.cfg_guidance_scale', args.cfg_scale)

if args.csv_path is not None:
    OmegaConf.update(cfg, 'splits_csv', args.csv_path)

# ── Ablation-sweep overrides ───────────────────────────────────────────────
if args.offset_noise is not None:
    OmegaConf.update(cfg, 'diffusion.offset_noise', args.offset_noise)

if args.sampling_steps is not None:
    OmegaConf.update(cfg, 'diffusion.sampling_steps', list(args.sampling_steps))

if args.sdf_bg_loss_weight is not None:
    OmegaConf.update(cfg, 'diffusion.sdf_bg_loss_weight', args.sdf_bg_loss_weight)

if args.sdf_bg_threshold is not None:
    OmegaConf.update(cfg, 'diffusion.sdf_bg_threshold', args.sdf_bg_threshold)

if args.snr_gate is not None:
    OmegaConf.update(cfg, 'diffusion.snr_gate', args.snr_gate)

if args.bio_curvature_softness is not None:
    OmegaConf.update(cfg, 'diffusion.bio_curvature_softness', args.bio_curvature_softness)

if args.mixed_precision:
    OmegaConf.update(cfg, 'training.mixed_precision', True)

if args.test_every is not None:
    OmegaConf.update(cfg, 'training.test_every', args.test_every)


print(cfg)

import os
os.makedirs(cfg.results_folder,exist_ok=True)

with open(cfg.results_folder+"/config.yaml", "w") as f:
   OmegaConf.save(cfg, f)

trainer = Trainer(
    cfg=cfg,
    train_batch_size = cfg.training.batch_size,
    save_and_sample_every = cfg.training.test_every,
    results_folder = cfg.results_folder,
    config_folder=  cfg.results_folder,
    num_samples = 1,
    train_lr = cfg.training.lr,
    train_num_steps = cfg.training.num_steps,         # total training steps
    gradient_accumulate_every = cfg.training.ga,    # gradient accumulation steps
    ema_decay = cfg.training.ema_decay,                # exponential moving average decay
)

#if cfg.load_weights:
#    trainer.load(cfg.num_weights)

# ── Exemplar style prior (optional) ────────────────────────────────────────
# Instantiate and register the ExemplarLossProfile only when --exemplar_path
# is supplied.  If omitted the registry stays empty for the "exemplar" slot
# and training falls back to the standard bio_loss_type from the config.
import os as _os
if args.exemplar_path is not None and _os.path.isfile(args.exemplar_path):
    _exemplar_profile = ExemplarLossProfile(
        exemplar_path=args.exemplar_path,
        mask=trainer.ds.mask,
        neighbors=trainer.ds.neighbors[-1].long(),
    )
    organelle_loss_registry.register_loss("exemplar", _exemplar_profile)
    OmegaConf.update(cfg, 'diffusion.bio_loss_type', 'exemplar')
    print(f"[main] Exemplar style prior registered. bio_loss_type forced to 'exemplar'.")
elif args.exemplar_path is not None:
    print(f"[main] WARNING: --exemplar_path '{args.exemplar_path}' not found; "
          f"exemplar prior skipped.")

trainer.train()
