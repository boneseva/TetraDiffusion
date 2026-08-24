import argparse
import os
import warnings
from omegaconf import OmegaConf
import torch

from lib.Trainer import Trainer
from lib.ops.BioConstraints import ExemplarLossProfile, organelle_loss_registry

warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning,
)
warnings.simplefilter(action="ignore", category=FutureWarning)

torch.set_float32_matmul_precision("high")
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TetraDiffusion Training Pipeline")
    parser.add_argument("--data_path", type=str, help="Root directory of datasets.")
    parser.add_argument("--grid_res", type=int, help="Tetrahedral grid resolution.")
    parser.add_argument(
        "--shapenet_id",
        type=str,
        nargs="+",
        help="One or more category names to train on.",
    )
    parser.add_argument("--name", type=str, help="Run name for logging and output folder.")
    parser.add_argument("--batch_size", type=int, help="Per-GPU batch size.")
    parser.add_argument("--ga", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--num_steps", type=int, help="Total training steps.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint.",
    )

    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name.")
    parser.add_argument(
        "--bio_loss_weight",
        type=float,
        help="Weight for biological constraint loss (overrides config).",
    )
    parser.add_argument(
        "--no_bio_loss",
        action="store_true",
        help="Disable biological constraint loss (sets bio_loss_weight=0).",
    )
    parser.add_argument(
        "--bio_loss_type",
        type=str,
        choices=["laplacian", "curvature", "both"],
        help="Which biological loss term to use (overrides config).",
    )
    parser.add_argument(
        "--no_snr_weighting",
        action="store_true",
        help="Disable SNR-based weighting of bio loss (applies it uniformly across all noise levels).",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["warmup_constant", "constant", "warmup_cosine", "cosine"],
        help="LR schedule to use (overrides config). Default: warmup_constant.",
    )
    parser.add_argument(
        "--image_cond",
        action="store_true",
        help="Enable 2D image conditioning (sets image_cond.enabled=True).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="CFG guidance scale for image-conditioned inference (overrides config).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to the train/test splits CSV (default: lib/all.csv).",
    )
    parser.add_argument(
        "--exemplar_path",
        type=str,
        default=None,
        help="Path to a preprocessed .pt exemplar sample file for shape prior loss.",
    )
    # Ablation sweep overrides
    parser.add_argument("--offset_noise", type=float, help="Offset noise strength.")
    parser.add_argument(
        "--sampling_steps",
        type=int,
        nargs="+",
        help="Inference sampling step counts as a space-separated list.",
    )
    parser.add_argument("--sdf_bg_loss_weight", type=float, help="Weight for SDF background loss.")
    parser.add_argument(
        "--sdf_bg_threshold",
        type=float,
        help='Normalised SDF cutoff defining "background" (0.2-0.5).',
    )
    parser.add_argument(
        "--snr_gate",
        type=str,
        choices=["soft", "hard_0.3", "hard_0.5", "hard_0.7", "none"],
        help="SNR gate mode for bio loss.",
    )
    parser.add_argument(
        "--bio_curvature_softness",
        type=float,
        help="Surface kernel width sigma for curvature bio loss.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable FP16 mixed precision training.",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        help="Number of training steps between saving checkpoints and sampling meshes.",
    )
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of training set to use (0.0 < fraction <= 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for dataset ordering and model training.",
    )
    parser.add_argument(
        "--train_split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable train/test dataset split (if omitted, respects config/path.yaml default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Global seeding
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load base configuration and path definitions
    cfg = OmegaConf.merge(OmegaConf.load("config/config.yaml"), OmegaConf.load("config/path.yaml"))

    # Apply CLI overrides to configuration
    if args.name is not None:
        OmegaConf.update(cfg, "name", args.name)
        OmegaConf.update(cfg, "results_folder", f"runs/{args.name}")

    if args.batch_size is not None:
        OmegaConf.update(cfg, "training.batch_size", args.batch_size)

    if args.ga is not None:
        OmegaConf.update(cfg, "training.ga", args.ga)

    if args.shapenet_id is not None:
        OmegaConf.update(cfg, "dataset.shapenet_ids", list(args.shapenet_id))

    if args.data_path is not None:
        OmegaConf.update(cfg, "data_path", args.data_path)

    if args.grid_res is not None:
        OmegaConf.update(cfg, "dataset.grid_res", args.grid_res)

    if args.num_steps is not None:
        OmegaConf.update(cfg, "training.num_steps", args.num_steps)

    if args.resume:
        OmegaConf.update(cfg, "load_weights", True)

    if args.wandb_project is not None:
        OmegaConf.update(cfg, "wandb_project", args.wandb_project)

    # Biological constraints overrides
    if args.no_bio_loss:
        OmegaConf.update(cfg, "diffusion.bio_loss_weight", 0.0)
    elif args.bio_loss_weight is not None:
        OmegaConf.update(cfg, "diffusion.bio_loss_weight", args.bio_loss_weight)

    if args.bio_loss_type is not None:
        OmegaConf.update(cfg, "diffusion.bio_loss_type", args.bio_loss_type)

    if args.no_snr_weighting:
        OmegaConf.update(cfg, "diffusion.bio_snr_weighting", False)

    if args.lr_schedule is not None:
        OmegaConf.update(cfg, "training.lr_schedule", args.lr_schedule)

    if args.image_cond:
        OmegaConf.update(cfg, "image_cond.enabled", True)

    if args.cfg_scale is not None:
        OmegaConf.update(cfg, "image_cond.cfg_guidance_scale", args.cfg_scale)

    if args.csv_path is not None:
        OmegaConf.update(cfg, "splits_csv", args.csv_path)

    # Ablation sweep overrides
    if args.offset_noise is not None:
        OmegaConf.update(cfg, "diffusion.offset_noise", args.offset_noise)

    if args.sampling_steps is not None:
        OmegaConf.update(cfg, "diffusion.sampling_steps", list(args.sampling_steps))

    if args.sdf_bg_loss_weight is not None:
        OmegaConf.update(cfg, "diffusion.sdf_bg_loss_weight", args.sdf_bg_loss_weight)

    if args.sdf_bg_threshold is not None:
        OmegaConf.update(cfg, "diffusion.sdf_bg_threshold", args.sdf_bg_threshold)

    if args.snr_gate is not None:
        OmegaConf.update(cfg, "diffusion.snr_gate", args.snr_gate)

    if args.bio_curvature_softness is not None:
        OmegaConf.update(cfg, "diffusion.bio_curvature_softness", args.bio_curvature_softness)

    if args.mixed_precision:
        OmegaConf.update(cfg, "training.mixed_precision", True)

    if args.dataset_fraction is not None:
        OmegaConf.update(cfg, "dataset.dataset_fraction", args.dataset_fraction)

    if args.seed is not None:
        OmegaConf.update(cfg, "seed", args.seed)

    if args.train_split is not None:
        OmegaConf.update(cfg, "dataset.train_split", args.train_split)

    if args.test_every is not None:
        OmegaConf.update(cfg, "training.test_every", args.test_every)

    # Ensure output results directory exists and dump final merged configuration
    os.makedirs(cfg.results_folder, exist_ok=True)
    config_save_path = os.path.join(cfg.results_folder, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    # Initialize distributed Trainer
    trainer = Trainer(
        cfg=cfg,
        train_batch_size=cfg.training.batch_size,
        save_and_sample_every=cfg.training.test_every,
        results_folder=cfg.results_folder,
        config_folder=cfg.results_folder,
        num_samples=1,
        train_lr=cfg.training.lr,
        train_num_steps=cfg.training.num_steps,
        gradient_accumulate_every=cfg.training.ga,
        ema_decay=cfg.training.ema_decay,
    )

    # Optional exemplar style prior registration
    if args.exemplar_path is not None and os.path.isfile(args.exemplar_path):
        exemplar_profile = ExemplarLossProfile(
            exemplar_path=args.exemplar_path,
            mask=trainer.ds.mask,
            neighbors=trainer.ds.neighbors[-1].long(),
        )
        organelle_loss_registry.register_loss("exemplar", exemplar_profile)
        OmegaConf.update(cfg, "diffusion.bio_loss_type", "exemplar")
        print("[main] Exemplar style prior registered. bio_loss_type set to 'exemplar'.")
    elif args.exemplar_path is not None:
        print(f"[main] WARNING: --exemplar_path '{args.exemplar_path}' not found; skipping.")

    # Start training loop
    trainer.train()


if __name__ == "__main__":
    main()

