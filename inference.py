from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from lib.Trainer import Trainer
from lib.ops.Utils import plot_and_save_meshes

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference and mesh generation."""
    parser = argparse.ArgumentParser(description="TetraDiffusion Mesh Generation (Inference)")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to directory containing run config.yaml and ds.pth.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of meshes to generate (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run inference on (cuda/cpu).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA GPU index when device=cuda.",
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Force Weights & Biases into offline mode.",
    )
    parser.add_argument(
        "--force_load_weights",
        action="store_true",
        help="Force loading model weights even if config load_weights is False.",
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default=None,
        help="Subdirectory name inside results folder for outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Explicit output directory (overrides out_subdir and results_folder).",
    )
    parser.add_argument(
        "--comparison_mode",
        action="store_true",
        help="Deterministic comparison mode: fixed initial noise & deterministic reverse sampling.",
    )
    parser.add_argument(
        "--generation_mode",
        action="store_true",
        help="Stochastic generation mode: shared initial noise & stochastic reverse sampling.",
    )
    parser.add_argument(
        "--stochastic_sampling",
        action="store_true",
        help="Use stochastic reverse sampling instead of deterministic mode.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to 2D conditioning image (.png, .tif, .npy, .npz).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Classifier-Free Guidance (CFG) scale for image conditioning.",
    )
    return parser.parse_args()


def _resolve_inference_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Tuple[str, bool]:
    """Validate argument combinations and return (inference_mode_str, is_deterministic)."""
    if args.comparison_mode and args.generation_mode:
        parser.error("--comparison_mode and --generation_mode cannot be used together")

    if args.comparison_mode and args.stochastic_sampling:
        parser.error("--comparison_mode conflicts with --stochastic_sampling")

    if args.generation_mode or args.stochastic_sampling:
        return "generation", False

    return "comparison", True


def _load_conditioning_image(image_path: str | None, device: torch.device) -> torch.Tensor | None:
    """
    Load a 2D conditioning image from disk and return a (1, 1, H, W) float32 tensor
    normalised to [0, 1] on the target device.
    """
    if image_path is None:
        return None

    path = pathlib.Path(image_path)
    ext = path.suffix.lower()

    if ext in (".npy",):
        arr = np.load(str(path)).astype(np.float32)
    elif ext in (".npz",):
        data = np.load(str(path))
        arr = data[list(data.keys())[0]].astype(np.float32)
    else:
        from PIL import Image as PILImage

        pil = PILImage.open(str(path))
        arr = np.array(pil).astype(np.float32)

    # Normalise pixel intensity to [0, 1]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()

    # Convert RGB (H, W, C) to grayscale (H, W) if needed
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor.to(device)


def _sanitize_name_component(value: Any, fallback: str = "sample") -> str:
    """Sanitize string components for filesystem safe folder naming."""
    val_str = str(value or "").strip()
    if not val_str:
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", val_str).strip("._-")
    return sanitized or fallback


def _get_organelle_name(cfg: Any) -> str:
    """Extract category or organelle name from configuration."""
    shapenet_ids = getattr(getattr(cfg, "dataset", None), "shapenet_ids", None)
    if shapenet_ids:
        return _sanitize_name_component(shapenet_ids[0], fallback="sample")
    return _sanitize_name_component(getattr(cfg, "name", None), fallback="sample")


def _get_default_output_subdir(cfg: Any, config_path: str) -> str:
    """Generate default timestamped output subdirectory name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _sanitize_name_component(
        getattr(cfg, "name", None) or os.path.basename(os.path.normpath(config_path)),
        fallback="run",
    )
    organelle_name = _get_organelle_name(cfg)

    if organelle_name.lower() in run_name.lower():
        return f"inference_{run_name}_{timestamp}"

    return f"inference_{organelle_name}_{run_name}_{timestamp}"


def _build_inference_manifest(
    cfg: Any,
    args: argparse.Namespace,
    output_dir: str,
    original_steps: List[int],
    effective_steps: List[int],
    inference_mode: str,
    deterministic_sampling: bool,
) -> Dict[str, Any]:
    """Create metadata dictionary detailing inference setup and settings."""
    organelle_name = _get_organelle_name(cfg)
    run_name = getattr(cfg, "name", None) or os.path.basename(os.path.normpath(args.config_path))
    return {
        "created_at": datetime.datetime.now().isoformat(),
        "run_name": run_name,
        "organelle": organelle_name,
        "config_path": os.path.abspath(args.config_path),
        "output_dir": os.path.abspath(output_dir),
        "num_images": args.num_images,
        "device": args.device,
        "inference_mode": inference_mode,
        "same_initial_noise_across_sampling_steps": True,
        "deterministic_reverse_sampling": deterministic_sampling,
        "original_sampling_steps": original_steps,
        "effective_sampling_steps": effective_steps,
        "sampling_steps_deduplicated_for_comparison": original_steps != effective_steps,
        "filename_step_label": "stepsize_<n>",
        "filename_step_label_note": "For backwards compatibility filenames use 'stepsize_<n>', where <n> is diffusion steps.",
        "comparison_note": "All effective sampling step counts start from the same initial latent noise.",
    }


def _write_inference_manifest(output_dir: str, manifest: Dict[str, Any]) -> None:
    """Save metadata JSON file into output directory."""
    manifest_path = os.path.join(output_dir, "inference_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[inference] Wrote manifest to: {manifest_path}")


def generate_meshes(
    trainer: Trainer,
    num_images: int = 10,
    batch_size: int = 1,
    device_type: str = "cuda",
    cond_image: torch.Tensor | None = None,
    output_dir: str = "",
    deterministic_sampling: bool = True,
) -> None:
    """
    Generate meshes using the trained diffusion model and save as Wavefront .obj files.

    Args:
        trainer: Instantiated Trainer object holding model and dataset.
        num_images: Number of meshes to generate.
        batch_size: Batch size per sampling step (default 1).
        device_type: 'cuda' or 'cpu'.
        cond_image: Optional 2D image tensor for conditioning.
        output_dir: Destination folder for output OBJ meshes.
        deterministic_sampling: Enable/disable stochastic reverse process noise.
    """
    acc = trainer.accelerator

    if not acc.is_main_process:
        print("[inference] Non-main process rank: skipping generation.")
        return

    # Extract evaluation / EMA model
    if hasattr(trainer, "ema") and getattr(trainer.ema, "ema_model", None) is not None:
        sampling_model = trainer.ema.ema_model
    else:
        if hasattr(acc, "unwrap_model"):
            sampling_model = acc.unwrap_model(trainer.model)
        else:
            sampling_model = getattr(trainer.model, "module", trainer.model)

    sampling_model.eval()
    organelle_prefix = _get_organelle_name(trainer.cfg)

    for k in tqdm(range(num_images), desc="Generating meshes"):
        with torch.inference_mode():
            if device_type == "cuda":
                with torch.autocast(device_type=device_type):
                    all_images_list = list(
                        sampling_model.sample(
                            batch_size=batch_size,
                            deterministic=deterministic_sampling,
                            image=cond_image,
                        )
                    )
            else:
                all_images_list = list(
                    sampling_model.sample(
                        batch_size=batch_size,
                        deterministic=deterministic_sampling,
                        image=cond_image,
                    )
                )

            all_images = torch.stack(all_images_list, dim=0)
            plot_and_save_meshes(
                all_images,
                trainer.ds,
                trainer.cfg,
                output_dir,
                k,
                file_prefix=organelle_prefix,
            )


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    args = parse_args()
    inference_mode, deterministic_sampling = _resolve_inference_mode(parser, args)

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_API_KEY"] = ""

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_type = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        device_type = "cuda"

    # Load run configuration
    cfg_path = os.path.join(args.config_path, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    original_sampling_steps = []

    # Deduplicate sampling step list for comparison mode
    if getattr(getattr(cfg, "diffusion", None), "sampling_steps", None):
        sampling_steps = list(cfg.diffusion.sampling_steps)
        original_sampling_steps = sampling_steps.copy()
        unique_sampling_steps = list(dict.fromkeys(sampling_steps))
        if unique_sampling_steps != sampling_steps:
            print(
                f"[inference] Deduplicated sampling_steps: {unique_sampling_steps} (from {sampling_steps})"
            )
            cfg.diffusion.sampling_steps = unique_sampling_steps

    effective_sampling_steps = list(getattr(getattr(cfg, "diffusion", None), "sampling_steps", []))

    if args.force_load_weights:
        cfg.load_weights = True

    if args.cfg_scale is not None:
        if not OmegaConf.select(cfg, "image_cond"):
            OmegaConf.update(cfg, "image_cond", {})
        OmegaConf.update(cfg, "image_cond.cfg_guidance_scale", args.cfg_scale)

    # Initialize Trainer in inference mode
    trainer = Trainer(
        train_batch_size=cfg.training.batch_size,
        save_and_sample_every=cfg.training.test_every,
        results_folder=cfg.results_folder,
        config_folder=args.config_path,
        num_samples=1,
        train_lr=cfg.training.lr,
        train_num_steps=cfg.training.num_steps,
        gradient_accumulate_every=cfg.training.ga,
        ema_decay=cfg.training.ema_decay,
        cfg=cfg,
        inference=True,
    )

    # Determine destination output directory
    if args.out_dir:
        output_dir = args.out_dir
    else:
        sub = args.out_subdir or _get_default_output_subdir(cfg, args.config_path)
        output_dir = os.path.join(cfg.results_folder, sub)

    os.makedirs(output_dir, exist_ok=True)

    print(f"[inference] Writing output OBJ meshes to: {output_dir}")
    print(f"[inference] Sampling steps: {effective_sampling_steps}")
    print(f"[inference] Mode: {inference_mode} (Deterministic={deterministic_sampling})")

    # Load 2D conditioning image if provided
    cond_image = _load_conditioning_image(args.image_path, device=trainer.accelerator.device)
    if cond_image is not None:
        guidance = getattr(getattr(cfg, "image_cond", None), "cfg_guidance_scale", 1.0)
        print(f"[inference] Loaded conditioning image: {args.image_path} (CFG scale={guidance})")
    elif getattr(getattr(cfg, "image_cond", None), "enabled", False):
        print("[inference] image_cond enabled but no image provided -> using unconditional null embedding.")

    # Write manifest and run generation
    manifest = _build_inference_manifest(
        cfg,
        args,
        output_dir,
        original_sampling_steps,
        effective_sampling_steps,
        inference_mode,
        deterministic_sampling,
    )
    _write_inference_manifest(output_dir, manifest)

    generate_meshes(
        trainer,
        num_images=args.num_images,
        device_type=device_type,
        cond_image=cond_image,
        output_dir=output_dir,
        deterministic_sampling=deterministic_sampling,
    )


if __name__ == "__main__":
    main()

