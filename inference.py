from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from lib.image_preprocessing import normalize_image, prepare_slice
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
        help="Path to a single 2D conditioning image (.npy, .png, .tif, .npz).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help=(
            "Directory of 2D conditioning images. Generates --num_images meshes "
            "per image. Filenames are used as output stems."
        ),
    )
    parser.add_argument(
        "--image_preprocessed",
        action="store_true",
        help=(
            "Input images are already anisotropy-corrected, letterboxed, normalised, "
            "and resized to proj_size x proj_size (e.g. image_xy_p*.npy from the "
            "extraction pipeline). No geometric or intensity preprocessing is applied."
        ),
    )
    parser.add_argument(
        "--voxel_spacing_xy",
        type=float,
        nargs=2,
        metavar=("SX", "SY"),
        default=None,
        help=(
            "Physical voxel spacing along axis-0 (sx) and axis-1 (sy) of the input "
            "image (same units; only ratio matters). Required in raw-input mode "
            "(when --image_preprocessed is NOT set). Ignored when --image_preprocessed "
            "is set."
        ),
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


def _load_conditioning_image(
    image_path: str | None,
    device: torch.device,
    proj_size: int = 64,
    preprocessed: bool = False,
    voxel_spacing_xy: Optional[Tuple[float, float]] = None,
) -> torch.Tensor | None:
    """
    Load a 2D conditioning image and return a (1, 1, proj_size, proj_size) float32
    tensor on *device*.

    Two mutually exclusive input modes:

    Raw mode (default, ``preprocessed=False``)
        The caller provides a tight 2D EM crop at any resolution.
        ``prepare_slice()`` and ``normalize_image()`` are applied exactly once.
        ``voxel_spacing_xy`` (sx, sy) is required for anisotropy correction.

    Preprocessed mode (``preprocessed=True``)
        The caller provides an image that is already proj_size x proj_size,
        float32, normalised to [0,1] — e.g. image_xy_p*.npy produced by
        scripts/extract_instances_from_nifti.py.  No geometric or intensity
        processing is applied; the array is validated and forwarded directly.
    """
    if image_path is None:
        return None

    path = pathlib.Path(image_path)
    ext  = path.suffix.lower()

    # ── Load to 2-D float32 numpy array ───────────────────────────────
    if ext == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    elif ext == ".npz":
        data = np.load(str(path))
        arr  = data[list(data.keys())[0]].astype(np.float32)
    else:
        from PIL import Image as PILImage
        pil = PILImage.open(str(path))
        arr = np.array(pil).astype(np.float32)
        if arr.ndim == 3:          # RGB/RGBA → grayscale
            arr = arr.mean(axis=2)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D image, got shape {arr.shape} in {path}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in conditioning image {path}")

    # ── Mode-dependent processing ─────────────────────────────────────
    if preprocessed:
        # Validate only: caller guarantees proj_size x proj_size, [0,1]
        if arr.shape != (proj_size, proj_size):
            raise ValueError(
                f"--image_preprocessed expects ({proj_size}, {proj_size}), "
                f"got {arr.shape}. Remove --image_preprocessed to let inference "
                f"resize and normalise automatically."
            )
    else:
        # Raw mode: apply the same pipeline as the extraction script
        if voxel_spacing_xy is None:
            raise ValueError(
                "--voxel_spacing_xy SX SY is required in raw-input mode. "
                "Pass the physical voxel spacing from the source volume's NIfTI "
                "header, or use --image_preprocessed if the image is already "
                f"resized to {proj_size}x{proj_size} and normalised."
            )
        sx, sy = float(voxel_spacing_xy[0]), float(voxel_spacing_xy[1])
        arr = prepare_slice(arr, sx=sx, sy=sy, proj_size=proj_size)
        arr = normalize_image(arr)

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor.to(device)  # (1, 1, proj_size, proj_size)


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

    # ── Determine proj_size from config ───────────────────────────────
    proj_size = int(
        getattr(getattr(cfg, "image_cond", None), "proj_size", 64)
    )

    def _load_image(p: str) -> torch.Tensor | None:
        return _load_conditioning_image(
            p,
            device=trainer.accelerator.device,
            proj_size=proj_size,
            preprocessed=args.image_preprocessed,
            voxel_spacing_xy=args.voxel_spacing_xy,
        )

    # ── Single image or directory mode ────────────────────────────────
    if args.image_dir:
        image_files = sorted(
            pathlib.Path(args.image_dir).glob("image_xy_p*.npy")
        )
        if not image_files:
            # Fallback: any .npy / .png in the directory
            image_files = (
                sorted(pathlib.Path(args.image_dir).glob("*.npy")) +
                sorted(pathlib.Path(args.image_dir).glob("*.png"))
            )
        if not image_files:
            print(
                f"[inference] ERROR: No image files found in {args.image_dir}",
                flush=True,
            )
            return

        print(
            f"[inference] --image_dir mode: {len(image_files)} image(s), "
            f"{args.num_images} mesh(es) each."
        )
        for img_file in image_files:
            cond_image = _load_image(str(img_file))
            img_output_dir = os.path.join(output_dir, img_file.stem)
            os.makedirs(img_output_dir, exist_ok=True)
            generate_meshes(
                trainer,
                num_images=args.num_images,
                device_type=device_type,
                cond_image=cond_image,
                output_dir=img_output_dir,
                deterministic_sampling=deterministic_sampling,
            )
    else:
        # Single image or unconditional
        cond_image = _load_image(args.image_path)
        if cond_image is not None:
            guidance = getattr(getattr(cfg, "image_cond", None), "cfg_guidance_scale", 1.0)
            print(f"[inference] Loaded conditioning image: {args.image_path} (CFG scale={guidance})")
        elif getattr(getattr(cfg, "image_cond", None), "enabled", False):
            print("[inference] image_cond enabled but no image provided -> unconditional null embedding.")

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

