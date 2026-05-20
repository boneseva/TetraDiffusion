import random
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from lib.Trainer import Trainer
from lib.ops.Utils import plot_and_save_meshes
import argparse 
import warnings
import datetime
import re
import json

# Suppress specific warnings - it doesnt matter in inference
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None", category=UserWarning)

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set float32 matrix multiplication precision and other torch configurations
torch.set_float32_matmul_precision('high')
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.suppress_errors = True  # suppress inductor/nvcc errors
torch._dynamo.disable()  # disable TorchDynamo JIT compilation to avoid nvcc permission issues
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def seed_everything(seed: int):
    """Seed all necessary libraries and settings for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Seed the environment for reproducibility
seed_everything(42)
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--num_images', type=int, default=10, help='Number of meshes to generate (default: 10)')
parser.add_argument('--device', type=str, choices=['cpu','cuda'], default='cuda', help='Device to run inference on')
parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device index to expose when running on CUDA')
parser.add_argument('--wandb_offline', action='store_true', help='Force wandb into offline mode for inference')
parser.add_argument('--force_load_weights', action='store_true', help='Force loading of model weights even if config.load_weights is false')
parser.add_argument('--out_subdir', type=str, default=None, help='Subdirectory inside the run results folder to write inference outputs (default: organelle-aware inference folder name)')
parser.add_argument('--out_dir', type=str, default=None, help='Explicit output directory (overrides out_subdir and cfg.results_folder)')
parser.add_argument('--comparison_mode', action='store_true', help='Use deterministic comparison mode: shared initial noise and deterministic reverse sampling')
parser.add_argument('--generation_mode', action='store_true', help='Use stochastic generation mode: shared initial noise but stochastic reverse sampling')
parser.add_argument('--stochastic_sampling', action='store_true', help='Use stochastic reverse sampling instead of deterministic comparison mode')
parser.add_argument('--image_path', type=str, default=None,
                    help='Path to a 2D conditioning image (PNG/TIF/NPY). '
                         'Only used when the model was trained with image_cond.enabled=True. '
                         'If omitted the null embedding is used (unconditional).')
parser.add_argument('--cfg_scale', type=float, default=None,
                    help='Classifier-free guidance scale (overrides config). '
                         '1.0 = no guidance, 3–7 = moderate, 10+ = strong. '
                         'Only meaningful when --image_path is provided.')
args = parser.parse_args()


def _resolve_inference_mode(parser, args):
    if args.comparison_mode and args.generation_mode:
        parser.error("--comparison_mode and --generation_mode cannot be used together")

    if args.comparison_mode and args.stochastic_sampling:
        parser.error("--comparison_mode conflicts with --stochastic_sampling")

    if args.generation_mode or args.stochastic_sampling:
        return "generation", False

    return "comparison", True


inference_mode, deterministic_sampling = _resolve_inference_mode(parser, args)

# Optionally force wandb to offline to avoid network calls on login nodes
if args.wandb_offline:
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_API_KEY'] = ''

# Configure visible CUDA devices before creating Trainer/Accelerator
if args.device == 'cpu':
    # Hide GPUs so torch/accelerate will pick CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device_type = 'cpu'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    device_type = 'cuda'

cfg = OmegaConf.load(os.path.join(args.config_path, "config.yaml"))
original_sampling_steps = []

# For inference comparisons, keep only the first occurrence of each sampling step
# so repeated entries like [32, 32, 64, 64] become [32, 64].
if getattr(getattr(cfg, 'diffusion', None), 'sampling_steps', None):
    sampling_steps = list(cfg.diffusion.sampling_steps)
    original_sampling_steps = sampling_steps.copy()
    unique_sampling_steps = list(dict.fromkeys(sampling_steps))
    if unique_sampling_steps != sampling_steps:
        print(f"[inference] using unique sampling_steps for comparison: {unique_sampling_steps} (from {sampling_steps})")
        cfg.diffusion.sampling_steps = unique_sampling_steps

effective_sampling_steps = list(getattr(getattr(cfg, 'diffusion', None), 'sampling_steps', []))

# Allow overriding config.load_weights from CLI
if args.force_load_weights:
    cfg.load_weights = True

# Override CFG guidance scale from CLI if provided
if args.cfg_scale is not None:
    if not OmegaConf.select(cfg, 'image_cond'):
        OmegaConf.update(cfg, 'image_cond', {})
    OmegaConf.update(cfg, 'image_cond.cfg_guidance_scale', args.cfg_scale)


def _load_conditioning_image(image_path: str, device) -> torch.Tensor | None:
    """
    Load a 2D conditioning image from disk and return a (1, 1, H, W) float32
    tensor on the given device, normalised to [0, 1].

    Supports: PNG / JPG / BMP (via PIL), TIF/TIFF (via PIL), .npy/.npz arrays.
    Grayscale and RGB images are both accepted; RGB is averaged to grayscale.
    """
    if image_path is None:
        return None
    import pathlib
    path = pathlib.Path(image_path)
    ext = path.suffix.lower()
    if ext in ('.npy',):
        arr = np.load(str(path)).astype(np.float32)
    elif ext in ('.npz',):
        d = np.load(str(path))
        arr = d[list(d.keys())[0]].astype(np.float32)
    else:
        from PIL import Image as PILImage
        pil = PILImage.open(str(path))
        arr = np.array(pil).astype(np.float32)

    # Normalise to [0, 1]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()

    # Ensure shape is (H, W) → (1, 1, H, W)
    if arr.ndim == 3:         # (H, W, C) RGB → grayscale
        arr = arr.mean(axis=2)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    return tensor.to(device)


def _sanitize_name_component(value, fallback="sample"):
    value = str(value or "").strip()
    if not value:
        return fallback
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or fallback


def _get_organelle_name(cfg):
    shapenet_ids = getattr(getattr(cfg, 'dataset', None), 'shapenet_ids', None)
    if shapenet_ids:
        return _sanitize_name_component(shapenet_ids[0], fallback="sample")
    return _sanitize_name_component(getattr(cfg, 'name', None), fallback="sample")


def _get_default_output_subdir(cfg, config_path):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = _sanitize_name_component(
        getattr(cfg, 'name', None) or os.path.basename(os.path.normpath(config_path)),
        fallback="run"
    )
    organelle_name = _get_organelle_name(cfg)

    if organelle_name.lower() in run_name.lower():
        return f"inference_{run_name}_{timestamp}"

    return f"inference_{organelle_name}_{run_name}_{timestamp}"


def _build_inference_manifest(cfg, args, output_dir, original_steps, effective_steps):
    organelle_name = _get_organelle_name(cfg)
    run_name = getattr(cfg, 'name', None) or os.path.basename(os.path.normpath(args.config_path))
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
        "filename_step_label_note": "For backwards compatibility filenames still use 'stepsize_<n>', but <n> is the number of diffusion sampling steps.",
        "comparison_note": "All effective sampling step counts start from the same initial latent noise within each generated sample index.",
        "deterministic_note": "When deterministic_reverse_sampling is true, intermediate reverse-process noise injection is disabled for stricter paired comparisons across different numbers of sampling steps.",
    }


def _write_inference_manifest(output_dir, manifest):
    manifest_path = os.path.join(output_dir, "inference_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[inference] wrote comparison manifest to: {manifest_path}")


# Initialize the trainer
trainer = Trainer(
    train_batch_size=cfg.training.batch_size,
    save_and_sample_every=cfg.training.test_every,
    results_folder=cfg.results_folder,
    config_folder=args.config_path,
    num_samples=1,
    train_lr=cfg.training.lr,
    train_num_steps=cfg.training.num_steps,  # total training steps
    gradient_accumulate_every=cfg.training.ga,  # gradient accumulation steps
    ema_decay=cfg.training.ema_decay,  # exponential moving average decay
    cfg=cfg,
    inference=True
)


def generate_meshes(trainer, num_images=1000, batch_size=1, device_type="cuda",
                    cond_image=None):
    """
    Generates meshes using the trainer model and saves them as OBJ files.

    Args:
        trainer:     Trainer object.
        num_images:  Number of meshes to generate.
        batch_size:  Batch size per sampling call.
        device_type: 'cuda' or 'cpu' (for autocast).
        cond_image:  Optional (1, 1, H, W) conditioning image tensor.
                     Pass None for unconditional generation.
    """
    acc = trainer.accelerator

    if not acc.is_main_process:
        print("Not main process: skipping mesh generation on this rank.")
        return

    sampling_model = None
    if hasattr(trainer, 'ema') and getattr(trainer.ema, 'ema_model', None) is not None:
        sampling_model = trainer.ema.ema_model
    else:
        if hasattr(acc, 'unwrap_model'):
            sampling_model = acc.unwrap_model(trainer.model)
        else:
            sampling_model = getattr(trainer.model, 'module', trainer.model)

    sampling_model.eval()
    organelle_prefix = _get_organelle_name(trainer.cfg)

    for k in tqdm(range(num_images), desc="Generating meshes"):
        with torch.inference_mode():
            if device_type == 'cuda':
                with torch.autocast(device_type=device_type):
                    all_images_list = list(sampling_model.sample(
                        batch_size=batch_size,
                        deterministic=deterministic_sampling,
                        image=cond_image,
                    ))
            else:
                all_images_list = list(sampling_model.sample(
                    batch_size=batch_size,
                    deterministic=deterministic_sampling,
                    image=cond_image,
                ))

            all_images = torch.stack(all_images_list, dim=0)
            plot_and_save_meshes(all_images, trainer.ds, trainer.cfg, output_dir, k,
                                 file_prefix=organelle_prefix)


# Generate images
# Prepare output directory: priority --out_dir > cfg.results_folder/out_subdir > cfg.results_folder
if args.out_dir:
    output_dir = args.out_dir
else:
    if args.out_subdir:
        sub = args.out_subdir
    else:
        sub = _get_default_output_subdir(cfg, args.config_path)
    output_dir = os.path.join(cfg.results_folder, sub)

os.makedirs(output_dir, exist_ok=True)

print(f"[inference] writing outputs to: {output_dir}")
print(f"[inference] comparison sampling_steps: {effective_sampling_steps}")
print(f"[inference] inference mode: {inference_mode}")
print(f"[inference] deterministic reverse sampling: {deterministic_sampling}")

# Load conditioning image (only meaningful for image-conditioned models)
cond_image = _load_conditioning_image(args.image_path, device=trainer.accelerator.device)
if cond_image is not None:
    guidance = getattr(getattr(cfg, 'image_cond', None), 'cfg_guidance_scale', 1.0)
    print(f"[inference] conditioning image: {args.image_path}  (CFG scale={guidance})")
else:
    if getattr(getattr(cfg, 'image_cond', None), 'enabled', False):
        print("[inference] image_cond enabled but no --image_path given → null embedding (unconditional).")

manifest = _build_inference_manifest(
    cfg, args, output_dir, original_sampling_steps, effective_sampling_steps,
)
_write_inference_manifest(output_dir, manifest)

generate_meshes(trainer, num_images=args.num_images, device_type=device_type,
                cond_image=cond_image)
