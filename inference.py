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
args = parser.parse_args()

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

# Allow overriding config.load_weights from CLI
if args.force_load_weights:
    cfg.load_weights = True


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


def generate_meshes(trainer, num_images=1000, batch_size=1, device_type="cuda"):
    """
    Generates images using the trainer model and saves them as mesh objects.

    Args:
        trainer (Trainer): Trainer object containing the model, dataset, and configuration.
        num_images (int): Number of images to generate. Default is 1000.
        batch_size (int): Batch size for image generation. Default is 1.
        device_type (str): Device type for torch.autocast. Default is "cuda".
    """
    trainer.ema.ema_model.eval()
    trainer.ema.eval()
    trainer.model.eval()

    for k in tqdm(range(num_images), desc="Generating meshes"):
        with torch.inference_mode():
            # Use autocast only for CUDA; CPU autocast may not be available/desired
            if device_type == 'cuda':
                with torch.autocast(device_type=device_type):
                    all_images_list = list(trainer.model.sample(batch_size=batch_size))
            else:
                all_images_list = list(trainer.model.sample(batch_size=batch_size))

            all_images = torch.stack(all_images_list, dim=0)
            plot_and_save_meshes(all_images, trainer.ds, trainer.cfg, cfg.results_folder, k)


# Generate images
generate_meshes(trainer, num_images=args.num_images, device_type=device_type)
