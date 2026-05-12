from pathlib import Path
import os
import time
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate.local_sgd import LocalSGD
from accelerate import Accelerator
import numpy as np
from glob import glob
from omegaconf import OmegaConf
from lib.ops.Misc import *
from lib.ops.Utils import plot_and_save_meshes
from lib.DDPM import GaussianDiffusion
from lib.Tetradata import MeshLoader
from lib.UVIT import UVIT
import wandb


class Trainer(object):
    def __init__(
            self,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=5,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            config_folder = "./config/",
            results_folder='./results',
            split_batches=True,
            cfg=None,
            inference=False
    ):
        super().__init__()
        import wandb
        import random

        self.cfg = cfg
        self.config_folder = config_folder
        self.inference = inference
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if self.cfg.training.mixed_precision else 'no',
        )

        # ------------------------------------------------------------------
        # wandb init — resume the same run if a saved run-id exists
        # ------------------------------------------------------------------
        wandb_id_file = os.path.join(config_folder, "wandb_run_id.txt")
        wandb_resume = "never"
        wandb_id = None
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file) as f:
                wandb_id = f.read().strip()
            wandb_resume = "must"

        run = wandb.init(
            project=getattr(cfg, 'wandb_project', 'TetraDiffusion'),
            name=getattr(cfg, 'name', None),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=wandb_resume,
            id=wandb_id,
        )

        # persist run id so future resumes reconnect to the same wandb run
        if self.accelerator.is_main_process and not os.path.exists(wandb_id_file):
            os.makedirs(config_folder, exist_ok=True)
            with open(wandb_id_file, "w") as f:
                f.write(run.id)

        if self.inference:
            self.ds = torch.load(os.path.join(config_folder, "ds.pth"), weights_only=False)
            self.ds.config = self.cfg
        else:
            print("[Trainer] Initializing MeshLoader (this may take a while if grid pruning is enabled)...")
            self.ds = MeshLoader(config=cfg, device="cpu", cuda_device=self.device, accelerator=self.accelerator)
            print("[Trainer] MeshLoader initialization complete.")
            torch.save(self.ds, config_folder + "/ds.pth")

        print("mixed_precision", 'fp16' if self.cfg.training.mixed_precision else 'no')
        model = UVIT(cfg, rank=self.device, ds=self.ds)
        print("[Trainer] UVIT model created.")

        # log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Trainer] Model params: {total_params:,} total, {trainable_params:,} trainable")
        if self.accelerator.is_main_process:
            wandb.summary["total_params"] = total_params
            wandb.summary["trainable_params"] = trainable_params

        wandb.watch(model, log_freq=200)

        if cfg.load_weights:
            all_weights = glob(config_folder + "/*.pt")
            latest = max(all_weights, key=os.path.getctime)
            print("loading model", latest)
            data = torch.load(latest, map_location="cpu", weights_only=False)
            checkpoint = data['model']
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('model.', '')] = checkpoint[key]
                del checkpoint[key]
            model.load_state_dict(checkpoint, strict=False)

        num_verts = len(self.ds.tet_verts)
        channels = 4 + (3 if cfg.dataset.color else 0)

        diffusion = GaussianDiffusion(
            model,
            num_verts=num_verts,
            channels=channels,
            image_size=self.cfg.dataset.grid_res,
            noise_d=self.cfg.diffusion.noise_d,
            cfg=self.cfg,
            pred_objective=self.cfg.diffusion.pred_objective,
            num_sample_steps=self.cfg.diffusion.sampling_steps,
            offset_noise_strength=self.cfg.diffusion.offset_noise
        )
        self.model = diffusion
        self.diffusion = diffusion

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # dataset and dataloader
        print(f"[Trainer] Creating DataLoader with batch_size={train_batch_size}, num_workers={cfg.num_workers}")
        dl = DataLoader(self.ds, batch_size=train_batch_size, num_workers=cfg.num_workers,
                        shuffle=True, pin_memory=True, persistent_workers=True)
        print("[Trainer] DataLoader created.")
        self.model.mask = self.ds.mask_verts

        optim_klass = AdamW

        if cfg.training.use_scheduler:
            div_factor = cfg.training.max_lr / cfg.training.start_lr
            final_div_factor = cfg.training.max_lr / (cfg.training.min_lr * div_factor)
            self.opt = optim_klass(self.model.parameters(), lr=train_lr, betas=adam_betas)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.opt, max_lr=cfg.training.max_lr, steps_per_epoch=1,
                epochs=cfg.training.num_steps, div_factor=div_factor,
                final_div_factor=final_div_factor)
        else:
            self.opt = optim_klass(self.model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=1e-4)

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)
            if cfg.load_weights:
                print("loading ema")
                try:
                    all_weights = glob(config_folder + "/*.pt")
                    latest = max(all_weights, key=os.path.getctime)
                    data = torch.load(latest, map_location="cpu", weights_only=False)
                    # restore step counter so wandb x-axis continues correctly
                    self.step = data.get('step', 0)
                    checkpoint = data["ema"]
                    self.ema.load_state_dict(checkpoint, strict=False)
                    print(f"success — resuming from step {self.step}")
                except Exception as e:
                    print(f"ema loading failed: {e}")
                    self.step = 0
            else:
                self.step = 0
        else:
            self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        if cfg.training.use_scheduler:
            self.model, self.opt, self.scheduler, dl = self.accelerator.prepare(
                self.model, self.opt, scheduler, dl)
        else:
            self.model, self.opt, dl = self.accelerator.prepare(self.model, self.opt, dl)
        self.dl = cycle(dl)
        print("[Trainer] Trainer initialization complete. Ready to start training.")

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }
        torch.save(data, str(self.config_folder + f'/model-{milestone}.pt'))

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return np.format_float_scientific(param_group['lr'], precision=1)

    def track_and_plot_exploding_gradients(self, clip_norm=1.0):
        exploding_gradients = {}
        for name, param in self.model.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if grad_norm > clip_norm:
                    exploding_gradients[name] = grad_norm
        if exploding_gradients:
            print(f"Exploding Gradients Detected (Threshold: {clip_norm}):")
            for name, grad_norm in exploding_gradients.items():
                print(f"Layer: {name}, Gradient Norm: {grad_norm}")

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(str(self.config_folder / f'model-{milestone}.pt'), map_location=device, weights_only=False)
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            checkpoint = data["ema"]
            self.ema.load_state_dict(checkpoint)
        if self.cfg.training.use_scheduler:
            self.scheduler = data["scheduler"]
        else:
            for g in self.opt.param_groups:
                g['lr'] = self.cfg.training.lr
        if 'version' in data:
            print(f"loading from version {data['version']}")
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        losses = []
        recent_step_times = []
        step_start = time.time()

        with tqdm(initial=self.step, total=self.train_num_steps,
                  disable=not accelerator.is_main_process) as pbar:
            with LocalSGD(accelerator=accelerator, model=self.model,
                          local_sgd_steps=8, enabled=self.cfg.use_local_sgd) as local_sgd:
                for index, data in enumerate(self.dl):
                    with accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            loss = self.model(data)
                            losses.append(loss.item())
                        self.accelerator.backward(loss)

                        grad_norm = None
                        if self.accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.model.parameters(), 1.0)
                            if hasattr(grad_norm, 'item'):
                                grad_norm = grad_norm.item()

                        self.opt.step()
                        self.opt.zero_grad()

                    local_sgd.step()
                    self.accelerator.log({"training_loss": loss})

                    if self.cfg.training.use_scheduler:
                        self.scheduler.step()

                    if index % self.gradient_accumulate_every == 0:
                        self.step += 1
                        pbar.update(1)

                        # --- timing ---
                        now = time.time()
                        recent_step_times.append(now - step_start)
                        if len(recent_step_times) > 100:
                            recent_step_times.pop(0)
                        step_start = now
                        steps_per_sec = 1.0 / np.mean(recent_step_times)

                        mean_loss = float(np.mean(losses))
                        current_lr = float(self.opt.param_groups[0]['lr'])
                        losses = []

                        if accelerator.is_main_process:
                            log_dict = {
                                "loss": mean_loss,
                                "lr": current_lr,
                                "steps_per_sec": steps_per_sec,
                                "step": self.step,
                            }
                            if grad_norm is not None:
                                log_dict["grad_norm"] = grad_norm
                            wandb.log(log_dict, step=self.step)

                        pbar.set_description(
                            f'loss: {mean_loss:.4f} | '
                            f'lr: {current_lr:.2e} | '
                            f'grad: {grad_norm:.3f}' if grad_norm is not None else
                            f'loss: {mean_loss:.4f} | lr: {current_lr:.2e}'
                        )

                        if accelerator.is_main_process:
                            self.ema.update()

                            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                                self.ema.ema_model.eval()
                                self.ema.eval()
                                self.model.eval()
                                with torch.no_grad():
                                    milestone = self.step // self.save_and_sample_every
                                    batches = num_to_groups(self.num_samples, self.batch_size)
                                    all_images_list = list(map(
                                        lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                                all_images = torch.cat(all_images_list, dim=0)
                                try:
                                    plot_and_save_meshes(
                                        all_images, self.ds, self.cfg,
                                        self.results_folder, milestone)
                                except Exception as e:
                                    print(f"could not generate mesh: {e}")
                                self.save(milestone % 2)
                                self.ema.ema_model.train()
                                self.ema.train()
                                self.model.train()

                    if self.step >= self.train_num_steps:
                        break

        accelerator.print('training complete')