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
from lib.ops.Utils import plot_and_save_meshes, log_training_samples_to_wandb, meshes_to_wandb_point_clouds
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
        # wandb init — only resume the same run when explicitly resuming
        # training (--resume flag → cfg.load_weights=True).
        # Fresh runs always create a new wandb run so deleted/stale runs
        # are never accidentally reused.
        # ------------------------------------------------------------------
        wandb_id_file = os.path.join(config_folder, "wandb_run_id.txt")
        if cfg.load_weights and os.path.exists(wandb_id_file):
            with open(wandb_id_file) as f:
                wandb_id = f.read().strip()
            wandb_resume = "allow"
            print(f"[Trainer] WandB: resuming run id={wandb_id}")
        else:
            wandb_id = None
            wandb_resume = "never"
            print("[Trainer] WandB: starting new run")

        run = wandb.init(
            project=getattr(cfg, 'wandb_project', 'TetraDiffusion'),
            name=getattr(cfg, 'name', None),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=wandb_resume,
            id=wandb_id,
            settings=wandb.Settings(init_timeout=300),
        )

        # Only persist the run id on fresh starts.  During resume we never
        # overwrite the file — if wandb silently creates a new run instead
        # of resuming, the original id is preserved for future retries.
        if self.accelerator.is_main_process and wandb_resume == "never":
            os.makedirs(config_folder, exist_ok=True)
            with open(wandb_id_file, "w") as f:
                f.write(run.id)
            print(f"[Trainer] WandB: saved run id={run.id} to {wandb_id_file}")

        if self.inference:
            self.ds = torch.load(os.path.join(config_folder, "ds.pth"), weights_only=False)
            self.ds.config = self.cfg
        else:
            # ── Dataset cache: keyed by category + grid_res so it is reused
            #    across runs of the same category without rerunning GridPruning.
            category_key = "_".join(sorted(cfg.dataset.shapenet_ids))
            ds_cache_dir = os.path.join(cfg.data_path, "ds_cache")
            ds_cache_path = os.path.join(ds_cache_dir, f"{category_key}_res{cfg.dataset.grid_res}.pth")

            if os.path.exists(ds_cache_path):
                print(f"[Trainer] Loading cached MeshLoader from {ds_cache_path}")
                try:
                    self.ds = torch.load(ds_cache_path, weights_only=False)
                    self.ds.config = cfg
                    print("[Trainer] Cached MeshLoader loaded.")
                except Exception as e:
                    print(f"[Trainer] WARNING: Failed to load MeshLoader cache "
                          f"({type(e).__name__}: {e}). "
                          f"This is usually a pandas/numpy version mismatch. "
                          f"Deleting stale cache and rebuilding…")
                    try:
                        os.remove(ds_cache_path)
                    except OSError:
                        pass
                    self.ds = None  # fall through to rebuild below
            else:
                self.ds = None

            if self.ds is None:
                print("[Trainer] Initializing MeshLoader (this may take a while if grid pruning is enabled)...")
                self.ds = MeshLoader(config=cfg, device="cpu", cuda_device=self.device, accelerator=self.accelerator)
                print("[Trainer] MeshLoader initialization complete.")
                # Save to category-level cache
                if self.accelerator.is_main_process:
                    os.makedirs(ds_cache_dir, exist_ok=True)
                    torch.save(self.ds, ds_cache_path)
                    print(f"[Trainer] MeshLoader cached to {ds_cache_path}")

            # Also save to run folder (required for inference)
            torch.save(self.ds, config_folder + "/ds.pth")

            # Log a few real training samples as point clouds so you can
            # verify the data pipeline looks correct before training starts.
            if self.accelerator.is_main_process and not cfg.load_weights:
                print("[Trainer] Logging training sample point clouds to WandB …")
                # Use the current wandb run step so step ordering is always
                # monotonically increasing, even when resuming a run.
                current_step = getattr(wandb.run, 'step', 0) if wandb.run else 0
                log_training_samples_to_wandb(self.ds, n_samples=4, step=current_step)

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

        # watch=None: disable automatic gradient logging which calls wandb.log()
        # internally and drifts the step counter ahead of self.step.
        wandb.watch(model, log=None)

        # Model loading is deferred to the end of __init__ to consolidate loading of weights,
        # EMA, optimizer, and scheduler states from a single torch.load call post-prepare().

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
            offset_noise_strength=self.cfg.diffusion.offset_noise,
            ds=self.ds,
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
        self.opt = optim_klass(self.model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=1e-4)

        # ── LR scheduler ────────────────────────────────────────────────────
        # Controlled by cfg.training.lr_schedule.  Four named options:
        #   warmup_constant  ramp 0→lr over warmup_steps, then flat  (default)
        #   constant         flat from step 0 (no warmup)
        #   warmup_cosine    ramp 0→lr, then cosine decay to lr_min
        #   cosine           cosine decay from step 0 (no warmup)
        lr_schedule  = str(getattr(cfg.training, 'lr_schedule', 'warmup_constant'))
        warmup_steps = int(getattr(cfg.training, 'warmup_steps', 2000))
        total_steps  = cfg.training.num_steps
        lr_min       = float(getattr(cfg.training, 'lr_min', 1e-6))
        lr_min_ratio = lr_min / train_lr   # fraction for cosine floor

        if lr_schedule == 'constant':
            self.scheduler = None
            print(f"[Trainer] LR schedule: constant {train_lr:.2e}")

        elif lr_schedule == 'warmup_constant':
            def _warmup_constant(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, _warmup_constant)
            print(f"[Trainer] LR schedule: warmup_constant  "
                  f"(0 → {train_lr:.2e} over {warmup_steps} steps, then flat)")

        elif lr_schedule == 'warmup_cosine':
            import math as _math
            def _warmup_cosine(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, _warmup_cosine)
            print(f"[Trainer] LR schedule: warmup_cosine  "
                  f"(warmup {warmup_steps} steps, cosine → {lr_min:.2e})")

        elif lr_schedule == 'cosine':
            import math as _math
            def _cosine(step):
                progress = float(step) / float(max(1, total_steps))
                cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, _cosine)
            print(f"[Trainer] LR schedule: cosine  "
                  f"({train_lr:.2e} → {lr_min:.2e} over {total_steps} steps)")

        else:
            raise ValueError(
                f"Unknown lr_schedule '{lr_schedule}'. "
                "Choose one of: warmup_constant, constant, warmup_cosine, cosine"
            )

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)
            self.step = 0
        else:
            self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        if self.scheduler is not None:
            self.model, self.opt, self.scheduler, dl = self.accelerator.prepare(
                self.model, self.opt, self.scheduler, dl)
        else:
            self.model, self.opt, dl = self.accelerator.prepare(self.model, self.opt, dl)
        self.dl = cycle(dl)

        # Consolidated loader: restores UVIT weights, EMA weights, optimizer, and
        # scheduler state from a single file load post-prepare() to minimize VRAM usage.
        if cfg.load_weights:
            import gc
            try:
                all_weights = glob(config_folder + "/*.pt")
                latest = max(all_weights, key=os.path.getctime)
                print(f"[Trainer] Loading consolidated checkpoint from {latest}")
                data = torch.load(latest, map_location="cpu", weights_only=False)

                # 1. Restore UVIT model weights
                checkpoint = data['model']
                for key in list(checkpoint.keys()):
                    checkpoint[key.replace('model.', '')] = checkpoint[key]
                    del checkpoint[key]
                raw_model = self.accelerator.unwrap_model(self.model)
                raw_model.model.load_state_dict(checkpoint, strict=False)
                print("[Trainer] Model state restored.")

                # 2. Restore EMA (rank 0 only)
                if self.accelerator.is_main_process:
                    self.step = data.get('step', 0)
                    checkpoint_ema = data["ema"]
                    self.ema.load_state_dict(checkpoint_ema, strict=False)
                    print("[Trainer] EMA state restored.")

                # 3. Restore Optimizer state
                if "opt" in data:
                    self.opt.load_state_dict(data["opt"])
                    print("[Trainer] Optimizer state restored.")

                # 4. Restore Scheduler state
                if self.scheduler is not None and "scheduler" in data:
                    self.scheduler.load_state_dict(data["scheduler"])
                    print(f"[Trainer] LR scheduler state restored (last_epoch={data['scheduler'].get('last_epoch', '?')}).")

                print(f"[Trainer] Success — resumed from step {self.step}")
                
                # Delete temporary dictionary and collect garbage to free CPU memory
                del data
                gc.collect()
            except Exception as e:
                print(f"[Trainer] Resume failed: {e}")
                self.step = 0

        # Broadcast self.step to non-main ranks so all processes start at the
        # same step (matters for multi-GPU runs with accelerate).
        import torch.distributed as _dist
        if self.accelerator.num_processes > 1 and _dist.is_initialized():
            _step_t = torch.tensor(self.step, dtype=torch.long, device=self.device)
            _dist.broadcast(_step_t, src=0)
            self.step = int(_step_t.item())

        # Clean CUDA memory cache before we start dataset loading / training loops
        torch.cuda.empty_cache()

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
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()
        pt_path = str(self.config_folder + f'/model-{milestone}.pt')
        torch.save(data, pt_path)
        # Write a lightweight sidecar so checkpoints can be inspected
        # (step, timestamp) without loading the full tensor file.
        import json as _json, datetime as _dt
        meta = {'step': self.step, 'milestone': milestone,
                'saved_at': _dt.datetime.now().isoformat(timespec='seconds')}
        with open(pt_path.replace('.pt', '.json'), 'w') as _f:
            _json.dump(meta, _f)

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
        if self.scheduler is not None and "scheduler" in data:
            self.scheduler.load_state_dict(data["scheduler"])
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
                            # Unpack (mesh_data, image) when image conditioning
                            # is enabled; fall back to unconditional otherwise.
                            if isinstance(data, (list, tuple)):
                                mesh_data, cond_image = data[0], data[1]
                            else:
                                mesh_data, cond_image = data, None
                            loss = self.model(mesh_data, image=cond_image)
                            losses.append(loss.item())
                        self.accelerator.backward(loss)

                        # Guard: skip optimizer step if loss is non-finite.
                        # A single NaN/inf step permanently corrupts Adam's m/v
                        # moment buffers, making recovery impossible.
                        if not torch.isfinite(loss):
                            if accelerator.is_main_process:
                                print(f"[Trainer] WARNING: non-finite loss "
                                      f"{loss.item()} at step {self.step} — "
                                      f"skipping optimizer step to protect Adam state")
                            self.opt.zero_grad()
                            continue

                        grad_norm = None
                        if self.accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.model.parameters(), 1.0)
                            if hasattr(grad_norm, 'item'):
                                grad_norm = grad_norm.item()

                        self.opt.step()
                        self.opt.zero_grad()

                    local_sgd.step()
                    # NOTE: do NOT call self.accelerator.log() here — it commits
                    # a wandb step without an explicit step number, causing the
                    # internal counter to drift ahead of self.step and producing
                    # "step X < current step Y" warnings.  Loss is logged below
                    # in log_dict with the correct step=self.step.

                    if self.scheduler is not None:
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
                            # Log bio-constraint sub-losses if active
                            raw_model = accelerator.unwrap_model(self.model)
                            if hasattr(raw_model, '_last_bio_loss'):
                                log_dict["bio_loss"] = raw_model._last_bio_loss.item()
                                log_dict["diffusion_loss"] = raw_model._last_diffusion_loss.item()
                            if hasattr(raw_model, '_last_sdf_bg_loss'):
                                log_dict["sdf_bg_loss"] = raw_model._last_sdf_bg_loss.item()
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
                                    saved_paths = plot_and_save_meshes(
                                        all_images, self.ds, self.cfg,
                                        self.results_folder, milestone)
                                    # Log generated shapes as point clouds in WandB
                                    pc_panels = meshes_to_wandb_point_clouds(
                                        all_images, self.ds, self.cfg,
                                        prefix="generated",
                                    )
                                    if pc_panels:
                                        wandb.log(pc_panels, step=self.step)
                                except Exception as e:
                                    print(f"could not generate mesh: {e}")
                                self.save(milestone % 2)
                                self.ema.ema_model.train()
                                self.ema.train()
                                self.model.train()

                    if self.step >= self.train_num_steps:
                        break

        accelerator.print('training complete')