from pathlib import Path
import os
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate.local_sgd import LocalSGD
from accelerate import Accelerator
import numpy as np
from glob import glob
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

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="debug_accumulate",

        )
        self.cfg = cfg
        self.config_folder = config_folder
        self.inference = inference
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if self.cfg.training.mixed_precision else 'no',
            #gradient_accumulation_steps=gradient_accumulate_every
        )

        if self.inference:
            self.ds = torch.load(os.path.join(config_folder, "ds.pth"))
            self.ds.config = self.cfg
        else:
            self.ds = MeshLoader(config=cfg, device="cpu", cuda_device=self.device,accelerator=self.accelerator )
            torch.save(self.ds, config_folder + "/ds.pth")



        print("mixed_precision", 'fp16' if self.cfg.training.mixed_precision else 'no')
        model = UVIT(cfg, rank=self.device, ds=self.ds)
        wandb.watch(model,log_freq=10)
        if cfg.load_weights:
            all_weights = glob(config_folder+ "/*.pt")
            latest = max(all_weights, key=os.path.getctime)

            print("loading model", latest)
            data = torch.load(latest, map_location="cpu")
            checkpoint = data['model']
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('model.', '')] = checkpoint[key]
                del checkpoint[key]

            model.load_state_dict(checkpoint,strict=False)

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
        dl = DataLoader(self.ds, batch_size=train_batch_size, num_workers=cfg.num_workers, shuffle=True,pin_memory=True, persistent_workers=True)

        self.model.mask = self.ds.mask_verts

        optim_klass = AdamW

        if cfg.training.use_scheduler:
            div_factor = cfg.training.max_lr / cfg.training.start_lr
            final_div_factor = cfg.training.max_lr / (cfg.training.min_lr * div_factor)
            self.opt = optim_klass(self.model.parameters(), lr=train_lr, betas=adam_betas)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=cfg.training.max_lr, steps_per_epoch=1,
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
                    data = torch.load(latest, map_location="cpu")
                    checkpoint = data["ema"]
                    self.ema.load_state_dict(checkpoint,strict=False)
                    print("success")
                except:
                    print("ema loading failed")
                    pass

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        if cfg.training.use_scheduler:
            self.model, self.opt, self.scheduler, dl = self.accelerator.prepare(self.model, self.opt, scheduler, dl)
        else:
            self.model, self.opt, dl = self.accelerator.prepare(self.model, self.opt, dl)
        self.dl = cycle(dl)

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
            #'scheduler': self.scheduler.state_dict() if exists(self.scheduler) else None
        }

        torch.save(data, str(self.config_folder + f'/model-{milestone}.pt'))

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return np.format_float_scientific(param_group['lr'], precision=1)

    def track_and_plot_exploding_gradients(self, clip_norm=1.0):
        """
        Tracks gradients and plots them if they are "exploding" (exceed the clip norm).

        Args:
            model (torch.nn.Module): The model whose gradients are being tracked.
            clip_norm (float): The threshold for gradient norm clipping.
        """
        exploding_gradients = {}

        for name, param in self.model.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if grad_norm > clip_norm:  # Check if the gradient norm exceeds the threshold
                    exploding_gradients[name] = grad_norm

        # If there are exploding gradients, plot them
        if exploding_gradients:
            print(f"Exploding Gradients Detected (Threshold: {clip_norm}):")
            for name, grad_norm in exploding_gradients.items():
                print(f"Layer: {name}, Gradient Norm: {grad_norm}")
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.config_folder / f'model-{milestone}.pt'), map_location=device)

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
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            with LocalSGD(accelerator=accelerator, model=self.model, local_sgd_steps=8, enabled=self.cfg.use_local_sgd) as local_sgd:
                for index, data in enumerate(self.dl):
                    with accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            loss = self.model(data)
                            losses.append(loss.item())
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                           accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.opt.zero_grad()

                    local_sgd.step()
                    self.accelerator.log({"training_loss": loss})

                    if self.cfg.training.use_scheduler:
                        self.scheduler.step()

                    if index % self.gradient_accumulate_every == 0:
                        self.step += 1
                        pbar.update(1)
                        wandb.log({"loss": np.mean(losses)})

                        pbar.set_description(f'loss: {np.mean(losses):.4f}, LR: {self.get_lr(self.opt)}')
                        losses = []
                        if accelerator.is_main_process:
                            self.ema.update()

                            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                                self.ema.ema_model.eval()
                                self.ema.ema_model.eval()
                                self.ema.eval()
                                self.model.eval()
                                with torch.no_grad():
                                    milestone = self.step // self.save_and_sample_every
                                    batches = num_to_groups(self.num_samples, self.batch_size)
                                    all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                                all_images = torch.cat(all_images_list, dim=0)
                                try:
                                    plot_and_save_meshes(all_images, self.ds, self.cfg,self.results_folder,milestone)
                                except:
                                    print("could not generate mesh")
                                    pass
                                self.save(milestone % 2)
                                self.ema.ema_model.train()
                                self.ema.train()
                                self.model.train()
                    if self.step == self.train_num_steps:
                        break

        accelerator.print('training complete')
