from functools import wraps
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from einops import rearrange, repeat, reduce
import numpy as np
import torch
from torch import Tensor, einsum, nn, sqrt
from torch.amp import autocast
import torch.nn.functional as F
from torch.special import expm1
from tqdm import tqdm

from lib.ops.BioConstraints import biological_constraints_loss, organelle_loss_registry
from lib.ops.Misc import default, exists


def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: Tensor) -> Tensor:
    return (t + 1) * 0.5


def right_pad_dims_to(x: Tensor, t: Tensor) -> Tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def log(t: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t: Tensor, logsnr_min: float = -15, logsnr_max: float = 15) -> Tensor:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))



def logsnr_schedule_shifted(fn: Callable, image_d: int, noise_d: int) -> Callable:
    """Shift log-SNR schedule to adjust noise levels for different grid resolutions."""
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs) + shift

    return inner


def logsnr_schedule_interpolated(
    fn: Callable, image_d: int, noise_d_low: int, noise_d_high: int
) -> Callable:
    """Interpolate log-SNR schedules between low and high resolution bounds."""
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion process manager executing forward noise corruption and reverse sampling."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_verts: int,
        channels: int = 3,
        pred_objective: str = "v",
        noise_schedule: Callable = logsnr_schedule_cosine,
        noise_d: Optional[int] = 32,
        noise_d_low: Optional[int] = None,
        noise_d_high: Optional[int] = None,
        num_sample_steps: int = 32,
        clip_sample_denoised: bool = True,
        min_snr_loss_weight: bool = True,
        min_snr_gamma: int = 5,
        image_size: int = 128,
        cfg: Any = None,
        offset_noise_strength: float = 0.1,
        ds: Any = None,
    ):
        super().__init__()
        assert pred_objective in {"v", "eps"}, 'Prediction objective must be "v" or "eps".'

        self.model = model
        self.cfg = cfg
        self.channels = channels
        self.num_verts = num_verts
        self.offset_noise_strength = offset_noise_strength
        self.pred_objective = pred_objective

        self.log_snr = noise_schedule
        if not getattr(getattr(cfg, "diffusion", None), "use_standard_noise", False):
            if exists(noise_d):
                self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)
            elif exists(noise_d_low) and exists(noise_d_high):
                self.log_snr = logsnr_schedule_interpolated(
                    self.log_snr, image_size, noise_d_low, noise_d_high
                )

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # Image conditioning configuration
        img_cond_cfg = getattr(cfg, "image_cond", None)
        self.use_image_cond = (
            img_cond_cfg is not None and bool(getattr(img_cond_cfg, "enabled", False))
        )
        self.cfg_dropout_prob = (
            float(getattr(img_cond_cfg, "cfg_dropout_prob", 0.1))
            if self.use_image_cond
            else 0.0
        )
        self.cfg_guidance_scale = (
            float(getattr(img_cond_cfg, "cfg_guidance_scale", 1.0))
            if self.use_image_cond
            else 1.0
        )

        # Biological constraint losses configuration
        bio_cfg = getattr(cfg, "diffusion", None)
        self.bio_loss_weight = float(getattr(bio_cfg, "bio_loss_weight", 0.0))
        self.bio_loss_type = str(getattr(bio_cfg, "bio_loss_type", "both"))
        self.bio_curvature_softness = float(getattr(bio_cfg, "bio_curvature_softness", 0.15))
        self.bio_eikonal_weight = float(getattr(bio_cfg, "bio_eikonal_weight", 0.0))
        self.sdf_bg_loss_weight = float(getattr(bio_cfg, "sdf_bg_loss_weight", 0.0))
        self.sdf_bg_threshold = float(getattr(bio_cfg, "sdf_bg_threshold", 0.3))

        self.snr_gate = str(getattr(bio_cfg, "snr_gate", "soft"))
        if not bool(getattr(bio_cfg, "bio_snr_weighting", True)):
            self.snr_gate = "none"

        self._any_bio_active = self.bio_loss_weight > 0.0 or self.sdf_bg_loss_weight > 0.0
        self._bio_neighbors = None
        self._bio_edge_lengths = None

        if self._any_bio_active and ds is not None:
            self._bio_neighbors = ds.neighbors[-1].long().cpu()
            if self.bio_eikonal_weight > 0.0:
                self._bio_edge_lengths = self._precompute_edge_lengths(
                    ds.tet_verts.cpu(), self._bio_neighbors
                )
            print(
                f"[DDPM] Biological constraints active: type={self.bio_loss_type}, "
                f"weight={self.bio_loss_weight}, snr_gate={self.snr_gate}, "
                f"eikonal_weight={self.bio_eikonal_weight}, sdf_bg_loss_weight={self.sdf_bg_loss_weight}"
            )

    @staticmethod
    def _precompute_edge_lengths(tet_verts: Tensor, neighbors: Tensor) -> Tensor:
        """Precompute per-edge Euclidean lengths for Eikonal regularization."""
        valid = neighbors >= 0
        neigh_safe = neighbors.clone()
        neigh_safe[~valid] = 0

        neigh_pos = tet_verts[neigh_safe]
        self_pos = tet_verts.unsqueeze(1).expand_as(neigh_pos)
        edge_len = (neigh_pos - self_pos).norm(dim=-1)
        edge_len[~valid] = 1.0
        return edge_len

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def p_mean_variance(
        self,
        x: Tensor,
        time: Tensor,
        time_next: Tensor,
        image: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute posterior mean and variance for reverse diffusion timestep t -> t_next."""
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
        pred = self.model(x, batch_log_snr, image=image)

        # Classifier-free guidance pass
        if self.cfg_guidance_scale > 1.0 and image is not None:
            pred_uncond = self.model(x, batch_log_snr, image=None)
            pred = pred_uncond + self.cfg_guidance_scale * (pred - pred_uncond)

        if self.pred_objective == "v":
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == "eps":
            x_start = (x - sigma * pred) / alpha

        x_start = x_start.clamp_(-1.0, 1.0)
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(
        self,
        x: Tensor,
        time: Tensor,
        time_next: Tensor,
        deterministic: bool = False,
        image: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample single reverse step t -> t_next."""
        model_mean, model_variance = self.p_mean_variance(
            x=x, time=time, time_next=time_next, image=image
        )
        if time_next == 0 or deterministic:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        initial_noise: Optional[Tensor] = None,
        deterministic: bool = False,
        image: Optional[Tensor] = None,
    ) -> Tensor:
        """Iteratively run reverse sampling loop for specified step count."""
        if initial_noise is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = initial_noise.clone().to(self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)

        for i in tqdm(
            range(self.num_sample_steps),
            desc="Sampling 3D Mesh",
            total=self.num_sample_steps,
        ):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(
                img, times, times_next, deterministic=deterministic, image=image
            )

        img.clamp_(-1.0, 1.0)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        deterministic: bool = False,
        image: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate a batch of 3D grid shapes for all configured sampling step counts."""
        num_sample_steps = self.cfg.diffusion.sampling_steps
        shapes = []
        base_noise = torch.randn(
            (batch_size, self.num_verts, self.channels), device=self.device
        )
        if image is not None and image.shape[0] != batch_size:
            image = image[:1].expand(batch_size, *image.shape[1:])
        for nss in num_sample_steps:
            self.num_sample_steps = nss
            result = self.p_sample_loop(
                (batch_size, self.num_verts, self.channels),
                initial_noise=base_noise,
                deterministic=deterministic,
                image=image,
            )
            shapes.append(result)
        return torch.cat(shapes, 0)

    @autocast("cuda", enabled=False)
    def q_sample(
        self, x_start: Tensor, times: Tensor, noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward diffusion process corrupting clean sample x_start with Gaussian noise."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma
        return x_noised, log_snr

    def p_losses(
        self,
        x_start: Tensor,
        times: Tensor,
        noise: Optional[Tensor] = None,
        offset_noise_strength: Optional[float] = None,
        image: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute training loss: MSE diffusion loss + biological constraints + SDF background loss."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start[:, 0, :].shape, device=self.device)
            offset_noise = offset_noise.clamp(-3.0, 3.0)
            noise += offset_noise_strength * offset_noise.unsqueeze(1)

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)

        drop_mask = None
        if self.use_image_cond and image is not None and self.cfg_dropout_prob > 0.0:
            drop_mask = torch.rand(x.shape[0], device=x.device) < self.cfg_dropout_prob

        model_out = self.model(x, log_snr, image=image, image_drop_mask=drop_mask)

        if self.pred_objective == "v":
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = sqrt(padded_log_snr.sigmoid()), sqrt((-padded_log_snr).sigmoid())
            target = alpha * noise - sigma * x_start
        elif self.pred_objective == "eps":
            target = noise

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max=self.min_snr_gamma)

        if self.pred_objective == "v":
            loss_weight = maybe_clip_snr / (snr + 1)
        elif self.pred_objective == "eps":
            loss_weight = maybe_clip_snr / snr

        diffusion_loss = (loss * loss_weight).mean()

        # Biological constraint losses + SDF background loss
        if self._any_bio_active and self._bio_neighbors is not None:
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha_p = sqrt(padded_log_snr.sigmoid())
            sigma_p = sqrt((-padded_log_snr).sigmoid())

            if self.pred_objective == "v":
                x_start_pred = alpha_p * x - sigma_p * model_out
            else:
                x_start_pred = (x - sigma_p * model_out) / alpha_p

            x_start_pred = x_start_pred.clamp(-1.0, 1.0)
            neighbors = self._bio_neighbors.to(x.device)
            edge_lengths = (
                self._bio_edge_lengths.to(x.device)
                if self._bio_edge_lengths is not None
                else None
            )

            # SNR gating
            gate = self.snr_gate
            if gate == "soft":
                snr_weight = (snr / (snr + 1)).mean()
            elif gate == "none":
                snr_weight = 1.0
            elif gate.startswith("hard_"):
                try:
                    t_thresh = float(gate.split("_")[1])
                except (IndexError, ValueError):
                    t_thresh = 0.5
                snr_weight = (times < t_thresh).float().mean()
            else:
                snr_weight = (snr / (snr + 1)).mean()

            total_bio = x_start_pred.new_zeros(1).squeeze()

            if self.bio_loss_weight > 0.0:
                bio = organelle_loss_registry.compute_loss(
                    organelle_name=self.bio_loss_type,
                    x_start=x_start_pred,
                    neighbors=neighbors,
                    surface_softness=self.bio_curvature_softness,
                    edge_lengths=edge_lengths,
                    eikonal_weight=self.bio_eikonal_weight,
                    sdf_bg_loss_weight=0.0,
                )
                self._last_bio_loss = bio.detach()
                total_bio = total_bio + self.bio_loss_weight * snr_weight * bio
            else:
                self._last_bio_loss = diffusion_loss.new_zeros(1).squeeze().detach()

            if self.sdf_bg_loss_weight > 0.0:
                from lib.ops.BioConstraints import sdf_background_loss

                bg_loss = sdf_background_loss(
                    x_start_pred, bg_threshold=self.sdf_bg_threshold
                )
                self._last_sdf_bg_loss = bg_loss.detach()
                total_bio = total_bio + self.sdf_bg_loss_weight * snr_weight * bg_loss
            else:
                self._last_sdf_bg_loss = (
                    diffusion_loss.new_zeros(1).squeeze().detach()
                )

            self._last_diffusion_loss = diffusion_loss.detach()
            return diffusion_loss + total_bio

        return diffusion_loss

    def forward(self, img: Tensor, image: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        b = img.shape[0]
        img = normalize_to_neg_one_to_one(img)
        times = torch.zeros((b,), device=self.device).uniform_(0, 1)
        return self.p_losses(img, times, image=image, *args, **kwargs)

