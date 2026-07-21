import math
from functools import wraps
from torch.amp import autocast
import torch
from torch import nn, Tensor, sqrt, einsum
import torch.nn.functional as F
from torch.special import expm1
from glob import glob
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from lib.ops.Misc import default, exists
from lib.ops.BioConstraints import biological_constraints_loss, organelle_loss_registry


# Utility functions
def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: Tensor) -> Tensor:
    return (t + 1) * 0.5


def right_pad_dims_to(x: Tensor, t: Tensor) -> Tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


# Logsnr schedules and shifting / interpolating decorators
def log(t: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t: Tensor, logsnr_min: float = -15, logsnr_max: float = 15) -> Tensor:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs) + shift

    return inner


def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner


# Main Gaussian Diffusion class
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            *,
            num_verts: int,
            channels: int = 3,
            pred_objective: str = 'v',
            noise_schedule=logsnr_schedule_cosine,
            noise_d: int = 32,
            noise_d_low: int = None,
            noise_d_high: int = None,
            num_sample_steps: int = 32,
            clip_sample_denoised: bool = True,
            min_snr_loss_weight: bool = True,
            min_snr_gamma: int = 5,
            image_size: int = 128,
            cfg=None,
            offset_noise_strength: float = 0.1,
            ds=None,                        # MeshLoader — used for biological constraints
    ):
        super().__init__()
        assert pred_objective in {'v', 'eps'}, 'Prediction objective must be either "v" or "eps".'

        self.model = model
        self.cfg = cfg
        self.channels = channels
        self.num_verts = num_verts
        self.offset_noise_strength = offset_noise_strength
        self.pred_objective = pred_objective


        assert not all(map(exists, (noise_d, noise_d_low, noise_d_high))), 'Set either noise_d or both noise_d_low and noise_d_high.'

        self.log_snr = noise_schedule
        if not cfg.diffusion.use_standard_noise:
            if exists(noise_d):
                self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

            if exists(noise_d_low) and exists(noise_d_high):
                self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # ------------------------------------------------------------------
        # Optional image conditioning config (mirrors UVIT)
        # ------------------------------------------------------------------
        img_cond_cfg = getattr(cfg, 'image_cond', None)
        self.use_image_cond = (
            img_cond_cfg is not None
            and bool(getattr(img_cond_cfg, 'enabled', False))
        )
        self.cfg_dropout_prob   = float(getattr(img_cond_cfg, 'cfg_dropout_prob',   0.1))  if self.use_image_cond else 0.0
        self.cfg_guidance_scale = float(getattr(img_cond_cfg, 'cfg_guidance_scale', 1.0))  if self.use_image_cond else 1.0

        # ------------------------------------------------------------------
        # Biological constraint losses
        # ------------------------------------------------------------------
        bio_cfg = getattr(cfg, 'diffusion', None)
        self.bio_loss_weight        = float(getattr(bio_cfg, 'bio_loss_weight', 0.0))
        self.bio_loss_type          = str(getattr(bio_cfg, 'bio_loss_type', 'both'))
        self.bio_curvature_softness = float(getattr(bio_cfg, 'bio_curvature_softness', 0.15))
        self.bio_eikonal_weight     = float(getattr(bio_cfg, 'bio_eikonal_weight', 0.0))
        self.sdf_bg_loss_weight     = float(getattr(bio_cfg, 'sdf_bg_loss_weight', 0.0))
        self.sdf_bg_threshold       = float(getattr(bio_cfg, 'sdf_bg_threshold', 0.3))

        # SNR gate — governs when biological losses are active.
        #   "soft"     : w(t) = SNR(t) / (SNR(t) + 1)  (default, continuous)
        #   "hard_X"   : 1.0 if SNR(t) corresponds to t < X, else 0.0
        #   "none"     : constant 1.0 (no gating)
        self.snr_gate = str(getattr(bio_cfg, 'snr_gate', 'soft'))
        # Legacy fallback: if old bio_snr_weighting=False was set, honour it
        if not bool(getattr(bio_cfg, 'bio_snr_weighting', True)):
            self.snr_gate = 'none'

        # Store the tetrahedral-graph neighbour table (finest resolution) on CPU;
        # it will be moved to the active device on first use.
        # Determine if the bio-constraint block should be active at all:
        # either the classic bio loss OR the new sdf_bg_loss are enabled.
        self._any_bio_active = (
            self.bio_loss_weight > 0.0 or self.sdf_bg_loss_weight > 0.0
        )

        self._bio_neighbors  = None
        self._bio_edge_lengths = None
        if self._any_bio_active and ds is not None:
            # After grid-pruning ds.neighbors[-1] is in the pruned vertex space —
            # the same space the training data lives in.
            self._bio_neighbors = ds.neighbors[-1].long().cpu()

            # Precompute edge lengths for the optional Eikonal regulariser.
            if self.bio_eikonal_weight > 0.0:
                self._bio_edge_lengths = self._precompute_edge_lengths(
                    ds.tet_verts.cpu(), self._bio_neighbors
                )

            print(
                f"[DDPM] Biological constraints enabled: "
                f"type={self.bio_loss_type}, weight={self.bio_loss_weight}, "
                f"snr_gate={self.snr_gate}, "
                f"eikonal_weight={self.bio_eikonal_weight}, "
                f"sdf_bg_loss_weight={self.sdf_bg_loss_weight}, "
                f"sdf_bg_threshold={self.sdf_bg_threshold}, "
                f"neighbor shape={self._bio_neighbors.shape}"
            )

    @staticmethod
    def _precompute_edge_lengths(tet_verts: torch.Tensor,
                                  neighbors: torch.Tensor) -> torch.Tensor:
        """Precompute per-edge Euclidean lengths for the Eikonal loss.

        Args:
            tet_verts:  [N, 3] float32 vertex positions (pruned space).
            neighbors:  [N, K] int64 neighbour table (−1 = absent).

        Returns:
            Tensor [N, K] with edge lengths; invalid slots filled with 1.0.
        """
        valid = (neighbors >= 0)          # [N, K]
        neigh_safe = neighbors.clone()
        neigh_safe[~valid] = 0            # redirect invalid to vertex 0

        neigh_pos = tet_verts[neigh_safe]                   # [N, K, 3]
        self_pos  = tet_verts.unsqueeze(1).expand_as(neigh_pos)  # [N, K, 3]
        edge_len  = (neigh_pos - self_pos).norm(dim=-1)     # [N, K]
        edge_len[~valid] = 1.0            # safe dummy for invalid entries
        return edge_len

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x: Tensor, time: Tensor, time_next: Tensor, image: Tensor = None):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model(x, batch_log_snr, image=image)

        # Classifier-free guidance: run a second unconditional pass and
        # interpolate.  Only active when guidance_scale > 1 and an image
        # was actually provided.
        if self.cfg_guidance_scale > 1.0 and image is not None:
            pred_uncond = self.model(x, batch_log_snr, image=None)
            pred = pred_uncond + self.cfg_guidance_scale * (pred - pred_uncond)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha


        x_start = x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x: Tensor, time: Tensor, time_next: Tensor, deterministic: bool = False, image: Tensor = None) -> Tensor:
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next, image=image)
        if time_next == 0 or deterministic:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, initial_noise: Tensor = None, deterministic: bool = False, image: Tensor = None) -> Tensor:
        if initial_noise is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = initial_noise.clone().to(self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, deterministic=deterministic, image=image)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16, deterministic: bool = False, image: Tensor = None) -> Tensor:
        num_sample_steps = self.cfg.diffusion.sampling_steps
        shapes = []
        base_noise = torch.randn((batch_size, self.num_verts, self.channels), device=self.device)
        # If a conditioning image is given but batch size differs, tile it.
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

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start: Tensor, times: Tensor, noise: Tensor = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma
        return x_noised, log_snr



    def p_losses(self, x_start: Tensor, times: Tensor, noise: Tensor = None,
                 offset_noise_strength: float = None, image: Tensor = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start[:, 0, :].shape, device=self.device)
            offset_noise = offset_noise.clamp(-3., 3.)   # prevent 6-σ tail samples from overflowing v-target
            noise += offset_noise_strength * offset_noise.unsqueeze(1)

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)

        # Classifier-free guidance: per-sample dropout of image conditioning.
        # Dropped samples use the null_image_emb inside UVIT.
        drop_mask = None
        if self.use_image_cond and image is not None and self.cfg_dropout_prob > 0.0:
            drop_mask = torch.rand(x.shape[0], device=x.device) < self.cfg_dropout_prob

        model_out = self.model(x, log_snr, image=image, image_drop_mask=drop_mask)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = sqrt(padded_log_snr.sigmoid()), sqrt((-padded_log_snr).sigmoid())
            target = alpha * noise - sigma * x_start
        elif self.pred_objective == 'eps':
            target = noise

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max=self.min_snr_gamma)

        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)
        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr

        diffusion_loss = (loss * loss_weight).mean()

        # ------------------------------------------------------------------
        # Biological constraint losses  +  SDF background loss
        # Recover predicted x_start (clean sample estimate) and apply
        # curvature / smoothness regularisation on it.  The SDF background
        # loss additionally fires even when bio_loss_weight=0 so it can be
        # used as a standalone training signal.
        # ------------------------------------------------------------------
        if self._any_bio_active and self._bio_neighbors is not None:
            # Recover x_start_pred from model output without re-running the model
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha_p = sqrt(padded_log_snr.sigmoid())
            sigma_p = sqrt((-padded_log_snr).sigmoid())
            if self.pred_objective == 'v':
                x_start_pred = alpha_p * x - sigma_p * model_out
            else:  # eps
                x_start_pred = (x - sigma_p * model_out) / alpha_p

            x_start_pred = x_start_pred.clamp(-1., 1.)

            neighbors = self._bio_neighbors.to(x.device)
            edge_lengths = (
                self._bio_edge_lengths.to(x.device)
                if self._bio_edge_lengths is not None else None
            )

            # ── SNR gate ─────────────────────────────────────────────────
            gate = self.snr_gate
            if gate == 'soft':
                snr_weight = (snr / (snr + 1)).mean()
            elif gate == 'none':
                snr_weight = 1.0
            elif gate.startswith('hard_'):
                try:
                    t_thresh = float(gate.split('_')[1])
                except (IndexError, ValueError):
                    t_thresh = 0.5
                snr_weight = (times < t_thresh).float().mean()
            else:
                snr_weight = (snr / (snr + 1)).mean()

            total_bio = x_start_pred.new_zeros(1).squeeze()

            # ── Classic bio constraints (laplacian + curvature + eikonal) ─
            if self.bio_loss_weight > 0.0:
                bio = organelle_loss_registry.compute_loss(
                    organelle_name=self.bio_loss_type,
                    x_start=x_start_pred,
                    neighbors=neighbors,
                    surface_softness=self.bio_curvature_softness,
                    edge_lengths=edge_lengths,
                    eikonal_weight=self.bio_eikonal_weight,
                    # sdf_bg handled separately below so it can be
                    # logged and weighted independently
                    sdf_bg_loss_weight=0.0,
                )
                self._last_bio_loss = bio.detach()
                total_bio = total_bio + self.bio_loss_weight * snr_weight * bio
            else:
                self._last_bio_loss = diffusion_loss.new_zeros(1).squeeze().detach()

            # ── Surface-weighted SDF background loss ─────────────────────
            if self.sdf_bg_loss_weight > 0.0:
                from lib.ops.BioConstraints import sdf_background_loss
                bg_loss = sdf_background_loss(
                    x_start_pred, bg_threshold=self.sdf_bg_threshold
                )
                self._last_sdf_bg_loss = bg_loss.detach()
                # SNR-gated: only meaningful when x_start_pred is reasonable
                total_bio = total_bio + self.sdf_bg_loss_weight * snr_weight * bg_loss
            else:
                self._last_sdf_bg_loss = diffusion_loss.new_zeros(1).squeeze().detach()

            self._last_diffusion_loss = diffusion_loss.detach()
            return diffusion_loss + total_bio

        return diffusion_loss

    def forward(self, img: Tensor, image: Tensor = None, *args, **kwargs) -> Tensor:
        b, h, c, device = *img.shape, img.device

        img = normalize_to_neg_one_to_one(img)

        times = torch.zeros((b,), device=self.device).uniform_(0, 1)
        return self.p_losses(img, times, image=image, *args, **kwargs)
