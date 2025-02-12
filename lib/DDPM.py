import math
from functools import wraps
from torch.cuda.amp import autocast
import torch
from torch import nn, Tensor, sqrt, einsum
import torch.nn.functional as F
from torch.special import expm1
from glob import glob
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from lib.ops.Misc import default, exists


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
            offset_noise_strength: float = 0.1
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

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x: Tensor, time: Tensor, time_next: Tensor):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model(x, batch_log_snr)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha


        x_start = x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x: Tensor, time: Tensor, time_next: Tensor) -> Tensor:
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple) -> Tensor:
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16) -> Tensor:
        num_sample_steps = self.cfg.diffusion.sampling_steps
        shapes = []
        for nss in num_sample_steps:
            self.num_sample_steps = nss
            result = self.p_sample_loop((batch_size, self.num_verts, self.channels))
            shapes.append(result)
        return torch.cat(shapes, 0)

    @autocast(enabled=False)
    def q_sample(self, x_start: Tensor, times: Tensor, noise: Tensor = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma
        return x_noised, log_snr



    def p_losses(self, x_start: Tensor, times: Tensor, noise: Tensor = None,
                 offset_noise_strength: float = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start[:, 0, :].shape, device=self.device)
            noise += offset_noise_strength * offset_noise.unsqueeze(1)

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr)

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

        return (loss * loss_weight).mean()

    def forward(self, img: Tensor, *args, **kwargs) -> Tensor:
        b, h, c, device = *img.shape, img.device


        img = normalize_to_neg_one_to_one(img)

        times = torch.zeros((b,), device=self.device).uniform_(0, 1)
        return self.p_losses(img, times, *args, **kwargs)
