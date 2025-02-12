import torch
from torch import nn
from torch.nn import functional as F


# Helper functions
def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num: int, divisor: int) -> list:
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def exists(x):
    return x is not None


def append_dims(t: torch.Tensor, dims: int) -> torch.Tensor:
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


# Custom modules
class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale: bool = True, normalize_dim: int = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1
        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim=self.normalize_dim) * scale * (x.shape[self.normalize_dim] ** 0.5)
