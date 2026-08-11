from typing import Any, Callable, Generator, List, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def default(val: Any, d: Union[Any, Callable[[], Any]]) -> Any:
    return val if exists(val) else (d() if callable(d) else d)


def is_lambda(f: Any) -> bool:
    return callable(f) and f.__name__ == "<lambda>"


def cycle(dl: Any) -> Generator[Any, None, None]:
    while True:
        for data in dl:
            yield data


def num_to_groups(num: int, divisor: int) -> List[int]:
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def exists(x: Any) -> bool:
    return x is not None


def append_dims(t: Tensor, dims: int) -> Tensor:
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


def l2norm(t: Tensor) -> Tensor:
    return F.normalize(t, dim=-1)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.fn(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale: bool = True, normalize_dim: int = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1
        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x: Tensor) -> Tensor:
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim=self.normalize_dim) * scale * (x.shape[self.normalize_dim] ** 0.5)

