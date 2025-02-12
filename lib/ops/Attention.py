import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat

from lib.ops.Misc import RMSNorm, Residual, PreNorm, l2norm


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.att = torch.compile(Residual(PreNorm(dim, _LinearAttention(dim, heads=heads, dim_head=dim_head))))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.att(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        self.att = torch.compile(Residual(_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.att(x)
        return x


class _LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, num_mem_kv: int = 4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            RMSNorm(dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, h, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x (h c) -> b h c x', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)

        k = torch.cat([mk, k], -1)
        v = torch.cat([mv, v], -1)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c x -> b x (h c)', h=self.heads)
        return self.to_out(out)


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, scale: int = 8, dropout: float = 0.0, rmsnorm: bool = True, mem_efficient: bool = True, num_mem_kv: int = 4):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim) if rmsnorm else nn.LayerNorm(dim)
        self.dropout = dropout
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        self.cuda_config = AttentionConfig(True, False, False) if device_properties.major == 8 and device_properties.minor == 0 else AttentionConfig(True, True, False)
        self.cpu_config = AttentionConfig(True, True, True)

    def flash_attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        config = self.cuda_config if q.is_cuda else self.cpu_config
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.
            )
        return out

    def forward(self, x: Tensor) -> Tensor:
        b, h, c = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k = torch.cat([mk, k], -2)
        v = torch.cat([mv, v], -2)

        out = self.flash_attn(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbeddingNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, time_emb: Tensor) -> tuple:
        tb2 = self.mlp(time_emb)
        tb2 = rearrange(tb2, 'b 1 c -> b 1 c')
        return tb2.chunk(2, dim=2)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
