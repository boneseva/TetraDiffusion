import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from lib.ops.Attention import TimeEmbeddingNet
from lib.ops.Misc import RMSNorm

class UnaryBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_groupnorm: bool, num_groups: int = 32, norm_cfg: str = "Silu"):
        super().__init__()
        self.mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.use_groupnorm = use_groupnorm
        if norm_cfg == "None":
            self.norm = nn.Identity()
            self.act = nn.Identity()
        else:
            if self.use_groupnorm:
                self.norm = nn.GroupNorm(num_groups,out_channels)
            else:
                self.norm = RMSNorm(out_channels)
            self.act = nn.SiLU()

    def forward(self, feats: Tensor, scale_shift: tuple = None) -> Tensor:
        feats = self.mlp(feats)
        if self.use_groupnorm:
            feats = self.norm(feats.transpose(-1, -2)).transpose(-1, -2)
        else:
            feats = self.norm(feats)
        if scale_shift is not None:
            scale, shift = scale_shift
            feats = feats * (scale + 1) + shift
        return self.act(feats)


class TetraConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,  use_groupnorm: bool, num_groups: int = 32,act_cfg: str = "SiLU", neighbor_indices=None, vertices=None, down_up_: bool = False):
        super().__init__()
        self.conv = TetraConv(in_channels, out_channels, kernel_size, neighbor_indices, vertices=vertices, down_up=down_up_)
        self.use_groupnorm = use_groupnorm
        if act_cfg != "None":
            if self.use_groupnorm:
                self.norm = nn.GroupNorm(num_groups, out_channels)
            else:
                self.norm = RMSNorm(out_channels)
            self.act = nn.SiLU()
        else:
            self.norm = nn.Identity()
            self.act = nn.Identity()

    def forward(self, vert_features: Tensor, scale_shift: tuple = None) -> Tensor:
        vert_feats = self.conv(vert_features)
        if self.use_groupnorm:
            vert_feats = self.norm(vert_feats.transpose(-1, -2)).transpose(-1, -2)
        else:
            vert_feats = self.norm(vert_feats)
        if scale_shift is not None:
            scale, shift = scale_shift
            vert_feats = vert_feats * (scale + 1) + shift
        return self.act(vert_feats)


class TetraResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, use_groupnorm: bool, num_groups: int,
                 act_cfg: str = "SiLU", time_emb_dim: int = None, neighbor_indices=None, vertices=None):
        super().__init__()
        self.conv = torch.compile(_TetraResidualBlock(in_channels, out_channels, kernel_size, use_groupnorm, num_groups, act_cfg, time_emb_dim, neighbor_indices, vertices))

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        return self.conv(x, t)


class _TetraResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, use_groupnorm: bool, num_groups: int,
                 act_cfg: str = "SiLU", time_emb_dim: int = None, neighbor_indices=None, vertices=None):
        super().__init__()
        self.conv = TetraConvBlock(in_channels, out_channels, kernel_size, use_groupnorm, num_groups, act_cfg, neighbor_indices, vertices)
        self.conv2 = TetraConvBlock(out_channels, out_channels, kernel_size, use_groupnorm, num_groups, act_cfg, neighbor_indices, vertices)
        self.unary_shortcut = UnaryBlock(in_channels, out_channels, use_groupnorm, num_groups=0, norm_cfg="None") if in_channels != out_channels else nn.Identity()
        self.mlp1 = TimeEmbeddingNet(time_emb_dim, out_channels * 2) if time_emb_dim is not None else None

    def forward(self, verts_features: Tensor, time_emb: Tensor = None) -> Tensor:
        scale_shift1 = self.mlp1(time_emb) if self.mlp1 is not None and time_emb is not None else None
        residual = self.conv(verts_features, scale_shift1)
        residual = self.conv2(residual)
        shortcut = self.unary_shortcut(verts_features)
        return residual + shortcut


class SingleTetraConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,  use_groupnorm: bool,num_groups: int,
                 act_cfg: str = "SiLU", time_emb_dim: int = None, neighbor_indices=None, vertices=None, down_up_: bool = False):
        super().__init__()
        self.conv = torch.compile(_SingleTetraConvBlock(in_channels, out_channels, kernel_size,use_groupnorm, num_groups, act_cfg, time_emb_dim, neighbor_indices, vertices, down_up_))

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        return self.conv(x, t)


class _SingleTetraConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, use_groupnorm: bool, num_groups: int,
                 act_cfg: str = "SiLU", time_emb_dim: int = None, neighbor_indices=None, vertices=None, down_up_: bool = False):
        super().__init__()
        self.conv = TetraConvBlock(in_channels, out_channels, kernel_size, use_groupnorm, num_groups, act_cfg, neighbor_indices, vertices, down_up_)
        self.mlp1 = TimeEmbeddingNet(time_emb_dim, out_channels * 2) if time_emb_dim is not None else None

    def forward(self, verts_features: Tensor, time_emb: Tensor = None) -> Tensor:
        scale_shift1 = self.mlp1(time_emb) if self.mlp1 is not None and time_emb is not None else None
        return self.conv(verts_features, scale_shift1)


class TetraConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, neighbor_indices: Tensor, use_bias: bool = True, vertices=None, down_up: bool = False):
        super().__init__()
        with torch.no_grad():
            neighbor_indices = neighbor_indices.clone()
            self.neighbors_mask = neighbor_indices == -1
            self.valid_neighbors = neighbor_indices != -1
            neighbor_indices[self.neighbors_mask] = torch.max(neighbor_indices) + 1
            self.neighbor_indices = neighbor_indices

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.weights = nn.Parameter(torch.ones(size=(out_channels, kernel_size * in_channels)))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def index_select(self, inputs: Tensor, indices: Tensor, dim: int) -> Tensor:
        outputs = inputs.index_select(dim, indices.view(-1))
        if indices.dim() > 1:
            if dim < 0:
                dim += inputs.dim()
            output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1:]
            outputs = outputs.view(*output_shape)
        return outputs

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, vert_features: Tensor) -> Tensor:
        vert_features = F.pad(vert_features, (0, 0, 0, 1, 0, 0), value=0)
        unfold = self.index_select(vert_features, self.neighbor_indices, 1)
        b, m, k, c = unfold.size()
        unfold_reshaped = unfold.view(b * m, k * c)
        weights_reshaped = self.weights.view(k * c, -1)
        res = torch.matmul(unfold_reshaped, weights_reshaped).view(b, m, -1)
        if self.use_bias:
            res.add_(self.bias)
        return res
