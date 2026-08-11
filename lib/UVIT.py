from typing import Any, List, Optional, Tuple, Union
from einops.layers.torch import Rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from lib.ops.Attention import (
    Attention,
    LearnedSinusoidalPosEmb,
    LinearAttention,
    RMSNorm,
    SinusoidalPosEmb,
)
from lib.ops.ImageEncoder import build_image_encoder
from lib.ops.TetraConv import SingleTetraConvBlock, TetraConvBlock, TetraResidualBlock


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale=False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange("b d -> b 1 d"),
        )
        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias=False),
            nn.SiLU(),
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj_in(x)
        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)
        x = x * (scale + 1) + shift
        return self.proj_out(x)


class Transformer(nn.Module):
    """Vision Transformer bottleneck operating on deep tetrahedral features."""

    def __init__(
        self,
        dim: int,
        time_cond_dim: int,
        depth: int,
        dim_head: int = 32,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                        FeedForward(
                            dim=dim, mult=ff_mult, cond_dim=time_cond_dim, dropout=dropout
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, t)
            x = ff(x, t) + x
        return x


def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """Interleave sine and cosine positional embedding components."""
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    """1D Sinusoidal Positional Encoding for 3D coordinates."""

    def __init__(self, channels: int):
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) != 3:
            raise RuntimeError("Input tensor must be 3D (batch_size, seq_len, channels)")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return torch.cat([tensor, self.cached_penc], -1)

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return torch.cat([tensor, self.cached_penc], -1)


class UVIT(nn.Module):
    """3D Tetrahedral U-Net + Transformer neural network for diffusion noise estimation."""

    def __init__(self, config: Any, rank: Union[int, str, torch.device], ds: Any):
        super().__init__()
        self.config = config
        self.rank = rank
        self.ds = ds

        self.channels = 4 + (3 if config.dataset.color else 0)
        self.self_condition = False
        self.num_blocks = config.network.num_blocks
        self.rot_verts = self.ds.tet_verts.clone().to(self.rank)

        first_dim = self.config.network.width
        kernel_size = [16, 16, 16, 16, 16]
        time_dim = first_dim * 4
        learned_sinusoidal_dim = self.config.network.learned_sinusoidal_dim

        channel_multiplier = np.asarray(self.config.network.channel_multiplier)
        self.dropout = np.asarray(self.config.network.dropout)
        self.GC = self.config.network.GC
        self.downsample = self.config.network.downsample
        self.upsample = self.config.network.upsample
        self.channel_multiplier = channel_multiplier
        self.attentions = self.config.network.attentions
        self.linear_attentions = self.config.network.linear_attentions
        self.grid_res = self.config.dataset.grid_res

        self.neighbors = [n.to(rank) for n in self.ds.neighbors]
        self.upsample_indices = [u.to(rank).int() for u in self.ds.upsample]
        self.downsample_indices = [d.to(rank).int() for d in self.ds.downsample]
        self.vertices = [v.to(rank).float() for v in self.ds.vertices]

        input_dim = self.channels * (2 if self.self_condition else 1)
        if self.config.network.add_coords:
            self.VPE = PositionalEncoding1D(3)
            input_dim += 3

        if self.config.network.learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim

        self.time_mlp = torch.compile(
            nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        )

        # 2D Image Conditioning
        img_cfg = getattr(config, "image_cond", None)
        self.use_image_cond = (
            bool(getattr(img_cfg, "enabled", False)) if img_cfg is not None else False
        )
        if self.use_image_cond:
            self.image_encoder = build_image_encoder(img_cfg, out_dim=time_dim)
            self.null_image_emb = nn.Parameter(torch.zeros(time_dim))

        encoder = []
        self.first_conv = TetraResidualBlock(
            input_dim,
            first_dim * channel_multiplier[0],
            kernel_size[-1],
            use_groupnorm=self.config.network.use_group_norm,
            num_groups=self.config.network.group_norm_groups,
            time_emb_dim=time_dim,
            neighbor_indices=self.neighbors[-1],
            vertices=self.vertices[-1],
        )
        if self.config.network.add_coords:
            input_dim -= 3

        in_channel = first_dim * channel_multiplier[0]
        down = 1
        for i in range(len(self.channel_multiplier)):
            out_channel = first_dim * channel_multiplier[i]
            for _ in range(self.num_blocks[i]):
                encoder.append(
                    TetraResidualBlock(
                        in_channel,
                        out_channel,
                        kernel_size[-(i + 1)],
                        use_groupnorm=self.config.network.use_group_norm,
                        num_groups=self.config.network.group_norm_groups,
                        time_emb_dim=time_dim,
                        neighbor_indices=self.neighbors[-down],
                        vertices=self.vertices[-down],
                    )
                )
                in_channel = out_channel
            if self.attentions[i]:
                encoder.append(Attention(out_channel))
            if self.linear_attentions[i]:
                encoder.append(LinearAttention(out_channel))
            if i != len(self.channel_multiplier) - 1 and self.downsample[i]:
                encoder.append(
                    SingleTetraConvBlock(
                        out_channel,
                        out_channel,
                        16,
                        use_groupnorm=self.config.network.use_group_norm,
                        num_groups=self.config.network.group_norm_groups,
                        time_emb_dim=time_dim,
                        neighbor_indices=self.downsample_indices[-down],
                        vertices=self.vertices[-down],
                        down_up_=True,
                    )
                )
                down += 1
            if self.dropout[i] > 0.0:
                encoder.append(nn.Dropout(self.dropout[i], inplace=True))
        self.encoder = nn.ModuleList(encoder)

        b_dim = len(channel_multiplier) - 1
        self.vit = Transformer(
            dim=first_dim * channel_multiplier[b_dim],
            time_cond_dim=time_dim,
            depth=self.config.network.vit_depth,
            dim_head=32,
            heads=4,
            ff_mult=4,
            dropout=0.1,
        )
        self.vit = torch.compile(self.vit)

        decoder = []
        ksize = 0
        in_channel = first_dim * channel_multiplier[b_dim] * 2

        for i in reversed(range(len(channel_multiplier))):
            out_channel = first_dim * channel_multiplier[i]
            for _ in range(self.num_blocks[i]):
                decoder.append(
                    TetraResidualBlock(
                        in_channel,
                        out_channel,
                        kernel_size[ksize],
                        use_groupnorm=self.config.network.use_group_norm,
                        num_groups=self.config.network.group_norm_groups,
                        time_emb_dim=time_dim,
                        neighbor_indices=self.neighbors[-down],
                        vertices=self.vertices[-down],
                    )
                )
                in_channel = out_channel * 2

            ksize += 1
            b_dim -= 1
            in_channel = out_channel + first_dim * channel_multiplier[b_dim]

            if self.attentions[i]:
                decoder.append(Attention(out_channel))
            if self.linear_attentions[i]:
                decoder.append(LinearAttention(out_channel))
            if self.upsample[i]:
                down -= 1
                decoder.append(
                    SingleTetraConvBlock(
                        out_channel,
                        out_channel,
                        16,
                        use_groupnorm=self.config.network.use_group_norm,
                        num_groups=self.config.network.group_norm_groups,
                        time_emb_dim=time_dim,
                        neighbor_indices=self.upsample_indices[-down],
                        vertices=self.vertices[-down],
                        down_up_=True,
                    )
                )
            if self.dropout[i] > 0.0:
                decoder.append(nn.Dropout(self.dropout[i], inplace=True))

        self.decoder = nn.ModuleList(decoder)
        self.end_decoder = TetraResidualBlock(
            first_dim * channel_multiplier[0] + input_dim,
            first_dim * channel_multiplier[0],
            kernel_size[-1],
            use_groupnorm=self.config.network.use_group_norm,
            num_groups=self.config.network.group_norm_groups,
            time_emb_dim=time_dim,
            neighbor_indices=self.neighbors[-1],
            vertices=self.vertices[-1],
        )
        self.final = torch.compile(
            TetraConvBlock(
                first_dim * channel_multiplier[0],
                input_dim,
                kernel_size[-1],
                use_groupnorm=False,
                num_groups=0,
                act_cfg="None",
                neighbor_indices=self.neighbors[-down],
                vertices=self.vertices[-down],
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        image_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_reentrant = True
        x.requires_grad = True
        time.requires_grad = True
        bs = x.shape[0]

        # Timestep embedding calculation
        t = checkpoint.checkpoint(self.time_mlp, time, use_reentrant=use_reentrant)

        # Combine with 2D image conditioning if enabled
        if self.use_image_cond:
            if image is not None:
                img_emb = self.image_encoder(image)
                if image_drop_mask is not None:
                    null = self.null_image_emb.unsqueeze(0).expand_as(img_emb)
                    img_emb = torch.where(
                        image_drop_mask.view(-1, 1).to(img_emb.device),
                        null,
                        img_emb,
                    )
            else:
                img_emb = self.null_image_emb.unsqueeze(0).expand(bs, -1)
            t = t + img_emb

        t = t.unsqueeze(1)

        if self.config.network.add_coords:
            if self.config.dataset.color:
                rot_verts = self.rot_verts.to(self.rank).repeat((bs, 1, 1))
                rot_verts.requires_grad = False
                x = torch.cat([x, rot_verts], -1)
            else:
                x = self.VPE(x)

        # U-Net First Convolution Block
        feats = checkpoint.checkpoint(self.first_conv, x, t, use_reentrant=use_reentrant)

        feat_list = []
        c = 0

        # Encoder Blocks
        for i in range(len(self.channel_multiplier)):
            for _ in range(self.num_blocks[i]):
                feats = self.process_block(feats, t, c, self.encoder, self.GC[i])
                c += 1
                feat_list.append(feats)
            if self.attentions[i]:
                feats = self.process_block(feats, t, c, self.encoder, self.GC[i])
                c += 1
            if self.linear_attentions[i]:
                feats = self.process_block(feats, t, c, self.encoder, self.GC[i])
                c += 1
            if i != len(self.channel_multiplier) - 1 and self.downsample[i]:
                feats = self.process_block(feats, t, c, self.encoder, self.GC[i])
                c += 1
            if self.dropout[i] > 0.0:
                feats = self.encoder[c](feats)
                c += 1

        # Transformer Bottleneck
        feats = self.process_vit(feats, t, self.vit, self.GC[-1])

        # Decoder Blocks
        c = 0
        for i in reversed(range(len(self.channel_multiplier))):
            for _ in range(self.num_blocks[i]):
                feats = torch.cat([feats, feat_list.pop()], -1)
                feats = self.process_block(feats, t, c, self.decoder, self.GC[i])
                c += 1
            if self.attentions[i]:
                feats = self.process_block(feats, t, c, self.decoder, self.GC[i])
                c += 1
            if self.linear_attentions[i]:
                feats = self.process_block(feats, t, c, self.decoder, self.GC[i])
                c += 1
            if i != 0 and self.upsample[i]:
                feats = self.process_block(feats, t, c, self.decoder, self.GC[i])
                c += 1
            if self.dropout[i] > 0.0:
                feats = self.decoder[c](feats)
                c += 1

        # Final Decoder Reconstruction Block
        feats = torch.cat([feats, x], -1)
        feats = checkpoint.checkpoint(self.end_decoder, feats, t, use_reentrant=use_reentrant)

        return checkpoint.checkpoint(self.final, feats, use_reentrant=use_reentrant)

    def process_block(
        self,
        feats: torch.Tensor,
        t: torch.Tensor,
        c: int,
        layer: nn.ModuleList,
        gc_flag: bool,
    ) -> torch.Tensor:
        if gc_flag:
            return checkpoint.checkpoint(layer[c], feats, t, use_reentrant=True)
        else:
            return layer[c](feats, t)

    def process_vit(
        self,
        feats: torch.Tensor,
        t: torch.Tensor,
        vit_layer: nn.Module,
        gc_flag: bool,
    ) -> torch.Tensor:
        if gc_flag:
            return checkpoint.checkpoint(vit_layer, feats, t.squeeze(1), use_reentrant=True)
        else:
            return vit_layer(feats, t.squeeze(1))



