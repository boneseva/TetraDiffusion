"""
2D image encoder for conditioning TetraDiffusion on microscopy images.

The encoder maps a batch of 2D images  (B, C, H, W)  to a conditioning
vector of shape  (B, time_dim)  that is added to the diffusion timestep
embedding inside UVIT.  All existing unconditional code paths are
completely unaffected when image_cond.enabled=False.

Architecture
------------
SmallCNNEncoder: 3 × [Conv3×3 → BatchNorm → ReLU → MaxPool2×2]
                 → AdaptiveAvgPool → Flatten
                 → Linear(base*4 → time_dim) → SiLU → Linear(time_dim → time_dim)

The last linear is zero-initialised so image conditioning starts as a
zero offset and is gradually learned — safe to fine-tune on top of an
existing unconditional checkpoint.
"""

import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    """
    Lightweight CNN encoder.

    Args:
        in_channels:   Number of input image channels (1 = grayscale, 3 = RGB).
        out_dim:       Output vector dimension — should match UVIT's time_dim.
        base_channels: Number of feature maps in the first conv block; doubled
                       each block so the third block has base_channels*4 maps.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 512, base_channels: int = 32):
        super().__init__()

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.cnn = nn.Sequential(
            # block 1  (B, in_ch, H,   W)   → (B, c1, H/2, W/2)
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 2  (B, c1, H/2, W/2)   → (B, c2, H/4, W/4)
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 3  (B, c2, H/4, W/4)   → (B, c3, H/8, W/8)
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # global pool → (B, c3)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(c3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Zero-initialise final projection so the image conditioning starts
        # as a zero offset — identical to the unconditional path at step 0.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, out_dim)"""
        return self.proj(self.cnn(x))


def build_image_encoder(cfg_image_cond, out_dim: int) -> nn.Module:
    """
    Factory that instantiates the image encoder from the image_cond config
    block.  Currently only 'cnn' (SmallCNNEncoder) is supported; additional
    encoders (e.g. pretrained BioMedCLIP projection head) can be added here.

    Args:
        cfg_image_cond: The ``image_cond`` sub-config (OmegaConf node).
        out_dim:        Must equal UVIT's ``time_dim`` (first_dim * 4).
    """
    encoder_type  = str(getattr(cfg_image_cond, 'encoder', 'cnn'))
    in_channels   = int(getattr(cfg_image_cond, 'in_channels', 1))
    base_channels = int(getattr(cfg_image_cond, 'base_channels', 32))

    if encoder_type == 'cnn':
        return SmallCNNEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            base_channels=base_channels,
        )
    else:
        raise ValueError(
            f"Unknown image encoder type '{encoder_type}'. "
            "Supported: 'cnn'."
        )

