from typing import Any
import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 512, base_channels: int = 32):
        super().__init__()

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (B, c1, H/2, W/2) -> (B, c2, H/4, W/4)
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: (B, c2, H/4, W/4) -> (B, c3, H/8, W/8)
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Global Average Pooling -> (B, c3)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(c3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Zero-initialise final projection layer so image conditioning starts as zero offset
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, H, W) -> (B, out_dim)."""
        return self.proj(self.cnn(x))


def build_image_encoder(cfg_image_cond: Any, out_dim: int) -> nn.Module:
    """
    Factory creating an image encoder module based on configuration options.

    Args:
        cfg_image_cond: The image_cond sub-config (OmegaConf node).
        out_dim: Target feature dimension matching UVIT's time_dim.
    """
    encoder_type = str(getattr(cfg_image_cond, "encoder", "cnn"))
    in_channels = int(getattr(cfg_image_cond, "in_channels", 1))
    base_channels = int(getattr(cfg_image_cond, "base_channels", 32))

    if encoder_type == "cnn":
        return SmallCNNEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            base_channels=base_channels,
        )
    else:
        raise ValueError(
            f"Unknown image encoder type '{encoder_type}'. Supported: 'cnn'."
        )


