"""Progressive U-Net style segmentation decoder with residual refinement."""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_tune.decoders.segmentation_decoder import SegmentationDecoder


class ResidualBlock(nn.Module):
    """Residual convolutional block with residual connection."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class UNetDecoder(nn.Module):
    """Progressive multi-stage U-Net decoder with residual refinement and skip fusion."""

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        hidden_dims: Tuple[int, ...] = (256, 128, 64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial projection
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.GELU(),
        )

        # Stage 0: 14x14 (256) + skip s8 (96) -> 28x28 (128)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block0 = nn.Sequential(
            nn.Conv2d(hidden_dims[0] + 96, hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.GELU(),
            ResidualBlock(hidden_dims[1], dropout=dropout),
        )
        self.block0_noskip = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.GELU(),
            ResidualBlock(hidden_dims[1], dropout=dropout),
        )

        # Stage 1: 28x28 (128) + skip s4 (64) -> 56x56 (64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dims[1] + 64, hidden_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.GELU(),
            ResidualBlock(hidden_dims[2], dropout=dropout),
        )
        self.block1_noskip = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.GELU(),
            ResidualBlock(hidden_dims[2], dropout=dropout),
        )

        # Stage 2: 56x56 (64) + skip s2 (32) -> 112x112 (32)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dims[2] + 32, hidden_dims[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.GELU(),
            ResidualBlock(hidden_dims[3], dropout=dropout),
        )
        self.block2_noskip = nn.Sequential(
            nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.GELU(),
            ResidualBlock(hidden_dims[3], dropout=dropout),
        )

        # Stage 3: 112x112 (32) -> 224x224 (16)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_dims[3], 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            ResidualBlock(16, dropout=dropout),
        )

        # Final classification head
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(
        self,
        features: Any,
        target_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if isinstance(features, dict):
            out = features["out"]
            skips = features.get("skips", None)
        elif isinstance(features, (tuple, list)):
            out = features[0]
            skips = features[1] if len(features) > 1 else None
        else:
            out = features
            skips = None

        if out.ndim == 2:
            b, c = out.shape
            out = out.view(b, c, 1, 1)

        x = self.in_proj(out)

        # Stage 0: 14 -> 28
        x = self.up0(x)
        if skips is not None and len(skips) > 0 and skips[0].shape[-2:] == x.shape[-2:]:
            x = self.block0(torch.cat([x, skips[0]], dim=1))
        else:
            x = self.block0_noskip(x)

        # Stage 1: 28 -> 56
        x = self.up1(x)
        if skips is not None and len(skips) > 1 and skips[1].shape[-2:] == x.shape[-2:]:
            x = self.block1(torch.cat([x, skips[1]], dim=1))
        else:
            x = self.block1_noskip(x)

        # Stage 2: 56 -> 112
        x = self.up2(x)
        if skips is not None and len(skips) > 2 and skips[2].shape[-2:] == x.shape[-2:]:
            x = self.block2(torch.cat([x, skips[2]], dim=1))
        else:
            x = self.block2_noskip(x)

        # Stage 3: 112 -> 224
        x = self.up3(x)
        x = self.block3(x)

        logits = self.head(x)

        if target_size is not None and (logits.shape[-2] != target_size[0] or logits.shape[-1] != target_size[1]):
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits


def build_decoder(
    decoder_name: str,
    in_channels: int,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Factory function for instantiating segmentation decoders."""
    name = decoder_name.lower().replace("-", "_")
    if name in ("unet_decoder", "unet"):
        return UNetDecoder(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif name in ("segmentation_decoder", "fcn", "conv"):
        return SegmentationDecoder(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}. Choose 'segmentation_decoder' or 'unet_decoder'.")
