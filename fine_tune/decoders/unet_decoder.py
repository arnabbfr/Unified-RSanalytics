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
    """Progressive multi-stage U-Net decoder with residual refinement."""

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

        stages = []
        for i in range(len(hidden_dims) - 1):
            in_d = hidden_dims[i]
            out_d = hidden_dims[i + 1]
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_d, out_d, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_d),
                    nn.GELU(),
                    ResidualBlock(out_d, dropout=dropout),
                )
            )
        self.stages = nn.ModuleList(stages)

        # Final classification head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dims[-1] // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if features.ndim == 2:
            b, c = features.shape
            features = features.view(b, c, 1, 1)

        x = self.in_proj(features)
        for stage in self.stages:
            x = stage(x)

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
