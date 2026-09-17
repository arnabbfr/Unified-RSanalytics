"""Adaptive convolutional segmentation decoder for foundation model feature maps."""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Convolution -> BatchNorm -> GELU -> Dropout block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """Feature-adaptive multi-scale convolutional segmentation decoder.

    Adapts dynamically to any backbone feature dimension (e.g. 128, 256, 768, 1024)
    and produces crisp full-resolution (H, W) 1-channel flood logits.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial projection to bridge backbone dimension to decoder dimension
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.GELU(),
        )

        # Upsampling stages
        stages = []
        curr_in = hidden_dims[0]
        for h_dim in hidden_dims:
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBlock(curr_in, h_dim, dropout=dropout),
                )
            )
            curr_in = h_dim

        self.stages = nn.ModuleList(stages)

        # Final prediction head
        self.head = nn.Conv2d(hidden_dims[-1], num_classes, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Args:
            features: Feature map tensor of shape (B, in_channels, H_feat, W_feat).
            target_size: Optional (H, W) of original input to match exactly.

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        if isinstance(features, dict):
            features = features["out"]
        elif isinstance(features, (tuple, list)):
            features = features[0]

        # Ensure 4D tensor (B, C, H, W)
        if features.ndim == 2:
            # Flattened embedding (B, C) -> reshape to (B, C, 1, 1) or (B, C, 7, 7)
            b, c = features.shape
            features = features.view(b, c, 1, 1)

        x = self.proj(features)
        for stage in self.stages:
            x = stage(x)

        logits = self.head(x)

        if target_size is not None and (logits.shape[-2] != target_size[0] or logits.shape[-1] != target_size[1]):
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits
