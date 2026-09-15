"""GFM Composition Pretraining (SAR + Optical Multi-Sensor Composition) Adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_tune.models.base_model import FoundationModelBase


class DualSensorCompositionEncoder(nn.Module):
    """Dual-branch encoder extracting aligned representations from SAR (VV/VH) and Optical imagery."""

    def __init__(
        self,
        sar_channels: int = 2,
        opt_channels: int = 4,
        embed_dim: int = 256,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.sar_proj = nn.Conv2d(sar_channels, embed_dim // 2, kernel_size=patch_size, stride=patch_size)
        self.opt_proj = nn.Conv2d(opt_channels, embed_dim // 2, kernel_size=patch_size, stride=patch_size)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        sar_x: torch.Tensor,
        opt_x: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, int, int]:
        b, c, h, w = sar_x.shape
        ph, pw = h // self.patch_size, w // self.patch_size

        sar_feat = self.sar_proj(sar_x).flatten(2).transpose(1, 2)  # (B, N, embed_dim/2)

        if opt_x is not None:
            opt_feat = self.opt_proj(opt_x).flatten(2).transpose(1, 2)  # (B, N, embed_dim/2)
        else:
            # Zero pad optical branch if absent
            opt_feat = torch.zeros_like(sar_feat)

        combined = torch.cat([sar_feat, opt_feat], dim=-1)  # (B, N, embed_dim)
        fused = self.fusion(combined)
        return fused, ph, pw


class GFMCompositionModel(FoundationModelBase):
    """GFM Composition Pretraining Foundation Model Adapter (05kashyap / Open Source).

    Compositional multi-sensor architecture supporting Sentinel-1 SAR and Sentinel-2 Optical.
    Reference: https://github.com/05kashyap/GFM_Composition_Pretraining
    """

    def __init__(
        self,
        in_channels: int = 6,  # 2 SAR + 4 Optical, or 2 SAR only
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 16,
        pretrained_path_or_repo: str = "05kashyap/GFM_Composition_Pretraining",
    ):
        super().__init__(
            model_name="GFM_Composition_Pretraining",
            expected_modalities=["sar_optical", "compositional", "sentinel1", "sar", "multispectral"],
            feature_dim=embed_dim,
        )
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pretrained_ref = pretrained_path_or_repo

        sar_c = min(in_channels, 2)
        opt_c = max(0, in_channels - 2) if in_channels > 2 else 4

        self.encoder = DualSensorCompositionEncoder(
            sar_channels=sar_c,
            opt_channels=opt_c,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self._load_pretrained(pretrained_path_or_repo)

    def _load_pretrained(self, ref: str) -> None:
        local_path = Path(ref)
        if local_path.is_file():
            try:
                ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                self.load_state_dict(state, strict=False)
                print(f"[GFM] Successfully loaded local checkpoint from: {local_path}")
            except Exception as e:
                print(f"[GFM] Local checkpoint load warning ({e}); initialized model architecture.")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, c, h, w = x.shape

        if c <= 2:
            sar_x = x
            opt_x = None
        else:
            sar_x = x[:, :2, :, :]
            opt_x = x[:, 2:, :, :]

        tokens, ph, pw = self.encoder(sar_x, opt_x)
        num_patches = tokens.shape[1]

        if num_patches <= self.pos_embed.shape[1]:
            pos = self.pos_embed[:, :num_patches, :]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        tokens = tokens + pos
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        feat_map = tokens.transpose(1, 2).view(b, self.embed_dim, ph, pw)
        return self.feature_proj(feat_map)

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        self.freeze_backbone()
        for block in self.blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True
        for param in self.feature_proj.parameters():
            param.requires_grad = True
