"""Loss functions for flood water segmentation with mask-ignoring support."""
from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """Numerically stable Binary Cross-Entropy with Logits, ignoring invalid pixels (-1, 255)."""

    def __init__(self, pos_weight: float | None = None, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Predicted raw logits of shape (B, 1, H, W) or (B, H, W).
            targets: Ground truth binary mask (0, 1, or ignore_index) of matching shape.
        """
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        # Create valid mask
        valid_mask = (targets != self.ignore_index) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask].float()

        weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=weight)


class MaskedDiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation with smooth factor and ignore-mask support."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)
        valid_mask = (targets != self.ignore_index) & (targets != 255)

        if not valid_mask.any():
            return logits.sum() * 0.0

        probs_valid = probs * valid_mask.float()
        targets_valid = (targets == 1).float() * valid_mask.float()

        # Batch-wise Dice computation
        probs_flat = probs_valid.view(probs.size(0), -1)
        targets_flat = targets_valid.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class MaskedFocalLoss(nn.Module):
    """Focal Loss with ignore index support, down-weighting easy background pixels."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        valid_mask = (targets != self.ignore_index) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        logits_v = logits[valid_mask]
        targets_v = targets[valid_mask].float()

        bce = F.binary_cross_entropy_with_logits(logits_v, targets_v, reduction="none")
        probs = torch.sigmoid(logits_v)
        p_t = probs * targets_v + (1 - probs) * (1 - targets_v)
        alpha_t = self.alpha * targets_v + (1 - self.alpha) * (1 - targets_v)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)

        return (focal_weight * bce).mean()


class MaskedTverskyLoss(nn.Module):
    """Tversky Loss for optimizing high-recall flood water boundaries."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha  # FP penalty
        self.beta = beta    # FN penalty (higher beta = higher recall for flood)
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)
        valid_mask = (targets != self.ignore_index) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        p = probs * valid_mask.float()
        g = (targets == 1).float() * valid_mask.float()

        p_flat = p.view(p.size(0), -1)
        g_flat = g.view(g.size(0), -1)

        tp = (p_flat * g_flat).sum(dim=1)
        fp = (p_flat * (1 - g_flat)).sum(dim=1)
        fn = ((1 - p_flat) * g_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()


class CombinedBceDiceLoss(nn.Module):
    """Weighted combination of Masked BCE, Focal, and Soft Dice / Tversky Loss."""

    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.7,
        pos_weight: float | None = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.dice = MaskedDiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalTverskyLoss(nn.Module):
    """Hybrid Focal + Tversky loss tailored for sparse Earth Observation flood inundation."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 2.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal = MaskedFocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.tversky = MaskedTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        f_loss = self.focal(logits, targets)
        t_loss = self.tversky(logits, targets)
        return self.focal_weight * f_loss + self.tversky_weight * t_loss


def build_loss_fn(loss_cfg: Dict[str, Any] | Any) -> nn.Module:
    """Build the loss function module based on YAML configuration."""
    loss_type = str(getattr(loss_cfg, "type", "combined") if hasattr(loss_cfg, "type") else loss_cfg.get("type", "combined")).lower()
    bce_weight = float(getattr(loss_cfg, "bce_weight", 0.3) if hasattr(loss_cfg, "bce_weight") else loss_cfg.get("bce_weight", 0.3))
    dice_weight = float(getattr(loss_cfg, "dice_weight", 0.7) if hasattr(loss_cfg, "dice_weight") else loss_cfg.get("dice_weight", 0.7))
    pos_weight = getattr(loss_cfg, "pos_weight", None) if hasattr(loss_cfg, "pos_weight") else loss_cfg.get("pos_weight", None)
    pos_weight_val = float(pos_weight) if pos_weight is not None else None
    ignore_index = int(getattr(loss_cfg, "ignore_index", -1) if hasattr(loss_cfg, "ignore_index") else loss_cfg.get("ignore_index", -1))

    if loss_type == "bce":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight_val, ignore_index=ignore_index)
    elif loss_type == "dice":
        return MaskedDiceLoss(ignore_index=ignore_index)
    elif loss_type == "focal":
        return MaskedFocalLoss(ignore_index=ignore_index)
    elif loss_type == "tversky":
        return MaskedTverskyLoss(ignore_index=ignore_index)
    elif loss_type in ("focal_tversky", "focaltversky"):
        return FocalTverskyLoss(ignore_index=ignore_index)
    elif loss_type == "combined":
        return CombinedBceDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight_val,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Choose 'combined', 'focal_tversky', 'bce', 'dice', or 'tversky'.")

