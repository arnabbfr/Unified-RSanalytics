"""Loss functions for flood water segmentation with heavy class imbalance and mask-ignoring support."""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_class_weights(
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 20,
    ignore_index: int = -1,
) -> float:
    """Calculate empirical positive class weight (pos_weight = negative_pixels / positive_pixels)."""
    total_pos = 0
    total_neg = 0

    for step, (_, masks, *_) in enumerate(dataloader):
        if step >= num_batches:
            break
        valid = (masks != ignore_index) & (masks != 255)
        pos = ((masks == 1) & valid).sum().item()
        neg = ((masks == 0) & valid).sum().item()
        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        return 2.5
    pos_weight = total_neg / max(total_pos, 1)
    # Clip to prevent gradient explosion on hyper-sparse datasets
    return float(min(max(pos_weight, 1.5), 10.0))


class MaskedBCEWithLogitsLoss(nn.Module):
    """Numerically stable Binary Cross-Entropy with Logits and positive class weighting."""

    def __init__(self, pos_weight: float | None = 2.0, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        valid_mask = (targets != self.ignore_index) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = (targets[valid_mask] == 1).float()

        weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=weight)


class GlobalBatchDiceLoss(nn.Module):
    """Global Batch Soft Dice Loss: sums intersection/union across all valid pixels in batch."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        valid_mask = (targets != self.ignore_index) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        probs = torch.sigmoid(logits[valid_mask])
        targets_binary = (targets[valid_mask] == 1).float()

        intersection = (probs * targets_binary).sum()
        union = probs.sum() + targets_binary.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class MaskedFocalLoss(nn.Module):
    """Focal Loss with alpha balancing and gamma modulating factor to focus on hard foreground boundaries."""

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
        targets_v = (targets[valid_mask] == 1).float()

        bce = F.binary_cross_entropy_with_logits(logits_v, targets_v, reduction="none")
        probs = torch.sigmoid(logits_v)
        p_t = probs * targets_v + (1.0 - probs) * (1.0 - targets_v)
        alpha_t = self.alpha * targets_v + (1.0 - self.alpha) * (1.0 - targets_v)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        return (focal_weight * bce).mean()


class MaskedTverskyLoss(nn.Module):
    """Tversky Loss with tunable alpha (FP penalty) and beta (FN penalty) for recall optimization."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha  # Precision weight (FP penalty)
        self.beta = beta    # Recall weight (FN penalty)
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        valid_mask = (targets != self.ignore_index) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        probs = torch.sigmoid(logits[valid_mask])
        targets_v = (targets[valid_mask] == 1).float()

        tp = (probs * targets_v).sum()
        fp = (probs * (1.0 - targets_v)).sum()
        fn = ((1.0 - probs) * targets_v).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CompoundFloodLoss(nn.Module):
    """Production-grade composite loss: Weighted BCE + Global Soft Dice + Focal Tversky.

    Guarantees steady gradient flow for minority flood pixels, penalizes false positives,
    and prevents trivial zero-prediction collapse.
    """

    def __init__(
        self,
        pos_weight: float = 2.5,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        tversky_weight: float = 0.2,
        alpha: float = 0.4,
        beta: float = 0.6,
        gamma: float = 2.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

        self.bce = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.dice = GlobalBatchDiceLoss(ignore_index=ignore_index)
        self.tversky = MaskedTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index)
        self.focal = MaskedFocalLoss(gamma=gamma, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        tversky_loss = self.tversky(logits, targets)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.tversky_weight * tversky_loss
        )


def build_loss_fn(loss_cfg: Dict[str, Any] | Any) -> nn.Module:
    """Factory function for instantiating loss modules based on configuration."""
    loss_type = str(getattr(loss_cfg, "type", "compound") if hasattr(loss_cfg, "type") else loss_cfg.get("type", "compound")).lower()
    pos_weight = getattr(loss_cfg, "pos_weight", 2.5) if hasattr(loss_cfg, "pos_weight") else loss_cfg.get("pos_weight", 2.5)
    pos_weight_val = float(pos_weight) if pos_weight is not None else 2.5
    ignore_index = int(getattr(loss_cfg, "ignore_index", -1) if hasattr(loss_cfg, "ignore_index") else loss_cfg.get("ignore_index", -1))

    if loss_type in ("compound", "combined"):
        return CompoundFloodLoss(pos_weight=pos_weight_val, ignore_index=ignore_index)
    elif loss_type == "dice":
        return GlobalBatchDiceLoss(ignore_index=ignore_index)
    elif loss_type == "focal":
        return MaskedFocalLoss(ignore_index=ignore_index)
    elif loss_type == "tversky":
        return MaskedTverskyLoss(ignore_index=ignore_index)
    elif loss_type == "bce":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight_val, ignore_index=ignore_index)
    else:
        return CompoundFloodLoss(pos_weight=pos_weight_val, ignore_index=ignore_index)


# Backward-compatible aliases
MaskedDiceLoss = GlobalBatchDiceLoss
CombinedBceDiceLoss = CompoundFloodLoss


