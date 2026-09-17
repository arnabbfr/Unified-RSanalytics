"""Segmentation evaluation metrics with focus on imbalanced flood class."""
from __future__ import annotations

from typing import Dict
import torch


class SegmentationMetricsTracker:
    """Accumulates confusion matrix counts across batches for accurate epoch-level metrics."""

    def __init__(self, threshold: float = 0.5, ignore_index: int = -1):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Reset accumulated confusion matrix statistics."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.num_batches = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Update running metrics with a new batch of predictions and targets.

        Args:
            logits: Output logits of shape (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth (0, 1, or ignore_index) of shape (B, 1, H, W) or (B, H, W).

        Returns:
            Dictionary of batch-level metrics.
        """
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).long()

        # Valid mask (filter out ignored pixels)
        valid = (targets != self.ignore_index) & (targets != 255)
        targets_binary = (targets == 1).long()

        preds_valid = preds[valid]
        targets_valid = targets_binary[valid]

        tp = int(((preds_valid == 1) & (targets_valid == 1)).sum().item())
        fp = int(((preds_valid == 1) & (targets_valid == 0)).sum().item())
        fn = int(((preds_valid == 0) & (targets_valid == 1)).sum().item())
        tn = int(((preds_valid == 0) & (targets_valid == 0)).sum().item())

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        self.num_batches += 1

        return self._compute_scores(tp, fp, fn, tn)

    def compute(self) -> Dict[str, float]:
        """Compute aggregated epoch-level metrics from all accumulated counts."""
        return self._compute_scores(
            self.total_tp, self.total_fp, self.total_fn, self.total_tn
        )

    @staticmethod
    def _compute_scores(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        eps = 1e-7

        # Flood Class Metrics (Positive Class = 1)
        flood_iou = tp / (tp + fp + fn + eps) if (tp + fp + fn) > 0 else 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0
        flood_dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps) if (2 * tp + fp + fn) > 0 else 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0
        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp + eps) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "iou": round(float(flood_iou), 4),
            "dice": round(float(flood_dice), 4),
            "f1": round(float(flood_dice), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(float(specificity), 4),
            "accuracy": round(float(accuracy), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
