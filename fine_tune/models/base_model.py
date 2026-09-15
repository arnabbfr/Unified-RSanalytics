"""Common abstract base interface and modality validator for all Geospatial Foundation Models."""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn


class ModalityMismatchError(ValueError):
    """Raised when dataset input modality is incompatible with the foundation model's pretrained architecture."""
    pass


class FoundationModelBase(nn.Module, abc.ABC):
    """Abstract base class for remote sensing foundation model adapters."""

    def __init__(self, model_name: str, expected_modalities: List[str], feature_dim: int):
        super().__init__()
        self.model_name = model_name
        self.expected_modalities = expected_modalities
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        """Channel dimension of the extracted spatial feature maps."""
        return self._feature_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract multi-scale or spatial feature maps suitable for segmentation decoders.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature map tensor of shape (B, feature_dim, H_feat, W_feat).
        """
        raise NotImplementedError

    def get_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Alias for forward feature extraction."""
        return self.forward(x, **kwargs)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for linear probing / decoder-only fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    @abc.abstractmethod
    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze the last N transformer or convolutional blocks."""
        raise NotImplementedError

    def parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Construct differential parameter groups with weight decay separation."""
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay on bias and normalization layers
            if "bias" in name or "norm" in name or "bn" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = []
        if decay_params:
            groups.append({
                "params": decay_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            })
        if no_decay_params:
            groups.append({
                "params": no_decay_params,
                "lr": backbone_lr,
                "weight_decay": 0.0,
            })
        return groups

    def validate_modality(self, dataset_modality: str, in_channels: int) -> None:
        """Strictly validate dataset modality against pretrained foundation model capabilities.

        Raises:
            ModalityMismatchError: If the input data violates the pretrained model's modality assumptions.
        """
        modality_lower = dataset_modality.lower()
        valid = any(exp.lower() in modality_lower for exp in self.expected_modalities)

        if not valid:
            raise ModalityMismatchError(
                f"\n{'=' * 75}\n"
                f"[MODALITY MISMATCH ERROR] Model '{self.model_name}' cannot process dataset modality '{dataset_modality}'.\n"
                f"  • Expected Modalities: {', '.join(self.expected_modalities)}\n"
                f"  • Provided Input:     {dataset_modality} with {in_channels} channels\n"
                f"  • Explanation: Pretrained checkpoints require compatible spectral/temporal sensor pathways.\n"
                f"    Do NOT silently reshape or spoof channel configurations without proper data or adapter projections.\n"
                f"{'=' * 75}"
            )

    def param_counts(self) -> Tuple[float, float]:
        """Returns (total_params_million, trainable_params_million)."""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return round(total, 2), round(trainable, 2)
