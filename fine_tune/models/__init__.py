"""Pretrained Geospatial Foundation Model Adapters and Factory."""
from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn

from fine_tune.models.base_model import FoundationModelBase, ModalityMismatchError
from fine_tune.models.terramind import TerraMindModel
from fine_tune.models.prithvi import PrithviEO2Model
from fine_tune.models.satmaepp import SatMaePPModel
from fine_tune.models.gfm import GFMCompositionModel
from fine_tune.utils.config import Config


def build_model(
    model_cfg: Config | Dict[str, Any],
    data_cfg: Config | Dict[str, Any] | None = None,
) -> FoundationModelBase:
    """Instantiate a foundation model adapter with configuration and modality validation.

    Args:
        model_cfg: Model section of YAML config.
        data_cfg: Optional data section of YAML config for modality validation.

    Returns:
        FoundationModelBase adapter instance.
    """
    name = str(model_cfg.get("name", "terramind")).lower().replace("-", "_")
    in_channels = int(model_cfg.get("in_channels", data_cfg.get("in_channels", 2) if data_cfg else 2))
    pretrained = str(model_cfg.get("pretrained", ""))
    feature_dim = int(model_cfg.get("feature_dim", 128 if name == "terramind" else 256))

    if "terramind" in name:
        model = TerraMindModel(
            in_channels=in_channels,
            embed_dim=feature_dim,
            pretrained_path_or_repo=pretrained or "ibm-esa-geospatial/TerraMind-1.0-base",
        )
    elif "prithvi" in name:
        model = PrithviEO2Model(
            in_channels=in_channels,
            embed_dim=feature_dim,
            pretrained_path_or_repo=pretrained or "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        )
    elif "satmae" in name:
        model = SatMaePPModel(
            in_channels=in_channels,
            embed_dim=feature_dim,
            pretrained_path_or_repo=pretrained or "BiliSakura/SATMAE-PP-transformers",
        )
    elif "gfm" in name:
        model = GFMCompositionModel(
            in_channels=in_channels,
            embed_dim=feature_dim,
            pretrained_path_or_repo=pretrained or "05kashyap/GFM_Composition_Pretraining",
        )
    else:
        raise ValueError(f"Unknown foundation model: {name}. Choose 'terramind', 'prithvi', 'satmaepp', or 'gfm'.")

    # Modality validation against dataset config
    if data_cfg is not None:
        modality = str(data_cfg.get("modality", "sentinel1"))
        model.validate_modality(dataset_modality=modality, in_channels=in_channels)

    return model


__all__ = [
    "FoundationModelBase",
    "ModalityMismatchError",
    "TerraMindModel",
    "PrithviEO2Model",
    "SatMaePPModel",
    "GFMCompositionModel",
    "build_model",
]
