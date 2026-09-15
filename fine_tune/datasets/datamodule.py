"""DataModule and DataLoader factory for geospatial foundation model training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader

from fine_tune.datasets.sen1floods_dataset import Sen1FloodsDataset
from fine_tune.utils.config import Config


def build_dataloaders(
    config: Config | Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build PyTorch DataLoaders for train, validation, and optional test splits.

    Args:
        config: Full or dataset configuration object.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config.get("data", config) if isinstance(config, dict) or hasattr(config, "get") else config
    train_cfg = config.get("training", {}) if isinstance(config, dict) or hasattr(config, "get") else {}

    data_root = data_cfg.get("root", "fine_tune/data/sample_sen1floods11")
    image_size = int(data_cfg.get("image_size", 224))
    in_channels = int(data_cfg.get("in_channels", 2))
    subset = data_cfg.get("subset", "all")
    ignore_index = int(data_cfg.get("ignore_index", -1))

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    # Normalization parameters
    norm_cfg = data_cfg.get("normalize", {})
    vv_min = float(norm_cfg.get("vv_min", -25.0))
    vv_max = float(norm_cfg.get("vv_max", 0.0))
    vh_min = float(norm_cfg.get("vh_min", -32.0))
    vh_max = float(norm_cfg.get("vh_max", -5.0))

    # 1. Train Dataset
    train_dataset = Sen1FloodsDataset(
        data_root=data_root,
        split="train",
        subset=subset,
        image_size=image_size,
        in_channels=in_channels,
        augment=True,
        ignore_index=ignore_index,
        vv_min=vv_min,
        vv_max=vv_max,
        vh_min=vh_min,
        vh_max=vh_max,
    )

    # 2. Validation Dataset
    val_dataset = Sen1FloodsDataset(
        data_root=data_root,
        split="valid",
        subset=subset,
        image_size=image_size,
        in_channels=in_channels,
        augment=False,
        ignore_index=ignore_index,
        vv_min=vv_min,
        vv_max=vv_max,
        vh_min=vh_min,
        vh_max=vh_max,
    )

    # 3. Test Dataset
    test_dataset = Sen1FloodsDataset(
        data_root=data_root,
        split="test",
        subset=subset,
        image_size=image_size,
        in_channels=in_channels,
        augment=False,
        ignore_index=ignore_index,
        vv_min=vv_min,
        vv_max=vv_max,
        vh_min=vh_min,
        vh_max=vh_max,
    )

    # If test set is empty, return None for test_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
