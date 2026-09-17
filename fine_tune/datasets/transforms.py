"""Data transforms and joint augmentations for geospatial imagery and masks."""
from __future__ import annotations

import random
from typing import Callable, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class ComposeJoint:
    """Compose multiple joint transforms operating on (image, mask) pairs."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlipJoint:
    """Random horizontal flip applied to both image and mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.flip(image, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
        return image, mask


class RandomVerticalFlipJoint:
    """Random vertical flip applied to both image and mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.flip(image, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()
        return image, mask


class RandomRotate90Joint:
    """Random 90, 180, or 270-degree rotation applied to both image and mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k=k, axes=(-2, -1)).copy()
        return image, mask


class SARIntensityJitter:
    """Subtle intensity scaling and Gaussian speckle simulation for SAR imagery."""

    def __init__(self, p: float = 0.3, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            noise = np.random.normal(0.0, 0.02, size=image.shape).astype(image.dtype)
            image = np.clip(image * scale + noise, 0.0, 1.0)
        return image, mask


class ResizeOrPadJoint:
    """Resize or zero-pad/crop image and mask to the target dimensions (H, W)."""

    def __init__(self, target_size: int | Tuple[int, int] = 224):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        th, tw = self.target_size
        c, h, w = image.shape

        if h == th and w == tw:
            return image, mask

        # Convert to tensors for interpolation
        img_t = torch.from_numpy(image).unsqueeze(0).float()  # (1, C, H, W)
        msk_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

        img_resized = F.interpolate(img_t, size=(th, tw), mode="bilinear", align_corners=False)
        msk_resized = F.interpolate(msk_t, size=(th, tw), mode="nearest")

        return img_resized.squeeze(0).numpy(), msk_resized.squeeze(0).squeeze(0).numpy()


def get_transforms(
    split: str = "train",
    image_size: int = 224,
    augment: bool = True,
) -> ComposeJoint:
    """Construct data augmentation pipeline for train or evaluation split."""
    transforms_list = []

    if split == "train" and augment:
        transforms_list.extend([
            RandomHorizontalFlipJoint(p=0.5),
            RandomVerticalFlipJoint(p=0.5),
            RandomRotate90Joint(p=0.5),
            SARIntensityJitter(p=0.3),
        ])

    transforms_list.append(ResizeOrPadJoint(target_size=image_size))
    return ComposeJoint(transforms_list)
