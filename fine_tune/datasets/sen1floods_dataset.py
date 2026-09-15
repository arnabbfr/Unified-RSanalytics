"""Dataset adapter for Sen1Floods11 Sentinel-1 SAR Flood Segmentation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from fine_tune.datasets.transforms import get_transforms


def read_geotiff(path: Path | str) -> np.ndarray:
    """Read multi-band GeoTIFF or standard raster image safely.

    Returns:
        NumPy array of shape (C, H, W) or (H, W).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image raster file not found: {path}")

    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            return arr.astype(np.float32)

    # Fallback via PIL/Pillow or NumPy
    try:
        from PIL import Image
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            return arr
        elif arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        return arr
    except Exception as exc:
        raise RuntimeError(f"Failed to read raster at {path}: {exc}") from exc


def normalize_sar(
    arr: np.ndarray,
    vv_min: float = -25.0,
    vv_max: float = 0.0,
    vh_min: float = -32.0,
    vh_max: float = -5.0,
) -> np.ndarray:
    """Decibel conversion and min-max feature normalization for Sentinel-1 (VV, VH)."""
    normed = np.zeros_like(arr, dtype=np.float32)

    # Channel 0: VV
    vv = arr[0]
    # Check if raw linear power or already dB
    if np.nanmin(vv) >= 0.0 and np.nanmax(vv) > 5.0:
        vv_db = 10.0 * np.log10(np.clip(vv, 1e-5, None))
    else:
        vv_db = vv
    normed[0] = np.clip((vv_db - vv_min) / (vv_max - vv_min + 1e-6), 0.0, 1.0)

    # Channel 1: VH (if present)
    if arr.shape[0] > 1:
        vh = arr[1]
        if np.nanmin(vh) >= 0.0 and np.nanmax(vh) > 5.0:
            vh_db = 10.0 * np.log10(np.clip(vh, 1e-5, None))
        else:
            vh_db = vh
        normed[1] = np.clip((vh_db - vh_min) / (vh_max - vh_min + 1e-6), 0.0, 1.0)

    # For any additional channels (e.g. ratio or optical)
    for c in range(2, arr.shape[0]):
        c_min = float(np.nanmin(arr[c]))
        c_max = float(np.nanmax(arr[c]))
        normed[c] = np.clip((arr[c] - c_min) / (c_max - c_min + 1e-6), 0.0, 1.0)

    return np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)


class Sen1FloodsDataset(Dataset):
    """PyTorch Dataset for Sen1Floods11 SAR flood inundation mapping."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        subset: str = "all",
        image_size: int = 224,
        in_channels: int = 2,
        transform: Optional[Callable] = None,
        augment: bool = True,
        ignore_index: int = -1,
        vv_min: float = -25.0,
        vv_max: float = 0.0,
        vh_min: float = -32.0,
        vh_max: float = -5.0,
    ):
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.subset = subset.lower()
        self.image_size = image_size
        self.in_channels = in_channels
        self.ignore_index = ignore_index
        self.vv_min = vv_min
        self.vv_max = vv_max
        self.vh_min = vh_min
        self.vh_max = vh_max

        self.transform = transform or get_transforms(
            split=self.split,
            image_size=image_size,
            augment=augment,
        )

        self.samples: List[Tuple[Path, Path]] = []
        self._discover_dataset_files()

    def _discover_dataset_files(self) -> None:
        """Locate image/mask pairs across standard Sen1Floods11 layouts."""
        if not self.data_root.exists():
            return

        # Strategy 1: Split CSV file (Sen1Floods11 official splits)
        # e.g., splits/flood_handlabeled/flood_train_data.csv
        split_csv_candidates = [
            self.data_root / "splits" / "flood_handlabeled" / f"flood_{self.split}_data.csv",
            self.data_root / f"flood_{self.split}_data.csv",
            self.data_root / f"{self.split}.csv",
        ]

        csv_found = False
        for csv_path in split_csv_candidates:
            if csv_path.is_file():
                self._load_from_csv(csv_path)
                csv_found = True
                break

        if csv_found and len(self.samples) > 0:
            return

        # Strategy 2: Structured split subdirectories
        # e.g. data_root / split / "images" & data_root / split / "masks"
        split_img_dir = self.data_root / self.split / "images"
        split_msk_dir = self.data_root / self.split / "masks"
        if split_img_dir.is_dir() and split_msk_dir.is_dir():
            self._load_from_dir_pair(split_img_dir, split_msk_dir)
            if len(self.samples) > 0:
                return

        # Strategy 3: Global HandLabeled directory
        # e.g. data_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        s1_hand = self.data_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        label_hand = self.data_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
        if not s1_hand.is_dir():
            s1_hand = self.data_root / "S1Hand"
            label_hand = self.data_root / "LabelHand"

        if s1_hand.is_dir() and label_hand.is_dir():
            all_s1 = sorted(list(s1_hand.glob("*.tif*")))
            for img_path in all_s1:
                # Corresponding mask has _LabelHand or same basename in label_hand dir
                msk_name = img_path.name.replace("_S1Hand", "_LabelHand")
                msk_path = label_hand / msk_name
                if not msk_path.is_file():
                    msk_path = label_hand / img_path.name
                if msk_path.is_file():
                    self.samples.append((img_path, msk_path))

            # Apply pseudo train/val/test split if no CSV was provided
            if len(self.samples) > 0:
                self._apply_hash_split()
                return

        # Strategy 4: Top-level images/ and masks/
        img_dir = self.data_root / "images"
        msk_dir = self.data_root / "masks"
        if img_dir.is_dir() and msk_dir.is_dir():
            self._load_from_dir_pair(img_dir, msk_dir)
            if len(self.samples) > 0:
                self._apply_hash_split()

    def _load_from_csv(self, csv_path: Path) -> None:
        """Parse Sen1Floods11 split CSV containing pair filenames."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                img_rel = row[0].strip()
                msk_rel = row[1].strip() if len(row) > 1 else img_rel.replace("_S1Hand", "_LabelHand")

                if self.subset != "all" and self.subset not in img_rel.lower():
                    continue

                img_path = self.data_root / img_rel
                msk_path = self.data_root / msk_rel

                if not img_path.is_file():
                    # Check relative to CSV parent or data_root / S1Hand
                    alt_img = self.data_root / "S1Hand" / Path(img_rel).name
                    alt_msk = self.data_root / "LabelHand" / Path(msk_rel).name
                    if alt_img.is_file() and alt_msk.is_file():
                        img_path, msk_path = alt_img, alt_msk

                if img_path.is_file() and msk_path.is_file():
                    self.samples.append((img_path, msk_path))

    def _load_from_dir_pair(self, img_dir: Path, msk_dir: Path) -> None:
        """Match image and mask files by filename or ID."""
        for img_path in sorted(list(img_dir.glob("*.tif*")) + list(img_dir.glob("*.png"))):
            if self.subset != "all" and self.subset not in img_path.name.lower():
                continue
            msk_path = msk_dir / img_path.name
            if not msk_path.is_file():
                msk_name = img_path.name.replace("image", "mask").replace("s1", "label").replace("_S1Hand", "_LabelHand")
                msk_path = msk_dir / msk_name

            if msk_path.is_file():
                self.samples.append((img_path, msk_path))

    def _apply_hash_split(self) -> None:
        """Deterministic split (70% train, 15% val, 15% test) based on sample hash."""
        filtered = []
        for img_p, msk_p in self.samples:
            h = abs(hash(img_p.name)) % 100
            if self.split == "train" and h < 70:
                filtered.append((img_p, msk_p))
            elif self.split in ("val", "valid", "validation") and 70 <= h < 85:
                filtered.append((img_p, msk_p))
            elif self.split == "test" and h >= 85:
                filtered.append((img_p, msk_p))
        self.samples = filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        img_path, msk_path = self.samples[idx]

        # 1. Read SAR image (C, H, W)
        raw_img = read_geotiff(img_path)
        if raw_img.ndim == 2:
            raw_img = raw_img[np.newaxis, ...]  # (1, H, W)

        # 2. Read Ground Truth mask (H, W)
        raw_msk = read_geotiff(msk_path)
        if raw_msk.ndim == 3:
            raw_msk = raw_msk[0]  # Take 1st band

        # 3. Normalize SAR VV/VH
        norm_img = normalize_sar(
            raw_img,
            vv_min=self.vv_min,
            vv_max=self.vv_max,
            vh_min=self.vh_min,
            vh_max=self.vh_max,
        )

        # 4. Ensure channel dimension matches self.in_channels
        if norm_img.shape[0] > self.in_channels:
            norm_img = norm_img[:self.in_channels]
        elif norm_img.shape[0] < self.in_channels:
            pad_width = ((0, self.in_channels - norm_img.shape[0]), (0, 0), (0, 0))
            norm_img = np.pad(norm_img, pad_width, mode="edge")

        # 5. Standardize mask values (0: non-flood, 1: flood, ignore: -1/255)
        mask = raw_msk.astype(np.int64)

        # 6. Apply joint transforms
        if self.transform is not None:
            norm_img, mask = self.transform(norm_img, mask)

        img_tensor = torch.from_numpy(norm_img).float()
        mask_tensor = torch.from_numpy(mask).long()

        meta = {
            "image_path": str(img_path),
            "mask_path": str(msk_path),
            "sample_id": img_path.stem,
        }

        return img_tensor, mask_tensor, meta
