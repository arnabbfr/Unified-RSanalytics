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
    """Read multi-band GeoTIFF, NumPy array, or standard raster image safely.

    Returns:
        NumPy array of shape (C, H, W) or (H, W).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image raster file not found: {path}")

    # 1. NumPy file support (.npy, .npz)
    if path.suffix.lower() in (".npy", ".npz"):
        try:
            arr = np.load(path)
            if hasattr(arr, "files"):
                arr = arr[arr.files[0]]
            arr = arr.astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4, 6, 8, 10, 12) and arr.shape[0] > 12:
                arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            return arr
        except Exception as exc:
            raise RuntimeError(f"Failed to load numpy array from {path}: {exc}") from exc

    # 2. Rasterio GeoTIFF reading
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(path) as src:
                arr = src.read()  # (bands, H, W)
                return arr.astype(np.float32)
        except Exception:
            pass

    # 3. Fallback via PIL/Pillow
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
    vv_min: float = -30.0,
    vv_max: float = 0.0,
    vh_min: float = -35.0,
    vh_max: float = -5.0,
) -> np.ndarray:
    """Decibel conversion and robust min-max normalization for Sentinel-1 (VV, VH)."""
    normed = np.zeros_like(arr, dtype=np.float32)

    for c in range(arr.shape[0]):
        ch = arr[c]
        if np.all(np.isnan(ch)):
            normed[c] = 0.0
            continue

        valid = ch[~np.isnan(ch)]
        if valid.size == 0:
            normed[c] = 0.0
            continue

        c_min = float(np.min(valid))

        # 1. Determine if channel is linear power (non-negative) or already dB
        if c_min >= 0.0:
            ch_clipped = np.clip(ch, 1e-5, None)
            ch_db = 10.0 * np.log10(ch_clipped)
        else:
            ch_db = ch

        # 2. Channel-specific dB bounds
        if c == 0:
            b_min, b_max = vv_min, vv_max
        elif c == 1:
            b_min, b_max = vh_min, vh_max
        else:
            p2 = float(np.nanpercentile(ch_db, 2))
            p98 = float(np.nanpercentile(ch_db, 98))
            b_min, b_max = p2, max(p98, p2 + 1e-4)

        normed[c] = np.clip((ch_db - b_min) / (b_max - b_min + 1e-6), 0.0, 1.0)

    return np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)


def resolve_dataset_path(data_root: str | Path) -> Path:
    """Resolve dataset path with auto-discovery for Kaggle and Colab environments."""
    path = Path(data_root)
    if path.exists():
        return path

    # Auto-detect Kaggle input directory
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for candidate in kaggle_root.iterdir():
            if candidate.is_dir():
                name_lower = candidate.name.lower()
                if any(k in name_lower for k in ("sen1", "flood", "8channel", "dataset")):
                    print(f"[Dataset Auto-Resolution] Located Kaggle dataset at: {candidate}")
                    return candidate
        subdirs = [p for p in kaggle_root.iterdir() if p.is_dir()]
        if subdirs:
            print(f"[Dataset Auto-Resolution] Using Kaggle dataset: {subdirs[0]}")
            return subdirs[0]

    return path


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
        self.data_root = resolve_dataset_path(data_root)
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
        """Locate image/mask pairs across standard Sen1Floods11 and Kaggle 8-channel layouts."""
        if not self.data_root.exists():
            print(f"[Dataset Warning] Specified root does not exist: {self.data_root}")
            return

        # 1. Search for Split CSV Files
        split_csv_candidates = [
            self.data_root / "splits" / "flood_handlabeled" / f"flood_{self.split}_data.csv",
            self.data_root / f"flood_{self.split}_data.csv",
            self.data_root / f"{self.split}.csv",
        ]
        for csv_path in split_csv_candidates:
            if csv_path.is_file():
                self._load_from_csv(csv_path)
                if len(self.samples) > 0:
                    print(f"[Dataset Discovery] Split '{self.split}': Loaded {len(self.samples)} samples from CSV {csv_path.name}")
                    return

        # 2. Single-pass fast recursive file discovery
        VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz"}
        all_files = [
            p for p in self.data_root.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ]

        if not all_files:
            # Check one level up or down if data_root was slightly off
            parent_root = self.data_root.parent
            if parent_root.exists() and parent_root != self.data_root:
                all_files = [
                    p for p in parent_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in VALID_EXTS
                ]

        if not all_files:
            print(f"[Dataset Warning] No raster files (.tif/.png/.npy) found in {self.data_root}")
            return

        def extract_base_id(path: Path) -> str:
            s = path.stem.lower()
            for noise in (
                "_labelhand", "_s1hand", "_8channel", "_label", "_mask",
                "_image", "_qc", "_gt", "label", "mask", "target", "source"
            ):
                s = s.replace(noise, "")
            return s.strip("_- ")

        # Separate masks from images
        mask_files = []
        img_files = []
        for f in all_files:
            name_lower = f.name.lower()
            if any(k in name_lower for k in ("label", "mask", "target", "groundtruth", "gt", "qc")):
                mask_files.append(f)
            else:
                img_files.append(f)

        matched_pairs = []

        if mask_files and img_files:
            # Build fast hash map on cleaned base ID
            mask_map = {extract_base_id(m): m for m in mask_files}
            mask_name_map = {m.name: m for m in mask_files}

            for img in img_files:
                base_id = extract_base_id(img)
                # Try ID match
                if base_id in mask_map:
                    matched_pairs.append((img, mask_map[base_id]))
                else:
                    # Try direct pattern replacements
                    for alt_name in (
                        img.name.replace("_S1Hand", "_LabelHand").replace("_8Channel", "_LabelHand"),
                        img.name.replace("image", "mask").replace("s1", "label"),
                    ):
                        if alt_name in mask_name_map:
                            matched_pairs.append((img, mask_name_map[alt_name]))
                            break

        if not matched_pairs and img_files:
            # If all files are self-contained multi-band / 8-channel rasters
            matched_pairs = [(img, img) for img in img_files]
        elif not matched_pairs and all_files:
            # Fallback pairing
            matched_pairs = [(f, f) for f in all_files]

        # Apply subset filter if specified
        if self.subset != "all":
            matched_pairs = [p for p in matched_pairs if self.subset in p[0].name.lower()]

        # Apply train / val / test hash split
        self._all_discovered_samples = matched_pairs
        self.samples = self._apply_hash_split(matched_pairs)

        # Guarantee at least some samples if hash split had zero matches
        if len(self.samples) == 0 and len(matched_pairs) > 0:
            n = len(matched_pairs)
            n_train = max(1, int(0.7 * n))
            n_val = max(1, int(0.15 * n))
            if self.split == "train":
                self.samples = matched_pairs[:n_train]
            elif self.split in ("val", "valid", "validation"):
                self.samples = matched_pairs[n_train:n_train + n_val] if n > 1 else matched_pairs
            else:
                self.samples = matched_pairs[n_train + n_val:] if n > 2 else matched_pairs

        print(f"[Dataset Discovery] Split '{self.split}': Loaded {len(self.samples)} sample pairs (Total available: {len(matched_pairs)})")

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

    def _apply_hash_split(self, pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
        """Deterministic split (70% train, 15% val, 15% test) based on sample hash."""
        filtered = []
        for img_p, msk_p in pairs:
            h = abs(hash(img_p.name)) % 100
            if self.split == "train" and h < 70:
                filtered.append((img_p, msk_p))
            elif self.split in ("val", "valid", "validation") and 70 <= h < 85:
                filtered.append((img_p, msk_p))
            elif self.split == "test" and h >= 85:
                filtered.append((img_p, msk_p))
        return filtered

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

        # 4. Construct 3-channel SAR representation [VV, VH, VV - VH] if in_channels == 3
        if norm_img.shape[0] == 2 and self.in_channels == 3:
            diff = np.clip(norm_img[0] - norm_img[1] + 0.5, 0.0, 1.0)
            norm_img = np.stack([norm_img[0], norm_img[1], diff], axis=0)
        elif norm_img.shape[0] > self.in_channels:
            norm_img = norm_img[:self.in_channels]
        elif norm_img.shape[0] < self.in_channels:
            pad_width = ((0, self.in_channels - norm_img.shape[0]), (0, 0), (0, 0))
            norm_img = np.pad(norm_img, pad_width, mode="edge")

        # 5. Standardize mask values (0: non-flood, 1: flood, ignore: -1/255)
        raw_mask_int = raw_msk.astype(np.int64)
        unique_vals = np.unique(raw_mask_int)
        mask = np.copy(raw_mask_int)

        # Handle binary {0, 255} masks where 255 represents foreground water
        if set(unique_vals).issubset({0, 255}) and 255 in unique_vals:
            mask[mask == 255] = 1
        elif 255 in unique_vals and 1 in unique_vals:
            # 255 is no-data / ignore index
            mask[mask == 255] = -1

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
