"""Dataset preparation, synthetic benchmark generator, and Sen1Floods11 download helper."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_synthetic_sen1floods_dataset(
    output_dir: str | Path = "fine_tune/data/sample_sen1floods11",
    num_train: int = 10,
    num_val: int = 4,
    num_test: int = 4,
    image_size: int = 224,
) -> None:
    """Generate a realistic synthetic Sen1Floods11 SAR dataset for fast local sanity checks."""
    out_path = Path(output_dir)
    s1_dir = out_path / "S1Hand"
    label_dir = out_path / "LabelHand"
    splits_dir = out_path / "splits" / "flood_handlabeled"

    s1_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    regions = ["India_Assam", "India_Bihar", "India_Kerala", "Bolivia_Beni", "USA_Mekong", "Ghana_Volta"]

    splits = {
        "train": num_train,
        "valid": num_val,
        "test": num_test,
    }

    all_pairs = []
    split_records = {"train": [], "valid": [], "test": []}

    idx = 0
    for split_name, count in splits.items():
        for i in range(count):
            reg = regions[idx % len(regions)]
            base_name = f"{reg}_{idx:04d}"
            s1_filename = f"{base_name}_S1Hand.tif"
            label_filename = f"{base_name}_LabelHand.tif"

            # 1. Generate synthetic SAR VV/VH raster (C=2, H=224, W=224)
            # Land surface background: VV ~ -12 dB, VH ~ -18 dB
            land_vv = np.random.normal(-12.0, 3.0, (image_size, image_size))
            land_vh = np.random.normal(-18.0, 3.0, (image_size, image_size))

            # Synthetic flood water polygon/channel (specular reflection: lower backscatter VV ~ -22 dB, VH ~ -28 dB)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            y, x = np.ogrid[:image_size, :image_size]
            
            # Water body 1: Meandering river / flood corridor
            center_x = image_size // 2 + int(np.sin(idx) * 30)
            river_mask = np.abs(x - center_x - 15 * np.sin(y / 25.0)) < 18
            # Water body 2: Inundated pond/field
            pond_mask = ((x - (image_size // 3)) ** 2 + (y - (image_size // 3)) ** 2) < (25 ** 2)

            flood_mask = river_mask | pond_mask
            mask[flood_mask] = 1

            # Water pixels have strong radar specular loss (dark in SAR)
            land_vv[flood_mask] = np.random.normal(-22.0, 2.0, size=flood_mask.sum())
            land_vh[flood_mask] = np.random.normal(-28.0, 2.0, size=flood_mask.sum())

            # Convert to float32 SAR array (2, H, W)
            sar_array = np.stack([land_vv, land_vh], axis=0).astype(np.float32)

            # Save SAR (2-band) and Label (1-band)
            # If rasterio is available, write as multi-band GeoTIFF, else PIL/numpy
            try:
                import rasterio
                from rasterio.transform import from_origin
                transform = from_origin(92.0, 26.0, 0.0001, 0.0001)

                with rasterio.open(
                    s1_dir / s1_filename,
                    "w",
                    driver="GTiff",
                    height=image_size,
                    width=image_size,
                    count=2,
                    dtype=sar_array.dtype,
                    crs="EPSG:4326",
                    transform=transform,
                ) as dst:
                    dst.write(sar_array)

                with rasterio.open(
                    label_dir / label_filename,
                    "w",
                    driver="GTiff",
                    height=image_size,
                    width=image_size,
                    count=1,
                    dtype="int16",
                    crs="EPSG:4326",
                    transform=transform,
                ) as dst:
                    dst.write(mask.astype(np.int16), 1)

            except Exception:
                # PIL fallback: save as PNG / raw
                if PIL_AVAILABLE:
                    # Save VV/VH as 2-channel or RGB PNG
                    vv_norm = ((land_vv - (-25.0)) / 25.0 * 255.0).clip(0, 255).astype(np.uint8)
                    vh_norm = ((land_vh - (-32.0)) / 27.0 * 255.0).clip(0, 255).astype(np.uint8)
                    sar_rgb = np.stack([vv_norm, vh_norm, vv_norm], axis=-1)
                    Image.fromarray(sar_rgb).save(s1_dir / f"{base_name}_S1Hand.png")
                    Image.fromarray(mask * 255).save(label_dir / f"{base_name}_LabelHand.png")
                    s1_filename = f"{base_name}_S1Hand.png"
                    label_filename = f"{base_name}_LabelHand.png"

            split_records[split_name].append((f"S1Hand/{s1_filename}", f"LabelHand/{label_filename}"))
            idx += 1

    # Write split CSV files
    for split_name, rows in split_records.items():
        csv_file = splits_dir / f"flood_{split_name}_data.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

    print("=" * 70)
    print("SYNTHETIC SEN1FLOODS11 DATASET GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  - Root Directory:  {out_path.resolve()}")
    print(f"  - Train Samples:   {num_train}")
    print(f"  - Val Samples:     {num_val}")
    print(f"  - Test Samples:    {num_test}")
    print(f"  - Format:          Sentinel-1 VV/VH (2-Band) + Ground Truth Binary Mask")
    print("=" * 70)


def print_sen1floods_instructions():
    """Print instructions and download sources for official Sen1Floods11 dataset."""
    print("""
================================================================================
OFFICIAL SEN1FLOODS11 DATASET DOWNLOAD & SETUP INSTRUCTIONS
================================================================================
Dataset Reference:
  Bonafilia et al., 'Sen1Floods11: A Georeferenced Dataset to Train and
  Benchmark Deep Learning Flood Algorithms for Sentinel-1', CVPRW 2020.
  Official GitHub: https://github.com/cloudtostreet/Sen1Floods11

How to configure in this project:
1. Download or clone Sen1Floods11 into fine_tune/data/sen1floods11
2. Directory structure expected:
   fine_tune/data/sen1floods11/
   ├── S1Hand/                     # Sentinel-1 SAR GeoTIFFs (*_S1Hand.tif)
   ├── LabelHand/                  # Ground truth flood masks (*_LabelHand.tif)
   └── splits/
       └── flood_handlabeled/
           ├── flood_train_data.csv
           ├── flood_valid_data.csv
           ├── flood_test_data.csv
           └── flood_bolivia_data.csv

3. On Kaggle:
   Set environment variable or config parameter:
   export DATASET_ROOT="/kaggle/input/sen1floods11"
   or update 'data.root' in fine_tune/configs/terramind.yaml
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation and synthetic generator.")
    parser.add_argument("--create-synthetic", action="store_true", help="Generate synthetic Sen1Floods11 dataset.")
    parser.add_argument("--output-dir", type=str, default="fine_tune/data/sample_sen1floods11", help="Output directory.")
    parser.add_argument("--train-samples", type=int, default=12, help="Number of synthetic train samples.")
    parser.add_argument("--val-samples", type=int, default=4, help="Number of synthetic validation samples.")
    parser.add_argument("--test-samples", type=int, default=4, help="Number of synthetic test samples.")
    parser.add_argument("--info", action="store_true", help="Print Sen1Floods11 download instructions.")
    args = parser.parse_args()

    if args.info:
        print_sen1floods_instructions()

    if args.create_synthetic or not args.info:
        create_synthetic_sen1floods_dataset(
            output_dir=args.output_dir,
            num_train=args.train_samples,
            num_val=args.val_samples,
            num_test=args.test_samples,
        )


if __name__ == "__main__":
    main()
