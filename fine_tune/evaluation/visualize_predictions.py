"""Standardized 4-panel visual comparison generator for flood segmentation predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.datasets.sen1floods_dataset import Sen1FloodsDataset
from fine_tune.decoders.unet_decoder import build_decoder
from fine_tune.models import build_model
from fine_tune.utils.checkpoint import CheckpointManager
from fine_tune.utils.config import load_config
from fine_tune.utils.device import get_device

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_sar_false_color(sar_img: np.ndarray) -> np.ndarray:
    """Compose RGB visualization from 2-channel Sentinel-1 SAR (VV, VH, VV/VH ratio)."""
    # sar_img: shape (C, H, W) normalized in [0, 1]
    c, h, w = sar_img.shape
    vv = sar_img[0]
    vh = sar_img[1] if c > 1 else vv
    ratio = np.clip(vv / (vh + 1e-4), 0.0, 1.0)

    rgb = np.stack([vv, vh, ratio], axis=-1)  # (H, W, 3)
    return np.clip(rgb, 0.0, 1.0)


def create_confusion_overlay(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    ignore_index: int = -1,
) -> np.ndarray:
    """Render color-coded confusion overlay:
    - Green: True Positive (Hit)
    - Red: False Positive (False Alarm)
    - Blue: False Negative (Missed Flood)
    - Dark Slate: True Negative (Correct Non-Flood)
    - Gray: Ignored / No-Data
    """
    h, w = ground_truth.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    valid = (ground_truth != ignore_index) & (ground_truth != 255)
    gt_pos = (ground_truth == 1) & valid
    gt_neg = (ground_truth == 0) & valid
    pred_pos = (prediction == 1) & valid
    pred_neg = (prediction == 0) & valid

    # True Positives -> Vibrant Green [0.1, 0.9, 0.2]
    tp = gt_pos & pred_pos
    overlay[tp] = [0.1, 0.9, 0.2]

    # False Positives -> Bright Red [0.95, 0.15, 0.15]
    fp = gt_neg & pred_pos
    overlay[fp] = [0.95, 0.15, 0.15]

    # False Negatives -> Bright Blue/Cyan [0.15, 0.45, 0.95]
    fn = gt_pos & pred_neg
    overlay[fn] = [0.15, 0.45, 0.95]

    # True Negatives -> Dark Slate / Neutral [0.1, 0.12, 0.15]
    tn = gt_neg & pred_neg
    overlay[tn] = [0.1, 0.12, 0.15]

    # Ignored pixels -> Medium Gray [0.35, 0.35, 0.35]
    overlay[~valid] = [0.35, 0.35, 0.35]

    return overlay


def generate_visualizations(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "valid",
    num_samples: int = 5,
    output_dir: str | Path | None = None,
) -> None:
    """Generate and save 4-panel visual comparison cards for a set of evaluation samples."""
    if not MATPLOTLIB_AVAILABLE:
        print("[Visualization Warning] 'matplotlib' is not available. Skipping image rendering.")
        return

    config = load_config(config_path)
    device = get_device("auto")

    # Resolve Checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config.get_nested("checkpoint.dir", f"fine_tune/checkpoints/{config.model.name}")) / "best.pt"
    checkpoint_path = Path(checkpoint_path)

    # Resolve Output Directory
    if output_dir is None:
        output_dir = Path(config.get_nested("results.dir", f"fine_tune/results/{config.model.name}")) / "predictions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    data_cfg = config.data
    dataset = Sen1FloodsDataset(
        data_root=data_cfg.get("root", "fine_tune/data/sample_sen1floods11"),
        split=split,
        subset=data_cfg.get("subset", "all"),
        image_size=int(data_cfg.get("image_size", 224)),
        augment=False,
        ignore_index=int(data_cfg.get("ignore_index", -1)),
    )

    if len(dataset) == 0:
        print(f"[Visualization Warning] No samples found in split '{split}'.")
        return

    # Model and Decoder
    model = build_model(config.model, config.data)
    decoder_name = config.get_nested("model.decoder", "segmentation_decoder")
    num_classes = int(config.get_nested("data.num_classes", 1))
    decoder = build_decoder(
        decoder_name=decoder_name,
        in_channels=model.feature_dim,
        num_classes=num_classes,
    )

    if checkpoint_path.is_file():
        ckpt_mgr = CheckpointManager(checkpoint_dir=checkpoint_path.parent)
        ckpt_mgr.load(checkpoint_path, model=model, decoder=decoder, map_location=device)
        print(f"[Visualization] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[Visualization Notice] Checkpoint {checkpoint_path} not found. Using initialized model weights.")

    model.to(device).eval()
    decoder.to(device).eval()

    samples_to_render = min(num_samples, len(dataset))
    print(f"[Visualization] Generating {samples_to_render} comparison cards into: {output_dir}")

    for idx in range(samples_to_render):
        img_t, msk_t, meta = dataset[idx]
        sample_id = meta.get("sample_id", f"sample_{idx:03d}")

        with torch.no_grad():
            inp = img_t.unsqueeze(0).to(device)
            features = model(inp)
            logits = decoder(features, target_size=(img_t.shape[1], img_t.shape[2]))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob >= 0.5).astype(np.int64)

        img_np = img_t.numpy()
        msk_np = msk_t.numpy()

        sar_rgb = create_sar_false_color(img_np)
        overlay = create_confusion_overlay(msk_np, pred, ignore_index=int(data_cfg.get("ignore_index", -1)))

        # Plot 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#1a1c23")
        fig.suptitle(f"Model: {config.model.name.upper()} | Sample: {sample_id}", fontsize=14, color="#ffffff", y=0.98)

        # 1. SAR False Color
        axes[0].imshow(sar_rgb)
        axes[0].set_title("Input (SAR VV/VH/Ratio)", color="#e0e0e0", fontsize=12)
        axes[0].axis("off")

        # 2. Ground Truth Mask
        axes[1].imshow(msk_np, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth (Flood=1)", color="#e0e0e0", fontsize=12)
        axes[1].axis("off")

        # 3. Model Prediction
        axes[2].imshow(pred, cmap="Blues", vmin=0, vmax=1)
        axes[2].set_title("Predicted Mask (Sigmoid >= 0.5)", color="#e0e0e0", fontsize=12)
        axes[2].axis("off")

        # 4. Confusion Overlay
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (Green=TP, Red=FP, Blue=FN)", color="#e0e0e0", fontsize=12)
        axes[3].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"{sample_id}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  - Saved: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate 4-panel visual comparisons for predictions.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file.")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split to sample.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for images.")
    args = parser.parse_args()

    generate_visualizations(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
