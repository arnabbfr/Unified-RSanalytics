"""Inference utility for applying fine-tuned foundation models to new satellite rasters."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.datasets.sen1floods_dataset import normalize_sar, read_geotiff
from fine_tune.decoders.unet_decoder import build_decoder
from fine_tune.models import build_model
from fine_tune.utils.checkpoint import CheckpointManager
from fine_tune.utils.config import load_config
from fine_tune.utils.device import get_device

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def predict_raster(
    raster_path: str | Path,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform binary flood segmentation on a single satellite raster image.

    Returns:
        Tuple of (probability_map [0..1], binary_mask [0 or 1]).
    """
    raw = read_geotiff(raster_path)
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]

    orig_h, orig_w = raw.shape[-2], raw.shape[-1]
    normed = normalize_sar(raw)

    in_c = getattr(model, "in_channels", 2)
    if normed.shape[0] > in_c:
        normed = normed[:in_c]
    elif normed.shape[0] < in_c:
        normed = np.pad(normed, ((0, in_c - normed.shape[0]), (0, 0), (0, 0)), mode="edge")

    inp = torch.from_numpy(normed).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = model(inp)
        logits = decoder(features, target_size=(orig_h, orig_w))
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob >= threshold).astype(np.uint8)
    return prob, mask


def predict_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """Run batch flood segmentation on all raster files in a directory."""
    config = load_config(config_path)
    device = get_device("auto")

    model = build_model(config.model, config.data)
    decoder_name = config.get_nested("model.decoder", "segmentation_decoder")
    num_classes = int(config.get_nested("data.num_classes", 1))
    decoder = build_decoder(
        decoder_name=decoder_name,
        in_channels=model.feature_dim,
        num_classes=num_classes,
    )

    ckpt_path = Path(checkpoint_path)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_path.parent)
    ckpt_mgr.load(ckpt_path, model=model, decoder=decoder, map_location=device)

    model.to(device).eval()
    decoder.to(device).eval()

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rasters = sorted(list(in_dir.glob("*.tif*")) + list(in_dir.glob("*.png")))
    print(f"[Inference] Found {len(rasters)} rasters in {in_dir}. Running prediction...")

    for r_path in rasters:
        prob, mask = predict_raster(r_path, model, decoder, device, threshold=threshold)

        # Save output PNG mask
        out_mask_path = out_dir / f"{r_path.stem}_flood_mask.png"
        out_prob_path = out_dir / f"{r_path.stem}_flood_prob.png"

        if PIL_AVAILABLE:
            Image.fromarray(mask * 255).save(out_mask_path)
            Image.fromarray((prob * 255).astype(np.uint8)).save(out_prob_path)
        else:
            np.save(out_dir / f"{r_path.stem}_flood_mask.npy", mask)
            np.save(out_dir / f"{r_path.stem}_flood_prob.npy", prob)

        print(f"  - Processed: {r_path.name} -> Flood Ratio: {mask.mean() * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run flood inference on satellite rasters.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF file or directory.")
    parser.add_argument("--output-dir", type=str, default="fine_tune/results/predictions", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        predict_directory(
            input_dir=in_path,
            output_dir=args.output_dir,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
        )
    elif in_path.is_file():
        config = load_config(args.config)
        device = get_device("auto")
        model = build_model(config.model, config.data)
        decoder = build_decoder(
            decoder_name=config.get_nested("model.decoder", "segmentation_decoder"),
            in_channels=model.feature_dim,
            num_classes=int(config.get_nested("data.num_classes", 1)),
        )
        ckpt_mgr = CheckpointManager(checkpoint_dir=Path(args.checkpoint).parent)
        ckpt_mgr.load(args.checkpoint, model=model, decoder=decoder, map_location=device)
        model.to(device).eval()
        decoder.to(device).eval()

        prob, mask = predict_raster(in_path, model, decoder, device, threshold=args.threshold)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if PIL_AVAILABLE:
            Image.fromarray(mask * 255).save(out_dir / f"{in_path.stem}_flood_mask.png")
            Image.fromarray((prob * 255).astype(np.uint8)).save(out_dir / f"{in_path.stem}_flood_prob.png")
        print(f"[Success] Prediction saved to {out_dir}. Flood pixels: {mask.sum()} / {mask.size} ({mask.mean() * 100:.2f}%)")
    else:
        print(f"[Error] Input path not found: {in_path}")


if __name__ == "__main__":
    main()
