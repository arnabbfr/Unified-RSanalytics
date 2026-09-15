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


class FloodPredictor:
    """High-level Python inference API for fine-tuned geospatial foundation models."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        model_name: str = "terramind",
        threshold: float = 0.5,
        device: str = "auto",
    ):
        self.device = get_device(device)
        self.threshold = threshold
        ckpt_path = Path(checkpoint_path)

        if config_path is None:
            candidate = ckpt_path.parent / "config.yaml"
            if candidate.exists():
                self.config = load_config(candidate)
            else:
                self.config = load_config(PROJECT_ROOT / f"fine_tune/configs/{model_name}.yaml")
        else:
            self.config = load_config(config_path)

        self.model = build_model(self.config.model, self.config.data)
        decoder_name = self.config.get_nested("model.decoder", "segmentation_decoder")
        num_classes = int(self.config.get_nested("data.num_classes", 1))
        self.decoder = build_decoder(
            decoder_name=decoder_name,
            in_channels=self.model.feature_dim,
            num_classes=num_classes,
        )

        ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_path.parent)
        ckpt_mgr.load(ckpt_path, model=self.model, decoder=self.decoder, map_location=self.device)

        self.model.to(self.device).eval()
        self.decoder.to(self.device).eval()

    def predict(
        self,
        image_path: str | Path | np.ndarray,
        save_overlay_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run flood prediction on a single raster path or numpy array."""
        if isinstance(image_path, (str, Path)):
            prob, mask = predict_raster(image_path, self.model, self.decoder, self.device, threshold=self.threshold)
            raw = read_geotiff(image_path)
        else:
            raw = image_path
            normed = normalize_sar(raw)
            in_c = getattr(self.model, "in_channels", 2)
            if normed.shape[0] > in_c:
                normed = normed[:in_c]
            elif normed.shape[0] < in_c:
                normed = np.pad(normed, ((0, in_c - normed.shape[0]), (0, 0), (0, 0)), mode="edge")
            inp = torch.from_numpy(normed).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                features = self.model(inp)
                logits = self.decoder(features, target_size=(raw.shape[-2], raw.shape[-1]))
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            mask = (prob >= self.threshold).astype(np.uint8)

        flood_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        flood_fraction = float(flood_pixels / max(total_pixels, 1))

        if save_overlay_path and PIL_AVAILABLE:
            save_path = Path(save_overlay_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vv = raw[0] if raw.ndim == 3 else raw
            if np.all(np.isnan(vv)):
                base = np.zeros_like(vv, dtype=np.uint8)
            else:
                if np.nanmin(vv) >= 0 and np.nanmax(vv) > 5:
                    vv_db = 10.0 * np.log10(np.clip(vv, 1e-5, None))
                else:
                    vv_db = vv
                vv_norm = np.clip((vv_db - (-25.0)) / (25.0 + 1e-6), 0.0, 1.0)
                base = (np.nan_to_num(vv_norm, nan=0.0) * 255).astype(np.uint8)
            rgb = np.stack([base, base, base], axis=-1)
            # Tint flood pixels bright cyan/blue
            rgb[mask == 1, 0] = (rgb[mask == 1, 0] * 0.2).astype(np.uint8)
            rgb[mask == 1, 1] = np.clip(rgb[mask == 1, 1] * 0.7 + 100, 0, 255).astype(np.uint8)
            rgb[mask == 1, 2] = 255
            Image.fromarray(rgb).save(save_path)

        return {
            "probability": prob,
            "mask": mask,
            "flood_pixels": flood_pixels,
            "total_pixels": total_pixels,
            "flood_fraction": flood_fraction,
        }


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
