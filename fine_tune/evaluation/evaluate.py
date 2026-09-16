"""Independent evaluation script to compute research metrics from a saved model checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.datasets.sen1floods_dataset import Sen1FloodsDataset
from fine_tune.decoders.unet_decoder import build_decoder
from fine_tune.evaluation.metrics import SegmentationMetricsTracker
from fine_tune.models import build_model
from fine_tune.utils.checkpoint import CheckpointManager
from fine_tune.utils.config import load_config
from fine_tune.utils.device import get_device, log_device_info


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "valid",
    output_json: str | Path | None = None,
) -> dict:
    """Evaluate a trained model checkpoint against validation or test datasets."""
    config = load_config(config_path)
    device = get_device("auto")
    log_device_info(device)

    # Resolve Checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config.get_nested("checkpoint.dir", f"fine_tune/checkpoints/{config.model.name}")) / "best.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Build Dataset and Loader
    data_cfg = config.data
    in_channels = int(data_cfg.get("in_channels", 3))
    dataset = Sen1FloodsDataset(
        data_root=data_cfg.get("root", "fine_tune/data/sample_sen1floods11"),
        split=split,
        subset=data_cfg.get("subset", "all"),
        image_size=int(data_cfg.get("image_size", 224)),
        in_channels=in_channels,
        augment=False,
        ignore_index=int(data_cfg.get("ignore_index", -1)),
    )

    if len(dataset) == 0:
        print(f"[Evaluation Warning] Split '{split}' is empty in dataset.")
        return {}

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Build Model and Decoder
    model = build_model(config.model, config.data)
    decoder_name = config.get_nested("model.decoder", "segmentation_decoder")
    num_classes = int(config.get_nested("data.num_classes", 1))
    decoder = build_decoder(
        decoder_name=decoder_name,
        in_channels=model.feature_dim,
        num_classes=num_classes,
    )

    # Load weights
    ckpt_mgr = CheckpointManager(checkpoint_dir=checkpoint_path.parent)
    ckpt_mgr.load(checkpoint_path, model=model, decoder=decoder, map_location=device)

    model.to(device).eval()
    decoder.to(device).eval()

    eval_thresh = float(config.get_nested("evaluation.threshold", 0.35))
    tracker = SegmentationMetricsTracker(
        threshold=eval_thresh,
        ignore_index=int(data_cfg.get("ignore_index", -1)),
    )

    # Multi-threshold sweep trackers for full diagnostic visibility
    sweep_trackers = {
        0.25: SegmentationMetricsTracker(threshold=0.25, ignore_index=int(data_cfg.get("ignore_index", -1))),
        0.35: tracker,
        0.45: SegmentationMetricsTracker(threshold=0.45, ignore_index=int(data_cfg.get("ignore_index", -1))),
        0.50: SegmentationMetricsTracker(threshold=0.50, ignore_index=int(data_cfg.get("ignore_index", -1))),
    }

    print(f"\nEvaluating {config.model.name} on split: '{split}' ({len(dataset)} samples)...")
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            features = model(images)
            target_size = (images.shape[-2], images.shape[-1])
            logits = decoder(features, target_size=target_size)
            for t in sweep_trackers.values():
                t.update(logits, masks)

    metrics = tracker.compute()

    # Formatted Report
    print("=" * 65)
    print(f"EVALUATION RESULTS -- {config.model.name.upper()} ({split.upper()} SET | THRESHOLD: {eval_thresh})")
    print("=" * 65)
    print(f"  - Flood IoU (Jaccard):    {metrics['iou']:.4f}")
    print(f"  - Flood Dice / F1-Score:  {metrics['dice']:.4f}")
    print(f"  - Flood Precision:        {metrics['precision']:.4f}")
    print(f"  - Flood Recall (TPR):     {metrics['recall']:.4f}")
    print(f"  - Specificity (TNR):      {metrics['specificity']:.4f}")
    print(f"  - Overall Pixel Accuracy: {metrics['accuracy']:.4f}")
    print("-" * 65)
    print(f"  - Confusion Matrix: TP={metrics['tp']} | FP={metrics['fp']} | FN={metrics['fn']} | TN={metrics['tn']}")
    print("=" * 65)

    # Display Threshold Calibration Sweep
    print("\n  [Threshold Calibration Curve]")
    for th, trk in sweep_trackers.items():
        res = trk.compute()
        marker = " <-- (Selected)" if abs(th - eval_thresh) < 1e-3 else ""
        print(f"    - Thresh {th:.2f}: Dice={res['dice']*100:.2f}% | Precision={res['precision']*100:.2f}% | Recall={res['recall']*100:.2f}% | IoU={res['iou']*100:.2f}%{marker}")
    print("=" * 65)

    # Save to JSON
    if output_json is None:
        res_dir = Path(config.get_nested("results.dir", f"fine_tune/results/{config.model.name}"))
        res_dir.mkdir(parents=True, exist_ok=True)
        output_json = res_dir / f"metrics_{split}.json"
    else:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "model_name": config.model.name,
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_samples": len(dataset),
        "metrics": metrics,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"[Saved] Metrics written to: {output_json}")
    return report_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned foundation model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (defaults to best.pt).")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test", "train"], help="Dataset split to evaluate.")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file.")
    args = parser.parse_args()

    evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
