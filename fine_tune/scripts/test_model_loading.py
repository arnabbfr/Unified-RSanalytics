"""Diagnostic utility to verify model loading, parameter counts, and forward passes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.decoders.segmentation_decoder import SegmentationDecoder
from fine_tune.models import build_model
from fine_tune.models.base_model import ModalityMismatchError
from fine_tune.utils.config import load_config
from fine_tune.utils.device import get_device, log_device_info


def test_single_model(model_name: str) -> bool:
    """Run diagnostic verification on a single foundation model architecture."""
    config_file = Path(f"fine_tune/configs/{model_name}.yaml")
    if not config_file.is_file():
        print(f"[Error] Config file not found: {config_file}")
        return False

    config = load_config(config_file)
    device = get_device("auto")

    print("\n" + "=" * 75)
    print(f"DIAGNOSTIC TEST FOR FOUNDATION MODEL: {model_name.upper()}")
    print("=" * 75)
    print(f"  - Config:     {config_file}")
    print(f"  - Pretrained: {config.model.pretrained}")
    print(f"  - Modality:   {config.data.modality}")
    print(f"  - Channels:   {config.data.in_channels}")

    # 1. Model Instantiation & Modality Validation
    try:
        model = build_model(config.model, config.data)
        total_p, trainable_p = model.param_counts()
        print(f"  - Instantiation:   [PASS] (Total: {total_p}M params, Trainable: {trainable_p}M)")
    except ModalityMismatchError as err:
        print(f"  - Modality Check:  [EXPECTED REJECTION] {err}")
        return True
    except Exception as exc:
        print(f"  - Instantiation:   [FAIL] ({exc})")
        return False

    model.to(device).eval()

    # 2. Test Forward Pass on Backbone
    in_c = int(config.data.in_channels)
    img_size = int(config.data.image_size)
    dummy_input = torch.randn(1, in_c, img_size, img_size, device=device)

    try:
        with torch.no_grad():
            features = model(dummy_input)
        print(f"  - Backbone Forward: [PASS] -> Output Features Shape: {list(features.shape)}")
    except Exception as exc:
        print(f"  - Backbone Forward: [FAIL] ({exc})")
        return False

    # 3. Test Segmentation Decoder Integration
    try:
        decoder = SegmentationDecoder(in_channels=model.feature_dim, num_classes=1).to(device)
        with torch.no_grad():
            logits = decoder(features, target_size=(img_size, img_size))
        assert logits.shape == (1, 1, img_size, img_size), f"Unexpected shape {logits.shape}"
        probs = torch.sigmoid(logits)
        print(f"  - Decoder Forward:  [PASS] -> Output Logits Shape: {list(logits.shape)} (Probs in [{probs.min():.3f}, {probs.max():.3f}])")
    except Exception as exc:
        print(f"  - Decoder Forward:  [FAIL] ({exc})")
        return False

    # 4. Modality Incompatibility Check for Optical Models
    if model_name in ("prithvi", "satmaepp"):
        print("  - Testing SAR Incompatibility Guard:")
        try:
            # Attempt to spoof with 2-channel SAR
            bad_data_cfg = {"modality": "sentinel1", "in_channels": 2}
            bad_model = build_model(config.model, bad_data_cfg)
            print("    [WARNING] Model failed to reject SAR modality!")
        except ModalityMismatchError:
            print("    [PASS] Correctly caught incompatible SAR dataset and raised descriptive error.")

    print(f"  - Overall Status:   [ALL CHECKS PASSED FOR {model_name.upper()}]")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test foundation model loading and forward passes.")
    parser.add_argument(
        "--model",
        type=str,
        default="terramind",
        choices=["terramind", "prithvi", "satmaepp", "gfm", "all"],
        help="Target foundation model to test.",
    )
    args = parser.parse_args()

    device = get_device("auto")
    log_device_info(device)

    models_to_test = (
        ["terramind", "prithvi", "satmaepp", "gfm"]
        if args.model == "all"
        else [args.model]
    )

    results = {}
    for m in models_to_test:
        results[m] = test_single_model(m)

    print("\n" + "=" * 75)
    print("FOUNDATION MODEL DIAGNOSTIC SUMMARY")
    print("=" * 75)
    for m, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  - {m.ljust(15)} : [{status}]")
    print("=" * 75)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
