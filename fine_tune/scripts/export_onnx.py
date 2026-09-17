"""Export fine-tuned PyTorch Geospatial Foundation Model checkpoints to ONNX format.

Enables offline, low-latency, air-gapped inference in C# / .NET Avalonia Desktop App via ONNX Runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.decoders.unet_decoder import build_decoder
from fine_tune.models import build_model
from fine_tune.utils.checkpoint import CheckpointManager
from fine_tune.utils.config import load_config


class EndToEndFloodModel(nn.Module):
    """Encapsulates foundation backbone + U-Net decoder into a single end-to-end forward module."""

    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.decoder(features, target_size=(x.shape[-2], x.shape[-1]))
        probabilities = torch.sigmoid(logits)
        return probabilities


def export_to_onnx(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_onnx_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export a trained foundation model checkpoint to ONNX."""
    config = load_config(config_path)
    model = build_model(config.model, config.data)
    decoder_name = config.get_nested("model.decoder", "unet_decoder")
    num_classes = int(config.get_nested("data.num_classes", 1))
    decoder = build_decoder(
        decoder_name=decoder_name,
        in_channels=model.feature_dim,
        num_classes=num_classes,
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_path.parent)
    ckpt_mgr.load(ckpt_path, model=model, decoder=decoder, map_location=torch.device("cpu"))

    full_model = EndToEndFloodModel(model, decoder)
    full_model.eval()

    in_channels = int(config.get_nested("data.in_channels", getattr(model, "in_channels", 8)))
    dummy_input = torch.randn(1, in_channels, 224, 224, dtype=torch.float32)

    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX Export] Exporting {config.model.name} to {out_path}...")
    torch.onnx.export(
        full_model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["satellite_input"],
        output_names=["flood_probability"],
        dynamic_axes={
            "satellite_input": {0: "batch_size", 2: "height", 3: "width"},
            "flood_probability": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    print(f"  [Success] Saved ONNX model to: {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model checkpoint to ONNX.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output", type=str, default="fine_tune/checkpoints/model.onnx", help="Output .onnx path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    args = parser.parse_args()

    export_to_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
