"""CLI Training entrypoint for TerraMind-1.0-base Sentinel-1 SAR Flood Segmentation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.datasets.datamodule import build_dataloaders
from fine_tune.models import build_model
from fine_tune.training.trainer import FoundationModelTrainer
from fine_tune.utils.config import load_config
from fine_tune.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TerraMind-1.0-base on Sentinel-1 SAR Flood Segmentation.")
    parser.add_argument("--config", type=str, default="fine_tune/configs/terramind.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--strategy", type=str, choices=["frozen", "partial", "full"], default=None, help="Fine-tuning strategy.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    # Build overrides
    overrides = {}
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["training.learning_rate"] = args.lr
    if args.strategy is not None:
        overrides["training.strategy"] = args.strategy
    if args.data_root is not None:
        overrides["data.root"] = args.data_root
    if args.seed is not None:
        overrides["training.seed"] = args.seed

    # 1. Load Configuration
    config = load_config(args.config, overrides=overrides)

    # 2. Seed Everything
    seed = int(config.get_nested("training.seed", 42))
    seed_everything(seed)

    # 3. Build DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(f"[Dataset] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 4. Instantiate Model (Modality validated)
    model = build_model(config.model, config.data)

    # 5. Build Trainer and Run
    trainer = FoundationModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    summary = trainer.fit()
    print(f"\n[Finished] TerraMind fine-tuning complete. Experiment summary saved to: {trainer.results_dir / 'experiment.json'}")


if __name__ == "__main__":
    main()
