"""CLI Training entrypoint for GFM Composition Pretraining Model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.datasets.datamodule import build_dataloaders
from fine_tune.models import build_model
from fine_tune.models.base_model import ModalityMismatchError
from fine_tune.training.trainer import FoundationModelTrainer
from fine_tune.utils.config import load_config
from fine_tune.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GFM Composition Model.")
    parser.add_argument("--config", type=str, default="fine_tune/configs/gfm.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--strategy", type=str, choices=["frozen", "partial", "full"], default=None, help="Fine-tuning strategy.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    args = parser.parse_args()

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

    config = load_config(args.config, overrides=overrides)
    seed_everything(int(config.get_nested("training.seed", 42)))

    try:
        model = build_model(config.model, config.data)
    except ModalityMismatchError as err:
        print(f"\n[Validation Error] {err}")
        sys.exit(1)

    train_loader, val_loader, test_loader = build_dataloaders(config)

    trainer = FoundationModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    summary = trainer.fit()
    print(f"\n[Finished] GFM fine-tuning complete. Results at: {trainer.results_dir / 'experiment.json'}")


if __name__ == "__main__":
    main()
