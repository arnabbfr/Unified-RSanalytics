"""Universal CLI training entrypoint for foundation model fine-tuning."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.scripts.run_experiment import run_experiment
from fine_tune.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train geospatial foundation models for flood mapping.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--strategy", type=str, choices=["frozen", "partial", "full"], default=None, help="Override strategy.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only.")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization generation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg.model.name

    run_experiment(
        model_name=model_name,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        strategy=args.strategy,
        data_root=args.data_root,
        eval_only=args.eval_only,
        visualize=not args.no_vis,
    )


if __name__ == "__main__":
    main()
