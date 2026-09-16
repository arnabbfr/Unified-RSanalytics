"""Unified experiment runner coordinating data loading, fine-tuning, evaluation, and visualization."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.datasets.datamodule import build_dataloaders
from fine_tune.datasets.sen1floods_dataset import resolve_dataset_path
from fine_tune.evaluation.evaluate import evaluate_checkpoint
from fine_tune.evaluation.visualize_predictions import generate_visualizations
from fine_tune.models import build_model
from fine_tune.models.base_model import ModalityMismatchError
from fine_tune.scripts.prepare_dataset import create_synthetic_sen1floods_dataset
from fine_tune.training.trainer import FoundationModelTrainer
from fine_tune.utils.config import load_config
from fine_tune.utils.seed import seed_everything


def run_experiment(
    model_name: str = "terramind",
    config_path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    strategy: str | None = None,
    data_root: str | None = None,
    eval_only: bool = False,
    visualize: bool = True,
    seed: int = 42,
) -> None:
    """Execute end-to-end foundation model fine-tuning experiment."""
    # 1. Resolve Config Path
    if config_path is None:
        config_path = f"fine_tune/configs/{model_name}.yaml"
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    overrides = {}
    if epochs is not None:
        overrides["training.epochs"] = epochs
    if batch_size is not None:
        overrides["training.batch_size"] = batch_size
    if lr is not None:
        overrides["training.learning_rate"] = lr
    if strategy is not None:
        overrides["training.strategy"] = strategy
    if data_root is not None:
        overrides["data.root"] = data_root

    config = load_config(cfg_file, overrides=overrides)
    seed_everything(seed)

    # 2. Ensure dataset exists or generate synthetic fallback
    dataset_root = resolve_dataset_path(config.data.root)
    config.data.root = str(dataset_root)

    is_cloud_input = str(dataset_root).startswith("/kaggle/input")
    if is_cloud_input:
        if not dataset_root.exists():
            print(f"[Dataset Notice] Target '{dataset_root}' not found. Please verify dataset is attached under /kaggle/input.")
    else:
        if not dataset_root.exists() or len(list(dataset_root.glob("*"))) == 0:
            print(f"[Dataset] Local path '{dataset_root}' is empty. Generating synthetic Sen1Floods11 sample dataset...")
            create_synthetic_sen1floods_dataset(output_dir=dataset_root)

    # 3. Model & Modality Validation
    try:
        model = build_model(config.model, config.data)
    except ModalityMismatchError as err:
        print(f"\n[Experiment Terminated due to Modality Incompatibility]\n{err}")
        return

    # 4. Build DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # 5. Training / Evaluation
    ckpt_path = Path(config.get_nested("checkpoint.dir", f"fine_tune/checkpoints/{model_name}")) / "best.pt"

    if not eval_only:
        trainer = FoundationModelTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        trainer.fit()

    # 6. Evaluation
    if ckpt_path.is_file():
        print("\n" + "=" * 70)
        print("RUNNING FINAL BENCHMARK EVALUATION")
        print("=" * 70)
        eval_split = "test" if test_loader is not None and len(test_loader) > 0 else "valid"
        evaluate_checkpoint(config_path=config, checkpoint_path=ckpt_path, split=eval_split)

        # 7. Visualization
        if visualize:
            print("\n" + "=" * 70)
            print("GENERATING PREDICTION VISUALIZATION CARDS")
            print("=" * 70)
            generate_visualizations(
                config_path=config,
                checkpoint_path=ckpt_path,
                split=eval_split,
                num_samples=4,
            )

    print("\n" + "=" * 75)
    print(f"EXPERIMENT COMPLETED FOR: {model_name.upper()}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Unified Geospatial Foundation Model Experiment Runner.")
    parser.add_argument(
        "--model",
        type=str,
        default="terramind",
        choices=["terramind", "prithvi", "satmaepp", "gfm"],
        help="Target model architecture to run.",
    )
    parser.add_argument("--config", type=str, default=None, help="Custom configuration file path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--strategy", type=str, choices=["frozen", "partial", "full"], default=None, help="Fine-tuning strategy.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument("--no-vis", action="store_true", help="Skip prediction visualization rendering.")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
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
