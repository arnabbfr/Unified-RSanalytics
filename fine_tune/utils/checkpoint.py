"""Checkpoint management and experiment metadata tracking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from fine_tune.utils.config import save_config, Config


class CheckpointManager:
    """Manages saving and loading model checkpoints and experiment states."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_best: bool = True,
        monitor: str = "val_iou",
        mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")

    def is_better(self, current: float) -> bool:
        """Evaluate if the current metric beats the best recorded metric and update it."""
        if self.mode == "max":
            if current > self.best_metric:
                self.best_metric = current
                return True
            return False
        else:
            if current < self.best_metric:
                self.best_metric = current
                return True
            return False

    def save(
        self,
        epoch: int,
        model: nn.Module,
        decoder: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        metrics: Dict[str, float] | None = None,
        config: Config | Dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        """Save training state checkpoint."""
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics or {},
        }

        if decoder is not None:
            state["decoder_state_dict"] = decoder.state_dict()
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            state["scaler_state_dict"] = scaler.state_dict()
        if config is not None:
            state["config"] = config.to_dict() if isinstance(config, Config) else config

        # Always save last.pt
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(state, last_path)

        # Save best.pt if requested and improved
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)

        # Save config.yaml alongside
        if config is not None:
            save_config(config, self.checkpoint_dir / "config.yaml")

        return last_path

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        decoder: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Any]:
        """Load state from a checkpoint file into model, decoder, optimizer, etc."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])

        if decoder is not None and "decoder_state_dict" in checkpoint:
            decoder.load_state_dict(checkpoint["decoder_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]

        return checkpoint


def save_experiment_summary(
    results_dir: str | Path,
    summary_data: Dict[str, Any],
) -> Path:
    """Save comprehensive research-grade experiment summary JSON."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    return out_path
