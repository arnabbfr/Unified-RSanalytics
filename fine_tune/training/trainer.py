"""Reusable PyTorch Trainer engine for foundation model fine-tuning."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fine_tune.decoders.segmentation_decoder import SegmentationDecoder
from fine_tune.decoders.unet_decoder import build_decoder
from fine_tune.evaluation.metrics import SegmentationMetricsTracker
from fine_tune.models.base_model import FoundationModelBase
from fine_tune.training.losses import build_loss_fn
from fine_tune.utils.checkpoint import CheckpointManager, save_experiment_summary
from fine_tune.utils.config import Config
from fine_tune.utils.device import empty_cache, get_device, get_device_info, log_device_info, warn_vram_risk
from fine_tune.utils.logging import CSVLogger, SummaryWriterWrapper, setup_logger

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class FoundationModelTrainer:
    """Comprehensive Trainer for fine-tuning remote sensing foundation models on flood segmentation."""

    def __init__(
        self,
        config: Config,
        model: FoundationModelBase,
        decoder: nn.Module | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 1. Device Setup & Memory Logging
        self.device = get_device("auto")
        self.dev_info = log_device_info(self.device)

        # 2. Config sections
        self.train_cfg = self.config.get("training", {})
        self.loss_cfg = self.config.get("loss", {})
        self.ckpt_cfg = self.config.get("checkpoint", {})
        self.res_cfg = self.config.get("results", {})
        self.log_cfg = self.config.get("logging", {})

        self.strategy = str(self.train_cfg.get("strategy", "frozen")).lower()
        self.epochs = int(self.train_cfg.get("epochs", 10))
        self.grad_accum_steps = max(1, int(self.train_cfg.get("gradient_accumulation_steps", 4)))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 1.0))
        self.mixed_precision = bool(self.train_cfg.get("mixed_precision", True)) and self.device.type == "cuda"
        self.patience = int(self.train_cfg.get("early_stopping_patience", 5))

        # 3. Instantiate Decoder if not provided
        if decoder is None:
            dec_name = self.config.get_nested("model.decoder", "segmentation_decoder")
            num_classes = int(self.config.get_nested("data.num_classes", 1))
            self.decoder = build_decoder(
                decoder_name=dec_name,
                in_channels=self.model.feature_dim,
                num_classes=num_classes,
            )
        else:
            self.decoder = decoder

        # 4. Apply Fine-Tuning Strategy
        self._apply_strategy()

        # Move modules to device
        self.model.to(self.device)
        self.decoder.to(self.device)

        # 5. Loss and Metrics
        self.loss_fn = build_loss_fn(self.loss_cfg)
        eval_thresh = float(self.config.get_nested("evaluation.threshold", 0.35))
        self.metrics_tracker = SegmentationMetricsTracker(
            threshold=eval_thresh,
            ignore_index=int(self.config.get_nested("data.ignore_index", -1)),
        )

        # 6. Optimizer & Scheduler
        self.optimizer, self.scheduler = self._configure_optimizers()

        # 7. Mixed Precision Scaler
        if self.mixed_precision and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        elif self.mixed_precision and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        # 8. Checkpoint & Logging Managers
        ckpt_dir = self.ckpt_cfg.get("dir", f"fine_tune/checkpoints/{self.config.model.name}")
        results_dir = self.res_cfg.get("dir", f"fine_tune/results/{self.config.model.name}")
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            save_best=bool(self.ckpt_cfg.get("save_best", True)),
            monitor=str(self.ckpt_cfg.get("monitor", "val_iou")),
            mode=str(self.ckpt_cfg.get("mode", "max")),
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            name=f"fine_tune.{self.config.model.name}",
            log_file=self.results_dir / "train.log",
        )
        self.csv_logger = CSVLogger(self.results_dir / "training_log.csv")
        self.tb_writer = SummaryWriterWrapper(
            log_dir=self.results_dir / "tensorboard",
            enabled=bool(self.log_cfg.get("tensorboard", True)),
        )

        # Memory warning check
        total_m, trainable_m = self.model.param_counts()
        warn_vram_risk(self.strategy, total_m, self.dev_info)

    def _apply_strategy(self) -> None:
        """Apply frozen, partial, or full fine-tuning strategy."""
        if self.strategy == "frozen":
            self.model.freeze_backbone()
            print(f"[Strategy] Mode: FROZEN. Backbone frozen, training segmentation decoder only.")
        elif self.strategy == "partial":
            n_blocks = int(self.config.get_nested("model.unfreeze_last_n_blocks", 2))
            self.model.unfreeze_last_blocks(num_blocks=n_blocks)
            print(f"[Strategy] Mode: PARTIAL. Unfreezing last {n_blocks} transformer blocks + decoder.")
        elif self.strategy == "full":
            self.model.unfreeze_all()
            print(f"[Strategy] Mode: FULL. Unfreezing entire model backbone and decoder.")
        else:
            raise ValueError(f"Unknown fine-tuning strategy: '{self.strategy}'. Choose 'frozen', 'partial', or 'full'.")

        # Decoder is always trainable
        for param in self.decoder.parameters():
            param.requires_grad = True

    def _configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Any]:
        """Configure AdamW optimizer with differential learning rates."""
        lr_dec = float(self.train_cfg.get("learning_rate", 1e-4))
        lr_backbone = float(self.train_cfg.get("backbone_learning_rate", 1e-5))
        weight_decay = float(self.train_cfg.get("weight_decay", 0.01))

        param_groups = []
        # Backbone parameter groups
        backbone_groups = self.model.parameter_groups(
            backbone_lr=lr_backbone,
            weight_decay=weight_decay,
        )
        param_groups.extend(backbone_groups)

        # Decoder parameter groups
        dec_params = [p for p in self.decoder.parameters() if p.requires_grad]
        if dec_params:
            param_groups.append({
                "params": dec_params,
                "lr": lr_dec,
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-7,
        )
        return optimizer, scheduler

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch with gradient accumulation and AMP."""
        self.model.train() if self.strategy != "frozen" else self.model.eval()
        self.decoder.train()
        self.metrics_tracker.reset()

        total_loss = 0.0
        self.optimizer.zero_grad()

        loader = self.train_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.epochs} [Train]") if TQDM_AVAILABLE else loader
        num_batches = len(loader) if loader is not None else 0

        if num_batches == 0:
            return 0.0, {}

        for step, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                features = self.model(images)
                target_size = (images.shape[-2], images.shape[-1])
                logits = self.decoder(features, target_size=target_size)
                loss = self.loss_fn(logits, masks)
                loss_scaled = loss / self.grad_accum_steps

            if self.scaler is not None and self.mixed_precision:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                if self.scaler is not None and self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.decoder.parameters()),
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.decoder.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item()
            self.metrics_tracker.update(logits.detach(), masks)

            if TQDM_AVAILABLE and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute()
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None, desc: str = "Val") -> Tuple[float, Dict[str, float]]:
        """Evaluate model on a DataLoader split."""
        if loader is None or len(loader) == 0:
            return 0.0, {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

        self.model.eval()
        self.decoder.eval()
        self.metrics_tracker.reset()

        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[{desc}]") if TQDM_AVAILABLE else loader
        num_batches = len(loader)

        for images, masks, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                features = self.model(images)
                target_size = (images.shape[-2], images.shape[-1])
                logits = self.decoder(features, target_size=target_size)
                loss = self.loss_fn(logits, masks)

            total_loss += loss.item()
            self.metrics_tracker.update(logits, masks)

        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute()
        return avg_loss, metrics

    def fit(self) -> Dict[str, Any]:
        """Execute full training and validation lifecycle."""
        print("\n" + "=" * 75)
        print(f"STARTING FOUNDATION MODEL FINE-TUNING: {self.config.model.name.upper()}")
        print(f"Strategy: {self.strategy.upper()} | Epochs: {self.epochs} | Mixed Precision: {self.mixed_precision}")
        print("=" * 75)

        start_time = time.time()
        best_metric_val = float("-inf")
        best_epoch = 0
        patience_counter = 0

        history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_iou": [],
            "val_iou": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": [],
        }

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.evaluate(self.val_loader, desc="Validation")
            self.scheduler.step()

            elapsed = time.time() - t0
            curr_lr = self.optimizer.param_groups[-1]["lr"]

            # Monitor metric for early stopping and best checkpointing
            monitor_key = self.ckpt_cfg.get("monitor", "val_iou").replace("val_", "")
            current_metric = val_metrics.get(monitor_key, val_metrics.get("iou", 0.0))
            is_best = self.ckpt_manager.is_better(current_metric)

            if is_best:
                best_metric_val = current_metric
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Save Checkpoints
            self.ckpt_manager.save(
                epoch=epoch,
                model=self.model,
                decoder=self.decoder,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics=val_metrics,
                config=self.config,
                is_best=is_best,
            )

            # Log to CSV
            log_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_iou": train_metrics.get("iou", 0.0),
                "val_iou": val_metrics.get("iou", 0.0),
                "val_dice": val_metrics.get("dice", 0.0),
                "val_precision": val_metrics.get("precision", 0.0),
                "val_recall": val_metrics.get("recall", 0.0),
                "learning_rate": curr_lr,
                "time_sec": round(elapsed, 2),
            }
            self.csv_logger.log(log_row)

            # Log to TensorBoard
            self.tb_writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.tb_writer.add_scalars("IoU", {"train": train_metrics.get("iou", 0.0), "val": val_metrics.get("iou", 0.0)}, epoch)
            self.tb_writer.add_scalar("Dice", val_metrics.get("dice", 0.0), epoch)
            self.tb_writer.add_scalar("LearningRate", curr_lr, epoch)

            # Console output
            star = " * [BEST]" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val IoU: {val_metrics.get('iou', 0.0):.4f} | "
                f"Val Dice: {val_metrics.get('dice', 0.0):.4f}{star}"
            )

            # Early stopping check
            if patience_counter >= self.patience:
                print(f"\n[Early Stopping] No improvement in '{self.ckpt_manager.monitor}' for {self.patience} epochs. Stopping.")
                break

            empty_cache()

        total_time = time.time() - start_time
        print("\n" + "=" * 75)
        print(f"TRAINING COMPLETE IN {total_time / 60:.2f} MINUTES")
        print(f"Best {self.ckpt_manager.monitor}: {best_metric_val:.4f} (Epoch {best_epoch})")
        print("=" * 75)

        # Final experiment summary JSON
        total_p, train_p = self.model.param_counts()
        summary = {
            "model_name": self.config.model.name,
            "pretrained": self.config.model.pretrained,
            "dataset": self.config.get_nested("data.dataset", "sen1floods11"),
            "modality": self.config.get_nested("data.modality", "sentinel1"),
            "channels": self.config.get_nested("data.channels", ["VV", "VH"]),
            "image_size": self.config.get_nested("data.image_size", 224),
            "batch_size": self.train_cfg.get("batch_size", 1),
            "gradient_accumulation_steps": self.grad_accum_steps,
            "strategy": self.strategy,
            "epochs": self.epochs,
            "learning_rate": self.train_cfg.get("learning_rate", 1e-4),
            "backbone_learning_rate": self.train_cfg.get("backbone_learning_rate", 1e-5),
            "total_parameters_million": total_p,
            "trainable_parameters_million": train_p,
            "device": self.dev_info.get("device_name", "CPU"),
            "vram_gb": self.dev_info.get("total_memory_gb", 0.0),
            "training_time_seconds": round(total_time, 2),
            "best_epoch": best_epoch,
            "best_metric": self.ckpt_manager.monitor,
            "best_metric_value": best_metric_val,
            "final_metrics": val_metrics,
        }
        save_experiment_summary(self.results_dir, summary)
        self.tb_writer.close()

        return summary
