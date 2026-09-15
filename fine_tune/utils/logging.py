"""Unified logging system supporting Console, CSV metrics, and TensorBoard."""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


def setup_logger(
    name: str = "fine_tune",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a named logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Prevent duplicate handlers if re-initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class CSVLogger:
    """Logs scalar metrics per epoch to a CSV file."""

    def __init__(self, file_path: str | Path, fieldnames: List[str] | None = None, append: bool = False):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames or []
        if not append and self.file_path.exists():
            try:
                self.file_path.unlink()
            except Exception:
                pass
        self._is_header_written = self.file_path.exists() and self.file_path.stat().st_size > 0

    def log(self, metrics: Dict[str, Any]) -> None:
        """Append a row of metrics to the CSV."""
        if not self.fieldnames:
            self.fieldnames = list(metrics.keys())

        # Update fieldnames if new keys appear
        for k in metrics.keys():
            if k not in self.fieldnames:
                self.fieldnames.append(k)

        file_exists = self.file_path.exists() and self.file_path.stat().st_size > 0
        with open(self.file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists or not self._is_header_written:
                writer.writeheader()
                self._is_header_written = True
            writer.writerow({k: metrics.get(k, "") for k in self.fieldnames})


class SummaryWriterWrapper:
    """Wrapper around PyTorch TensorBoard SummaryWriter with graceful fallback."""

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self._writer = None

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                # Tensorboard not installed
                self._writer = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
