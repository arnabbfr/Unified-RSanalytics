"""Deterministic seeding across Python, NumPy, PyTorch, and CUDA."""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = False) -> int:
    """Seed all random number generators for reproducible experiments.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, enforces deterministic cuDNN algorithms
            (may slightly reduce training speed).

    Returns:
        The integer seed used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    return seed
