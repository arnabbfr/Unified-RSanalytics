"""Hardware device detection and memory management utilities."""
from __future__ import annotations

import gc
import logging
from typing import Any, Dict

import torch

logger = logging.getLogger("fine_tune.device")


def get_device(preference: str = "auto") -> torch.device:
    """Select the optimal compute device based on preference and hardware availability.

    Args:
        preference: 'auto', 'cuda', 'cuda:0', 'cpu', etc.

    Returns:
        torch.device instance.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    if preference.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("[Device] CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(preference)
        
    return torch.device(preference)


def get_device_info(device: torch.device | None = None) -> Dict[str, Any]:
    """Retrieve detailed hardware information for the active device.

    Returns:
        Dictionary with device type, GPU name, total VRAM (GB), free VRAM (GB), etc.
    """
    if device is None:
        device = get_device("auto")

    info: Dict[str, Any] = {
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
        "total_memory_gb": 0.0,
        "allocated_memory_gb": 0.0,
        "reserved_memory_gb": 0.0,
        "is_low_vram": False,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        dev_idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(dev_idx)
        total_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)

        info.update({
            "device_name": props.name,
            "device_index": dev_idx,
            "total_memory_gb": round(total_gb, 2),
            "allocated_memory_gb": round(allocated_gb, 2),
            "reserved_memory_gb": round(reserved_gb, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "is_low_vram": total_gb <= 6.5,
        })

    return info


def log_device_info(device: torch.device | None = None) -> Dict[str, Any]:
    """Log formatted device information to console/logger."""
    info = get_device_info(device)
    print("=" * 70)
    print("HARDWARE ACCELERATION & COMPUTE DEVICE INFO")
    print("=" * 70)
    print(f"  - Compute Device:    {info['device_type'].upper()}")
    if info["device_type"] == "cuda":
        print(f"  - GPU Model:         {info['device_name']} (Device ID: {info.get('device_index', 0)})")
        print(f"  - Total VRAM:        {info['total_memory_gb']} GB")
        print(f"  - Compute Capability: {info.get('compute_capability', 'N/A')}")
        if info["is_low_vram"]:
            print("  - VRAM Profile:      LOW VRAM DETECTED (<= 6.5 GB)")
            print("                       [Auto-enabling AMP FP16, Gradient Accumulation & Micro-Batches]")
        else:
            print("  - VRAM Profile:      STANDARD / HIGH CAPACITY VRAM")
    else:
        print("  - GPU Model:         None (Running on Host CPU)")
        print("  - Execution Mode:    CPU Native (Model loading, debug, CPU-inference ready)")
    print("=" * 70)
    return info


def empty_cache() -> None:
    """Clear cached GPU memory and trigger garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warn_vram_risk(strategy: str, model_param_count_m: float, device_info: Dict[str, Any]) -> None:
    """Issue a descriptive warning if training settings risk OOM on the current hardware."""
    if device_info.get("is_low_vram", False) and strategy == "full" and model_param_count_m > 100:
        print("\n" + "!" * 70)
        print("[WARNING] HIGH VRAM DEMAND ON 6GB GPU:")
        print(f"  You are running full fine-tuning on a {model_param_count_m:.1f}M parameter model")
        print(f"  with {device_info['total_memory_gb']} GB available VRAM.")
        print("  Recommendation: Use 'strategy: frozen' or 'strategy: partial' with")
        print("  'gradient_accumulation_steps: 4' and 'mixed_precision: true' to avoid CUDA OOM.")
        print("!" * 70 + "\n")
