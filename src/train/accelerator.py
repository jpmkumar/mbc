"""Training speed helpers: AMP, cuDNN, worker limits."""

from __future__ import annotations

import os

import torch


def configure_runtime(config: dict) -> dict:
    """Apply global PyTorch speed settings; return resolved training flags."""
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    if train_cfg.get("cudnn_benchmark", True) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if train_cfg.get("allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    num_workers = int(data_cfg.get("num_workers", 0))
    if data_cfg.get("auto_num_workers", True):
        # Colab and many laptops: >2 workers often stalls or warns
        num_workers = min(num_workers, 2)

    use_amp = bool(train_cfg.get("use_amp", True)) and torch.cuda.is_available()
    cache_batch_size = int(
        train_cfg.get("feature_cache_batch_size", train_cfg.get("batch_size", 16) * 2)
    )

    return {
        "num_workers": num_workers,
        "use_amp": use_amp,
        "cache_batch_size": cache_batch_size,
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 2)),
    }


def amp_device_type(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"


def maybe_compile_model(model, enabled: bool = False):
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)
    except Exception as exc:  # pragma: no cover
        print(f"torch.compile skipped: {exc}")
        return model
