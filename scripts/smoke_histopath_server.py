#!/usr/bin/env python3
"""Train-only GPU smoke test using the real histopathology model and loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_histopath_cv import (  # noqa: E402
    _apply_quantum_ablation_overrides,
    _build_model,
    _prepare_fold_manifests,
)
from src.data.constants import HISTOPATH_MODALITY  # noqa: E402
from src.data.dataloaders import create_dataloaders  # noqa: E402
from src.train.accelerator import configure_runtime  # noqa: E402
from src.train.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-qubits", type=int, default=8, choices=(4, 8, 12))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the server smoke test.")
    config = yaml.safe_load((ROOT / "configs/histopath.yaml").read_text())
    _apply_quantum_ablation_overrides(
        config,
        SimpleNamespace(
            n_qubits=args.n_qubits,
            n_layers=None,
            entanglement=None,
            encoding=None,
            data_reuploading=None,
        ),
    )
    set_seed(42 + args.fold)
    runtime = configure_runtime(config)
    train_cfg = config["training"]
    data_cfg = config["data"]

    manifests = _prepare_fold_manifests(
        args.splits_dir,
        args.archive_path,
        args.fold,
        float(train_cfg.get("val_ratio", 0.1)),
        42,
    )
    loader = create_dataloaders(
        {"train": manifests["train"]},
        batch_size=train_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=runtime["num_workers"],
        modality_filter=[HISTOPATH_MODALITY],
        preprocess_config=data_cfg.get("preprocessing"),
        prefetch_factor=runtime["prefetch_factor"],
        data_root=str(args.archive_path),
        max_samples=args.max_samples,
        augment_config=data_cfg.get("augmentation"),
    )["train"]

    model = _build_model(config, "E3")
    model.set_devices("cuda", "cpu")
    model.set_training_stage("stage_a")
    model.set_vqc_head_trainable(False)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["use_amp"])

    losses = []
    samples = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to("cuda", non_blocking=True)
        labels = batch["label"].to("cuda", non_blocking=True)
        modalities = batch["modality_id"].to("cuda", non_blocking=True)
        if runtime["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=runtime["use_amp"]):
            logits = model(images, modalities)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke-test loss.")
        losses.append(float(loss.detach().cpu()))
        samples += int(labels.numel())
        if step >= args.steps:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "status": "PASS",
        "scope": "train_only_stage_a_real_model_and_augmentation",
        "fold": args.fold,
        "n_qubits": args.n_qubits,
        "steps": len(losses),
        "samples": samples,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

