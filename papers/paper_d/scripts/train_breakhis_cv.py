#!/usr/bin/env python3
"""Paper D BreakHis runners: F-MLP / F-VQC / T-MLP / T-VQC.

Frozen (F): EfficientNet encoder fully frozen; train proj/transformer/compression/head.
Fine-tune (T): only the last EfficientNet feature block unfrozen, plus the stack above.
Heads: E2b-matched MLP vs Paper-B VQC (8q, 2L, angle-Y, linear).

Primary selection metric: validation case-balanced AUPRC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import create_dataloaders
from src.eval.metrics import evaluate_model
from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel
from src.train.accelerator import configure_runtime
from src.train.losses import FocalLoss, compute_class_weights
from src.train.seed import set_seed
from src.utils.metrics import threshold_sweep

ARMS = {
    "F-MLP": {"regime": "frozen", "head": "mlp"},
    "F-VQC": {"regime": "frozen", "head": "vqc"},
    "T-MLP": {"regime": "finetune", "head": "mlp"},
    "T-VQC": {"regime": "finetune", "head": "vqc"},
}


def case_balanced_auprc(labels, probs, case_ids) -> float:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    case_ids = [str(c) for c in case_ids]
    counts = Counter(case_ids)
    weights = np.asarray([1.0 / counts[c] for c in case_ids], dtype=np.float64)
    try:
        return float(average_precision_score(labels, probs, sample_weight=weights))
    except ValueError:
        return 0.0


def collect_case_ids(loader) -> list[str]:
    ids = []
    for batch in loader:
        raw = batch["patient_id"]
        if isinstance(raw, (list, tuple)):
            ids.extend(str(x) for x in raw)
        else:
            ids.extend(str(x) for x in raw)
    return ids


@torch.no_grad()
def evaluate_with_cases(model, loader, device, threshold=0.5, tta=False) -> dict:
    metrics = evaluate_model(model, loader, device, threshold=threshold, tta=tta)
    # Re-walk loader for case ids in the same order as evaluate_model.
    case_ids = collect_case_ids(loader)
    if len(case_ids) != len(metrics["labels"]):
        raise RuntimeError(
            f"case_id count {len(case_ids)} != label count {len(metrics['labels'])}"
        )
    metrics["case_balanced_auprc"] = case_balanced_auprc(
        metrics["labels"], metrics["probs"], case_ids
    )
    metrics["case_ids"] = case_ids
    return metrics


def apply_regime(model: nn.Module, regime: str, last_blocks: int = 1) -> None:
    encoder = model.encoder
    if regime == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False
        print("Regime F: EfficientNet encoder fully frozen")
        return
    if regime == "finetune":
        n = len(encoder.features)
        freeze_until = max(0, n - int(last_blocks))
        encoder.freeze_early(num_blocks=freeze_until)
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())
        print(
            f"Regime T: last {last_blocks} EfficientNet block(s) trainable "
            f"({trainable}/{total} encoder params)"
        )
        return
    raise ValueError(f"unknown regime: {regime}")


def build_model(config: dict, head: str):
    model_cfg = config["model"]
    quantum_cfg = model_cfg["quantum"]
    kwargs = dict(
        num_modalities=model_cfg["num_modality_tokens"],
        num_classes=model_cfg["num_classes"],
        encoder_dim=model_cfg["encoder_dim"],
        projection_dim=model_cfg["projection_dim"],
        compression_dims=list(model_cfg["compression_dims"]),
        transformer_layers=model_cfg["transformer"]["num_layers"],
        transformer_heads=model_cfg["transformer"]["num_heads"],
        classical_head_type="mlp",
        classical_head_hidden=model_cfg.get(
            "classical_head_hidden", model_cfg["compression_dims"][-1]
        ),
    )
    if head == "mlp":
        return ClassicalBreastCancerModel(**kwargs)
    if head == "vqc":
        model = HybridBreastCancerModel(
            **kwargs,
            n_qubits=int(quantum_cfg["n_qubits"]),
            n_vqc_layers=int(quantum_cfg["n_layers"]),
            entanglement=quantum_cfg.get("entanglement", "linear"),
            quantum_encoding=quantum_cfg.get("encoding", "angle_y"),
            quantum_data_reuploading=bool(quantum_cfg.get("data_reuploading", False)),
            quantum_feature_norm=bool(quantum_cfg.get("feature_norm", True)),
            quantum_full_readout=bool(quantum_cfg.get("full_readout", True)),
            quantum_backend=quantum_cfg.get("backend"),
            quantum_diff_method=quantum_cfg.get("diff_method"),
            use_quantum=True,
            use_fusion=False,
        )
        # Forward through VQC head for the whole run (not staged A→B).
        model.set_training_stage("stage_b")
        model.set_classical_head_trainable(False)
        model.set_vqc_head_trainable(True)
        return model
    raise ValueError(f"unknown head: {head}")


def build_criterion(train_cfg: dict, train_loader, device: str):
    weights = None
    if train_cfg.get("use_class_weights", True):
        weights = compute_class_weights(train_loader).to(device)
    if train_cfg.get("loss", "focal") == "focal":
        return FocalLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=weights
        ).to(device)
    return nn.CrossEntropyLoss(weight=weights).to(device)


def train_one_arm(config: dict, arm: str, fold: int, quick: bool = False) -> dict:
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {sorted(ARMS)}")
    spec = ARMS[arm]
    seed = int(config["project"]["seed"])
    set_seed(seed + fold)

    data_cfg = config["data"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    runtime_dir = ROOT / data_cfg["splits_dir"] / "runtime" / f"fold_{fold}"
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (runtime_dir / name).is_file():
            raise SystemExit(
                f"missing {runtime_dir / name}; run "
                f"prepare_breakhis_manifests.py --fold {fold} first"
            )

    configure_runtime(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} arm={arm} fold={fold}")

    loaders = create_dataloaders(
        {
            "train": str(runtime_dir / "train.csv"),
            "val": str(runtime_dir / "val.csv"),
            "test": str(runtime_dir / "test.csv"),
        },
        batch_size=int(train_cfg["batch_size"]),
        image_size=int(data_cfg["image_size"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        modality_filter=list(data_cfg.get("modalities", ["histopath"])),
        preprocess_config=data_cfg.get("preprocessing"),
        data_root=str(ROOT / data_cfg["root"]),
        augment_config=data_cfg.get("augmentation"),
        max_samples=64 if quick else None,
        max_eval_samples=64 if quick else None,
    )

    model = build_model(config, spec["head"])
    quantum_device = "cpu" if spec["head"] == "vqc" else device
    # Match HybridTrainer: place classical on GPU and VQC head on CPU via
    # set_devices. Do NOT call model.to(cuda) afterwards — that would move the
    # VQC LayerNorm to CUDA while forward still feeds CPU features.
    if hasattr(model, "set_devices"):
        model.set_devices(
            classical_device=device,
            quantum_device=quantum_device,
        )
        print(f"devices: classical={device} quantum={quantum_device}")
    else:
        model = model.to(device)

    apply_regime(
        model,
        spec["regime"],
        last_blocks=int(train_cfg.get("finetune_last_blocks", 1)),
    )

    if train_cfg.get("channels_last", True) and device == "cuda":
        model.encoder.to(memory_format=torch.channels_last)
        if hasattr(model, "set_devices"):
            model.set_devices(device, quantum_device)

    # AMP is unsafe across the CPU VQC boundary; keep it for MLP-only arms.
    use_amp = device == "cuda" and spec["head"] == "mlp"
    criterion = build_criterion(train_cfg, loaders["train"], device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs = 2 if quick else int(train_cfg["epochs"])
    patience = int(train_cfg.get("early_stopping_patience", 5))
    clip = float(train_cfg.get("grad_clip_norm", 1.0))
    tta = bool(train_cfg.get("tta", True)) and not quick

    out_dir = ROOT / paths_cfg["results"] / f"fold{fold}_{arm}"
    ckpt_dir = ROOT / paths_cfg["checkpoints"] / f"fold{fold}_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    best_score = -1.0
    best_epoch = -1
    stale = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep frozen encoder in eval for stable BN when regime=F.
        if spec["regime"] == "frozen":
            model.encoder.eval()
        running = 0.0
        n = 0
        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            if train_cfg.get("channels_last", True) and device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            labels = batch["label"].to(device, non_blocking=True)
            modality_ids = batch["modality_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images, modality_ids)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * labels.size(0)
            n += labels.size(0)

        val_metrics = evaluate_with_cases(
            model, loaders["val"], device, threshold=0.5, tta=False
        )
        score = float(val_metrics["case_balanced_auprc"])
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_case_balanced_auprc": score,
            "val_auprc": float(val_metrics["auprc"]),
            "val_auc": float(val_metrics["auc"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} loss={row['train_loss']:.4f} "
            f"val_cb_auprc={score:.4f} val_auprc={row['val_auprc']:.4f}"
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arm": arm,
                    "fold": fold,
                    "epoch": epoch,
                    "val_case_balanced_auprc": score,
                    "config": config,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break

    # Load best and score test
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if hasattr(model, "set_devices"):
        model.set_devices(device, quantum_device)
    if spec["head"] == "vqc":
        model.set_training_stage("stage_b")

    threshold = float(train_cfg.get("eval_threshold", 0.5))
    thr_meta = {"tuned": False, "threshold": threshold}
    if train_cfg.get("tune_threshold", True):
        val_full = evaluate_with_cases(
            model, loaders["val"], device, threshold=0.5, tta=tta
        )
        metric_name = str(train_cfg.get("threshold_metric", "fbeta"))
        if metric_name == "fbeta":
            _, best = threshold_sweep(
                val_full["labels"],
                val_full["probs"],
                metric="fbeta",
                beta=float(train_cfg.get("fbeta_beta", 1.5)),
            )
        else:
            _, best = threshold_sweep(
                val_full["labels"], val_full["probs"], metric=metric_name
            )
        threshold = float(best["threshold"])
        thr_meta = {
            "tuned": True,
            "threshold": threshold,
            "threshold_metric": metric_name,
            "val_score_at_threshold": float(best.get(metric_name, best.get("fbeta", 0.0))),
        }

    test_metrics = evaluate_with_cases(
        model, loaders["test"], device, threshold=threshold, tta=tta
    )
    summary = {
        "arm": arm,
        "regime": spec["regime"],
        "head": spec["head"],
        "fold": fold,
        "seed": seed,
        "effective_seed": seed + fold,
        "best_epoch": best_epoch,
        "best_val_case_balanced_auprc": best_score,
        "threshold": thr_meta,
        "test": {
            "case_balanced_auprc": float(test_metrics["case_balanced_auprc"]),
            "auprc": float(test_metrics["auprc"]),
            "auc": float(test_metrics["auc"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "n_samples": int(test_metrics["n_samples"]),
        },
        "train_time_s": time.time() - t0,
        "history": history,
        "checkpoint": str(best_path.relative_to(ROOT)),
    }
    # Drop large arrays from disk summary
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        labels=np.asarray(test_metrics["labels"]),
        probs=np.asarray(test_metrics["probs"]),
        case_ids=np.asarray(test_metrics["case_ids"]),
    )
    print(json.dumps(summary["test"], indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "papers" / "paper_d" / "config" / "breakhis.yaml",
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_one_arm(config, args.arm, args.fold, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
