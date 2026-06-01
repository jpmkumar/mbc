#!/usr/bin/env python3
"""Analyze a trained checkpoint: predictions, confusion matrix, threshold sweep."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import create_dataloaders
from src.data.splits import load_splits
from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel
from src.train.trainer import filter_compatible_state_dict
from src.utils.metrics import (
    compute_metrics_at_threshold,
    threshold_for_target_recall,
    threshold_sweep,
)


def load_model(config: dict, hybrid: bool = True):
    mc = config["model"]
    qcfg = mc.get("quantum", {})
    kwargs = dict(
        num_modalities=mc["num_modality_tokens"],
        num_classes=mc["num_classes"],
        encoder_dim=mc["encoder_dim"],
        projection_dim=mc["projection_dim"],
        compression_dims=mc["compression_dims"],
        transformer_layers=mc["transformer"]["num_layers"],
        transformer_heads=mc["transformer"]["num_heads"],
    )
    if hybrid:
        return HybridBreastCancerModel(
            **kwargs,
            n_qubits=qcfg.get("n_qubits", 8),
            n_vqc_layers=qcfg.get("n_layers", 2),
            entanglement=qcfg.get("entanglement", "linear"),
            quantum_feature_norm=qcfg.get("feature_norm", True),
            quantum_full_readout=qcfg.get("full_readout", True),
        )
    return ClassicalBreastCancerModel(**kwargs)


def setup_device(model, config: dict) -> torch.device:
    train_cfg = config.get("training", {})
    if getattr(model, "use_quantum", False) and hasattr(model, "set_devices"):
        if torch.cuda.is_available():
            classical = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            classical = "mps"
        else:
            classical = "cpu"
        classical = train_cfg.get("classical_device", classical)
        if classical == "auto":
            classical = "cuda" if torch.cuda.is_available() else "cpu"
        quantum = train_cfg.get("quantum_device", "cpu")
        model.set_devices(classical, quantum)
        return torch.device(classical)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return torch.device(device)


def load_checkpoint_weights(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_stage = "stage_a"
    if isinstance(ckpt, dict) and ckpt.get("best_state_dict"):
        state = ckpt["best_state_dict"]
        best_stage = ckpt.get("best_stage", "stage_a")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        best_stage = ckpt.get("best_stage", "stage_a")
    else:
        state = ckpt
    filtered, skipped = filter_compatible_state_dict(model, state)
    model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"Note: skipped {len(skipped)} shape-mismatch keys from checkpoint")
    return best_stage


def collect_probs(model, loader, device):
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            modality_ids = batch["modality_id"].to(device)
            logits = model(images, modality_ids)
            p = torch.softmax(logits, 1)
            labels.extend(batch["label"].tolist())
            probs.extend(p[:, 1].tolist())
    return labels, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", default="configs/mammo_enhanced.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--modality",
        default=None,
        choices=["mammo", "ultrasound", "thermo"],
    )
    parser.add_argument("--classical", action="store_true")
    parser.add_argument(
        "--eval-stage",
        default="a",
        choices=["a", "b", "c"],
        help="Head for hybrid eval: a=classical (Stage A), b/c=VQC",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold on P(malignant). Default: config or 0.5",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Search thresholds on this split; report best balanced accuracy",
    )
    parser.add_argument(
        "--sweep-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "recall", "f1"],
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Pick threshold achieving at least this recall (e.g. 0.85)",
    )
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    modality_filter = [args.modality] if args.modality else None
    preprocess_config = config.get("data", {}).get("preprocessing")
    splits = load_splits(str(ROOT / "data/splits"))
    loaders = create_dataloaders(
        splits,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
        modality_filter=modality_filter,
        preprocess_config=preprocess_config,
    )
    loader = loaders[args.split]

    model = load_model(config, hybrid=not args.classical)
    device = setup_device(model, config)
    best_stage = load_checkpoint_weights(model, Path(args.checkpoint))
    model.eval()

    if hasattr(model, "set_training_stage"):
        stage = args.eval_stage
        model.set_training_stage("stage_b" if stage in ("b", "c") else "stage_a")
        report_stage = stage
    else:
        report_stage = "classical"

    labels, probs = collect_probs(model, loader, device)
    default_threshold = config.get("training", {}).get("eval_threshold", 0.5)

    report = {
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "split": args.split,
        "eval_stage": report_stage,
        "best_stage_in_ckpt": best_stage,
        "modality_filter": args.modality,
        "label_distribution": dict(Counter(labels)),
        "preprocessing_enabled": bool(
            preprocess_config and preprocess_config.get("enabled")
        ),
    }

    if args.threshold_sweep:
        rows, best = threshold_sweep(labels, probs, metric=args.sweep_metric)
        report["threshold_sweep"] = [
            {
                k: v
                for k, v in row.items()
                if k not in ("roc", "labels", "preds", "probs")
            }
            for row in rows
        ]
        report["best_threshold"] = best
        metrics = best
    elif args.target_recall is not None:
        threshold, metrics = threshold_for_target_recall(
            labels, probs, target_recall=args.target_recall
        )
        report["threshold"] = threshold
        report["target_recall"] = args.target_recall
    else:
        threshold = (
            args.threshold if args.threshold is not None else default_threshold
        )
        metrics = compute_metrics_at_threshold(labels, probs, threshold)
        report["threshold"] = threshold

    preds = metrics.get("preds")
    if preds is None:
        from src.utils.metrics import preds_from_threshold

        preds = preds_from_threshold(probs, report.get("threshold", 0.5)).tolist()

    collapsed = len(set(preds)) == 1
    report.update(
        {
            "pred_distribution": dict(Counter(preds)),
            "class_collapse": collapsed,
            **{
                k: v
                for k, v in metrics.items()
                if k not in ("labels", "preds", "probs", "roc")
            },
        }
    )

    print(json.dumps(report, indent=2))
    if collapsed:
        print("\nWARNING: model predicts a single class for all samples (class collapse).")


if __name__ == "__main__":
    main()
