#!/usr/bin/env python3
"""Analyze a trained checkpoint: predictions, confusion matrix, collapse check."""

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
from src.utils.metrics import compute_metrics
from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel


def load_model(config: dict, hybrid: bool = True):
    mc = config["model"]
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
            n_qubits=mc["quantum"]["n_qubits"],
            n_vqc_layers=mc["quantum"]["n_layers"],
        )
    return ClassicalBreastCancerModel(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--modality", default=None, choices=["mammo", "ultrasound", "thermo"])
    parser.add_argument("--classical", action="store_true")
    parser.add_argument(
        "--eval-stage",
        default="a",
        choices=["a", "b", "c"],
        help="Head for hybrid eval: a=classical (Stage A), b=VQC, c=VQC",
    )
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    modality_filter = [args.modality] if args.modality else None
    splits = load_splits(str(ROOT / "data/splits"))
    loaders = create_dataloaders(
        splits,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
        modality_filter=modality_filter,
    )
    loader = loaders[args.split]

    model = load_model(config, hybrid=not args.classical)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "best_state_dict" in ckpt and ckpt["best_state_dict"]:
        model.load_state_dict(ckpt["best_state_dict"])
        best_stage = ckpt.get("best_stage", "stage_a")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        best_stage = ckpt.get("best_stage", "stage_a")
    else:
        model.load_state_dict(ckpt)
        best_stage = "stage_a"
    model.eval()
    if hasattr(model, "set_training_stage"):
        stage = args.eval_stage
        if stage in ("b", "c"):
            model.set_training_stage("stage_b")
        else:
            model.set_training_stage("stage_a")
        report_stage = stage
    else:
        report_stage = "classical"

    labels, preds, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"], batch["modality_id"])
            p = torch.softmax(logits, 1)
            preds.extend(logits.argmax(1).tolist())
            labels.extend(batch["label"].tolist())
            probs.extend(p[:, 1].tolist())

    metrics = compute_metrics(labels, preds, probs)
    collapsed = len(set(preds)) == 1

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "eval_stage": report_stage,
        "best_stage_in_ckpt": best_stage if hasattr(model, "use_quantum") else "classical",
        "modality_filter": args.modality,
        "label_distribution": dict(Counter(labels)),
        "pred_distribution": dict(Counter(preds)),
        "class_collapse": collapsed,
        **{k: v for k, v in metrics.items() if k not in ("labels", "preds", "probs", "roc")},
    }

    print(json.dumps(report, indent=2))
    if collapsed:
        print("\nWARNING: model predicts a single class for all samples (class collapse).")


if __name__ == "__main__":
    main()
