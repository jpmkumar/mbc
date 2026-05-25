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
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    if hasattr(model, "set_training_stage"):
        model.set_training_stage("stage_b" if model.use_quantum else "stage_a")

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
