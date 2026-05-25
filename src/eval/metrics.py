"""Model evaluation metrics."""

import torch
import torch.nn as nn

from src.utils.metrics import compute_metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: str) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        modality_ids = batch["modality_id"].to(device)
        logits = model(images, modality_ids)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["labels"] = all_labels
    metrics["preds"] = all_preds
    metrics["probs"] = all_probs
    return metrics


@torch.no_grad()
def evaluate_by_modality(model: nn.Module, loader, device: str) -> dict:
    model.eval()
    by_mod = {}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        modality_ids = batch["modality_id"].to(device)
        modalities = batch["modality"]

        logits = model(images, modality_ids)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i, mod in enumerate(modalities):
            if mod not in by_mod:
                by_mod[mod] = {"labels": [], "preds": [], "probs": []}
            by_mod[mod]["labels"].append(labels[i].item())
            by_mod[mod]["preds"].append(preds[i].item())
            by_mod[mod]["probs"].append(probs[i, 1].item())

    results = {}
    for mod, data in by_mod.items():
        results[mod] = {
            "accuracy": float(accuracy_score(data["labels"], data["preds"])),
            "f1": float(f1_score(data["labels"], data["preds"], zero_division=0)),
            "n_samples": len(data["labels"]),
        }
        try:
            results[mod]["auc"] = float(roc_auc_score(data["labels"], data["probs"]))
        except ValueError:
            results[mod]["auc"] = 0.0
    return results
