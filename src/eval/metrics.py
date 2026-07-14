"""Model evaluation metrics."""

import torch
import torch.nn as nn

from src.utils.metrics import compute_metrics, preds_from_threshold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def _tta_views(images: torch.Tensor):
    """Symmetry-preserving views for histopath patches (flips + 90-deg rots)."""
    return [
        images,
        torch.flip(images, dims=[3]),          # horizontal flip
        torch.flip(images, dims=[2]),          # vertical flip
        torch.rot90(images, k=1, dims=[2, 3]),  # 90 deg
        torch.rot90(images, k=2, dims=[2, 3]),  # 180 deg
        torch.rot90(images, k=3, dims=[2, 3]),  # 270 deg
    ]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: str,
    threshold: float = 0.5,
    tta: bool = False,
) -> dict:
    model.eval()
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        modality_ids = batch["modality_id"].to(device)

        if tta:
            prob_sum = None
            for view in _tta_views(images):
                logits = model(view, modality_ids)
                p = torch.softmax(logits.float(), dim=1)
                prob_sum = p if prob_sum is None else prob_sum + p
            probs = prob_sum / len(_tta_views(images))
        else:
            logits = model(images, modality_ids)
            probs = torch.softmax(logits.float(), dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = preds_from_threshold(all_probs, threshold).tolist()
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["threshold"] = float(threshold)
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
