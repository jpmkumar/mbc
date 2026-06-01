"""Shared classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def preds_from_threshold(probs, threshold: float = 0.5):
    return (np.asarray(probs) >= threshold).astype(int)


def compute_metrics(all_labels, all_preds, all_probs) -> dict:
    """Compute classification metrics from aggregated predictions."""
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy": float(
            balanced_accuracy_score(all_labels, all_preds)
        ),
        "precision": float(
            precision_score(all_labels, all_preds, zero_division=0)
        ),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "n_samples": len(all_labels),
        "pred_positive_rate": float(np.mean(all_preds)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        metrics["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def compute_metrics_at_threshold(
    labels, probs, threshold: float = 0.5
) -> dict:
    preds = preds_from_threshold(probs, threshold)
    metrics = compute_metrics(labels, preds, probs)
    metrics["threshold"] = float(threshold)
    return metrics


def threshold_sweep(
    labels,
    probs,
    metric: str = "balanced_accuracy",
    thresholds: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Try many cutoffs; return all rows and the best row for `metric`."""
    thresholds = (
        thresholds
        if thresholds is not None
        else np.linspace(0.05, 0.95, 19)
    )
    rows = []
    for threshold in thresholds:
        row = compute_metrics_at_threshold(labels, probs, float(threshold))
        rows.append(row)

    def score(row: dict) -> float:
        if metric == "recall":
            return row["recall"]
        if metric == "f1":
            return row["f1"]
        return row.get(metric, row["balanced_accuracy"])

    best = max(rows, key=score)
    return rows, best


def threshold_for_target_recall(
    labels, probs, target_recall: float = 0.85
) -> tuple[float, dict]:
    """Pick the highest threshold that still achieves target recall on this split."""
    thresholds = np.linspace(0.05, 0.95, 37)
    candidates = []
    for threshold in thresholds:
        row = compute_metrics_at_threshold(labels, probs, float(threshold))
        if row["recall"] >= target_recall:
            candidates.append(row)
    if not candidates:
        _, best = threshold_sweep(labels, probs, metric="recall")
        return best["threshold"], best
    best = max(candidates, key=lambda row: row["threshold"])
    return best["threshold"], best
