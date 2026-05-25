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
