"""Loss utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_class_weights(
    train_loader: DataLoader,
    num_classes: int = 2,
    malignant_multiplier: float = 1.0,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for batch in train_loader:
        labels = batch["label"]
        for c in range(num_classes):
            counts[c] += (labels == c).sum().float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    if malignant_multiplier != 1.0 and num_classes >= 2:
        weights[1] = weights[1] * float(malignant_multiplier)
    return weights
