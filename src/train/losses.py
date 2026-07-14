"""Loss utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights easy examples so training focuses on hard/ambiguous
    patches. ``weight`` acts as the per-class alpha term (reuse the same
    class weights as weighted CE). ``gamma`` controls how aggressively
    easy examples are suppressed (gamma=0 reduces to weighted CE).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        # Unweighted per-sample CE so pt reflects the true class probability.
        ce = F.nll_loss(log_probs, target, reduction="none")
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce

        if self.weight is not None:
            alpha = self.weight.gather(0, target)
            focal = alpha * focal
        else:
            alpha = None

        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        # Weighted mean matches nn.CrossEntropyLoss convention (so gamma=0
        # reduces exactly to weighted CE).
        if alpha is not None:
            return focal.sum() / alpha.sum().clamp(min=1e-8)
        return focal.mean()


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
