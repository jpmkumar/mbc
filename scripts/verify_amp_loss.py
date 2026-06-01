"""Smoke tests for AMP + weighted cross-entropy (matches Colab failure)."""

import torch
import torch.nn as nn


def test_amp_weighted_cross_entropy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("SKIP: CUDA not available locally")
        return

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.8], device=device)
    )
    images = torch.randn(4, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1, 0, 1], device=device)

    linear = nn.Linear(3 * 224 * 224, 2).to(device)

    with torch.autocast("cuda"):
        flat = images.view(4, -1)
        logits = linear(flat)

    # This pattern failed before the fix:
    loss = criterion(logits.float(), labels)
    loss.backward()
    print("AMP weighted CE OK:", float(loss))


if __name__ == "__main__":
    test_amp_weighted_cross_entropy()
