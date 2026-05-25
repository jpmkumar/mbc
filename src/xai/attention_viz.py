"""Transformer attention visualization."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_heatmap(
    model,
    image: torch.Tensor,
    modality_id: torch.Tensor,
    save_path: str,
    title: str = "Transformer Attention",
):
    model.eval()
    with torch.no_grad():
        _ = model.forward_features(
            image.unsqueeze(0) if image.dim() == 3 else image,
            modality_id.unsqueeze(0) if modality_id.dim() == 0 else modality_id,
            return_attention=True,
        )

    if model.transformer is None or model.transformer.attention_weights is None:
        attn = np.ones((2, 2)) / 2
    else:
        tokens = model.transformer.attention_weights.squeeze(0).cpu().numpy()
        attn = np.abs(tokens).mean(axis=1, keepdims=True)
        attn = attn / (attn.sum() + 1e-8)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(attn, cmap="hot", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Modality", "Feature"])
    ax.set_yticks([0])
    ax.set_yticklabels(["Pooled"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
