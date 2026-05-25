"""Generate 2x2 clinical case study figure for paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, resize

from .gradcam import GradCAM


def _denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.cpu() * std + mean


def generate_case_study_figure(
    model,
    sample: dict,
    device: str,
    save_path: str,
    label_names=("Benign", "Malignant"),
):
    model.eval()
    image = sample["image"].to(device)
    modality_id = sample["modality_id"].to(device)
    true_label = sample["label"].item()

    with torch.no_grad():
        logits = model(image.unsqueeze(0), modality_id.unsqueeze(0))
        pred = logits.argmax(dim=1).item()
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    # Grad-CAM
    target_layer = model.encoder.get_last_conv_layer()
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image, modality_id, class_idx=pred)

    img_display = _denormalize(image).permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    cam_resized = np.array(
        resize(
            to_pil_image(cam.astype(np.float32)),
            (image.shape[1], image.shape[2]),
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f"Input ({sample.get('modality', 'unknown')})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_display)
    axes[0, 1].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[0, 1].set_title("Grad-CAM")
    axes[0, 1].axis("off")

    # Attention placeholder from transformer tokens
    if model.transformer is not None:
        with torch.no_grad():
            _ = model.forward_features(
                image.unsqueeze(0), modality_id.unsqueeze(0), return_attention=True
            )
        tokens = model.transformer.attention_weights.squeeze(0).cpu().numpy()
        attn_map = np.abs(tokens).mean(axis=0)
        axes[1, 0].bar(["Modality", "Feature"], attn_map[:2])
        axes[1, 0].set_title("Attention Weights")
    else:
        axes[1, 0].text(0.5, 0.5, "No Transformer", ha="center", va="center")
        axes[1, 0].set_title("Attention")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.1, 0.7,
        f"True: {label_names[true_label]}\n"
        f"Pred: {label_names[pred]}\n"
        f"Malignant prob: {prob:.3f}",
        fontsize=14,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].set_title("Prediction + SHAP-ready features")

    plt.suptitle("Explainability Case Study", fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
