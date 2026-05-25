"""Grad-CAM for EfficientNet-B0 encoder."""

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image: torch.Tensor, modality_id: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        image = image.unsqueeze(0) if image.dim() == 3 else image
        modality_id = modality_id.unsqueeze(0) if modality_id.dim() == 0 else modality_id

        logits = self.model(image, modality_id)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
