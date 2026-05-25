"""EfficientNet-B0 feature encoder."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetEncoder(nn.Module):
    def __init__(self, output_dim: int = 1280, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x

    def freeze_early(self, num_blocks: int = 3):
        """Freeze first N feature blocks for staged fine-tuning."""
        for i, block in enumerate(self.features):
            for param in block.parameters():
                param.requires_grad = i >= num_blocks

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self):
        """Return last conv module for Grad-CAM."""
        return self.features[-1]
