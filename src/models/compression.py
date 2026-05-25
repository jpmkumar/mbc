"""Feature compression: 2048 -> 128 -> 32 -> 8."""

import torch
import torch.nn as nn


class FeatureCompression(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 32, 8]
        layers = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
            ])
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
