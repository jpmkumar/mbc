"""Modality token embedding and transformer encoder."""

import torch
import torch.nn as nn


class ModalityTokenEmbedding(nn.Module):
    """Learnable modality tokens: [MAMMO], [ULTRA], [THERMO]."""

    def __init__(self, num_modalities: int = 3, embed_dim: int = 256):
        super().__init__()
        self.embeddings = nn.Embedding(num_modalities, embed_dim)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(modality_ids)


class ModalityTransformerEncoder(nn.Module):
    """
    Transformer encoder for modality-invariant representation learning.
    Prepends modality token to pooled feature token.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        embed_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 2048,
        num_modalities: int = 3,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, embed_dim)
        self.modality_embed = ModalityTokenEmbedding(num_modalities, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim
        self.attention_weights = None

    def forward(
        self,
        features: torch.Tensor,
        modality_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        batch_size = features.size(0)
        feat_token = self.feature_proj(features).unsqueeze(1)
        mod_token = self.modality_embed(modality_ids).unsqueeze(1)
        tokens = torch.cat([mod_token, feat_token], dim=1) + self.pos_embed

        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        out = self.output_proj(pooled)

        if return_attention:
            self.attention_weights = encoded
        return out
