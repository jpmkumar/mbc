"""Full hybrid and classical model definitions."""

import torch
import torch.nn as nn

from .compression import FeatureCompression
from .encoder import EfficientNetEncoder
from .transformer import ModalityTransformerEncoder
from .vqc import VQCHead


class HybridBreastCancerModel(nn.Module):
    """
    Modality-Level Generalized Hybrid Quantum Framework.
    EfficientNet-B0 -> Transformer -> Compression -> VQC -> Classification
    """

    def __init__(
        self,
        num_modalities: int = 3,
        num_classes: int = 2,
        encoder_dim: int = 1280,
        projection_dim: int = 2048,
        compression_dims: list[int] | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        n_qubits: int = 8,
        n_vqc_layers: int = 2,
        use_modality_tokens: bool = True,
        use_transformer: bool = True,
        use_quantum: bool = True,
    ):
        super().__init__()
        compression_dims = compression_dims or [128, 32, 8]
        self.use_modality_tokens = use_modality_tokens
        self.use_transformer = use_transformer
        self.use_quantum = use_quantum

        self.encoder = EfficientNetEncoder(output_dim=encoder_dim)
        self.feature_proj = nn.Linear(encoder_dim, projection_dim)

        if use_transformer:
            self.transformer = ModalityTransformerEncoder(
                input_dim=encoder_dim,
                embed_dim=512,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                output_dim=projection_dim,
                num_modalities=num_modalities,
            )
        else:
            self.transformer = None

        self.compression = FeatureCompression(
            input_dim=projection_dim,
            hidden_dims=compression_dims,
        )

        qubit_dim = compression_dims[-1]
        self.classical_head = nn.Linear(qubit_dim, num_classes)
        if use_quantum:
            self.head = VQCHead(
                n_qubits=n_qubits,
                n_layers=n_vqc_layers,
                num_classes=num_classes,
            )
        else:
            self.head = self.classical_head

        self.n_qubits = n_qubits
        self._use_classical_head = not use_quantum  # stage A uses classical head for hybrid

    def forward_features(
        self,
        images: torch.Tensor,
        modality_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        features = self.encoder(images)

        if self.use_transformer and self.use_modality_tokens:
            features = self.transformer(
                features, modality_ids, return_attention=return_attention
            )
        elif self.use_transformer:
            zero_mod = torch.zeros_like(modality_ids)
            features = self.transformer(
                features, zero_mod, return_attention=return_attention
            )
        else:
            features = self.feature_proj(features)

        compressed = self.compression(features)
        return compressed

    def set_training_stage(self, stage: str):
        """Stage A: classical head; Stage B/C: VQC head (hybrid models only)."""
        if self.use_quantum:
            self._use_classical_head = stage == "stage_a"

    def set_backbone_trainable(self, trainable: bool):
        """Freeze or unfreeze encoder, transformer, and compression."""
        for module in (self.encoder, self.transformer, self.compression, self.feature_proj):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = trainable

    def set_classical_head_trainable(self, trainable: bool):
        for param in self.classical_head.parameters():
            param.requires_grad = trainable

    def set_vqc_head_trainable(self, trainable: bool):
        if not self.use_quantum:
            return
        for param in self.head.parameters():
            param.requires_grad = trainable

    def forward(
        self,
        images: torch.Tensor,
        modality_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        compressed = self.forward_features(images, modality_ids, return_attention)
        if self.use_quantum and self._use_classical_head:
            return self.classical_head(compressed)
        return self.head(compressed)


class ClassicalBreastCancerModel(HybridBreastCancerModel):
    """Unified classical model without quantum head."""

    def __init__(self, **kwargs):
        kwargs["use_quantum"] = False
        super().__init__(**kwargs)
