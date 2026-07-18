"""Full hybrid and classical model definitions."""

from contextlib import nullcontext

import torch
import torch.nn as nn

from .compression import FeatureCompression
from .encoder import EfficientNetEncoder
from .transformer import ModalityTransformerEncoder
from .vqc import VQCHead


class ClassicalMLPHead(nn.Module):
    """Classical control head that mirrors the VQC head's structure.

    Uses LayerNorm + a non-linear MLP (matching the VQC's LayerNorm +
    non-linear quantum transform + linear classifier), with equal or
    greater capacity than the VQC head. If the VQC still outperforms this
    head, any advantage cannot be attributed merely to extra parameters or
    non-linearity — isolating the genuine quantum contribution.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridBreastCancerModel(nn.Module):
    """
    Modality-Level Generalized Hybrid Quantum Framework.
    EfficientNet-B0 -> Transformer -> Compression -> head(s)

    Head modes:
    - classical only (use_quantum=False): linear or MLP
    - quantum swap (use_quantum=True, use_fusion=False): VQC replaces classical
    - fusion (use_fusion=True): classical MLP + VQC logits mixed by alpha
      fused = alpha * logits_classical + (1 - alpha) * logits_vqc
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
        entanglement: str = "linear",
        quantum_encoding: str = "angle_y",
        quantum_data_reuploading: bool = False,
        quantum_feature_norm: bool = True,
        quantum_full_readout: bool = True,
        quantum_backend: str | None = None,
        quantum_diff_method: str | None = None,
        classical_head_type: str = "linear",
        classical_head_hidden: int | None = None,
        use_modality_tokens: bool = True,
        use_transformer: bool = True,
        use_quantum: bool = True,
        use_fusion: bool = False,
        fusion_alpha: float | None = None,
        fusion_init_alpha: float = 0.5,
    ):
        super().__init__()
        compression_dims = compression_dims or [128, 32, 8]
        self.use_modality_tokens = use_modality_tokens
        self.use_transformer = use_transformer
        # Fusion implies a quantum path; keep use_quantum True for trainer stages.
        self.use_fusion = bool(use_fusion)
        self.use_quantum = bool(use_quantum) or self.use_fusion

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
        # Fusion always uses the matched MLP classical branch (E2b-style).
        head_type = "mlp" if self.use_fusion else classical_head_type
        if head_type == "mlp":
            self.classical_head = ClassicalMLPHead(
                qubit_dim, num_classes, hidden=classical_head_hidden or qubit_dim
            )
        else:
            self.classical_head = nn.Linear(qubit_dim, num_classes)

        if self.use_quantum:
            self.head = VQCHead(
                n_qubits=n_qubits,
                n_layers=n_vqc_layers,
                num_classes=num_classes,
                entanglement=entanglement,
                encoding=quantum_encoding,
                data_reuploading=quantum_data_reuploading,
                feature_norm=quantum_feature_norm,
                full_readout=quantum_full_readout,
                backend=quantum_backend,
                diff_method=quantum_diff_method,
            )
        else:
            self.head = self.classical_head

        self.n_qubits = n_qubits
        self._use_classical_head = not self.use_quantum
        # When True (stage C / eval for E4), mix classical + VQC logits.
        self._fusion_active = False
        self._backbone_frozen = False
        self.classical_device = torch.device("cpu")
        self.quantum_device = torch.device("cpu")

        self._fusion_fixed_alpha: float | None = None
        if self.use_fusion:
            if fusion_alpha is not None:
                self._fusion_fixed_alpha = float(fusion_alpha)
                self.fusion_logit = None
            else:
                init = min(max(float(fusion_init_alpha), 1e-4), 1.0 - 1e-4)
                # logit(init) so sigmoid(fusion_logit) starts near init_alpha
                logit = torch.log(torch.tensor(init / (1.0 - init)))
                self.fusion_logit = nn.Parameter(logit)
        else:
            self.fusion_logit = None

    def set_devices(self, classical_device, quantum_device=None):
        """Place classical backbone and quantum head on separate devices."""
        self.classical_device = torch.device(classical_device)
        self.quantum_device = torch.device(quantum_device or classical_device)
        for module in (
            self.encoder,
            self.transformer,
            self.compression,
            self.feature_proj,
            self.classical_head,
        ):
            if module is not None:
                module.to(self.classical_device)
        if self.fusion_logit is not None:
            self.fusion_logit.data = self.fusion_logit.data.to(self.classical_device)
        if self.use_quantum:
            self.head.to(self.quantum_device)

    def set_backbone_eval_mode(self, frozen: bool):
        """Keep BatchNorm/Dropout stable when backbone is frozen (Stage B)."""
        self._backbone_frozen = frozen
        modules = (self.encoder, self.transformer, self.compression, self.feature_proj)
        for module in modules:
            if module is None:
                continue
            if frozen:
                module.eval()
            else:
                module.train()

    def forward_features(
        self,
        images: torch.Tensor,
        modality_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        backbone_ctx = torch.no_grad() if self._backbone_frozen else nullcontext()

        with backbone_ctx:
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
        """Stage A: classical; Stage B: VQC; Stage C: fused (E4) or VQC (E3)."""
        if not self.use_quantum:
            self._use_classical_head = True
            self._fusion_active = False
            return
        if stage == "stage_a":
            self._use_classical_head = True
            self._fusion_active = False
        elif stage == "stage_b":
            self._use_classical_head = False
            self._fusion_active = False
        else:
            # stage_c / eval: fuse when E4, else VQC-only (E3)
            self._use_classical_head = False
            self._fusion_active = self.use_fusion

    def get_fusion_alpha(self) -> float:
        """Current classical mixing weight in [0, 1] (1 = all classical)."""
        if not self.use_fusion:
            return 0.0
        if self._fusion_fixed_alpha is not None:
            return float(self._fusion_fixed_alpha)
        if self.fusion_logit is None:
            return 0.5
        return float(torch.sigmoid(self.fusion_logit.detach()).cpu())

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

    def set_fusion_trainable(self, trainable: bool):
        if self.fusion_logit is not None:
            self.fusion_logit.requires_grad = trainable

    def forward_from_features(self, compressed: torch.Tensor) -> torch.Tensor:
        """Classify pre-compressed features (Stage B feature cache path)."""
        x_c = compressed.to(self.classical_device)

        if self.use_quantum and self._use_classical_head:
            return self.classical_head(x_c)

        if self.use_quantum and self._fusion_active and self.use_fusion:
            logits_c = self.classical_head(x_c)
            logits_q = self.head(compressed.to(self.quantum_device)).to(
                self.classical_device
            )
            if self._fusion_fixed_alpha is not None:
                alpha = torch.tensor(
                    self._fusion_fixed_alpha, device=logits_c.device, dtype=logits_c.dtype
                )
            else:
                alpha = torch.sigmoid(self.fusion_logit).to(
                    device=logits_c.device, dtype=logits_c.dtype
                )
            return alpha * logits_c + (1.0 - alpha) * logits_q

        if self.use_quantum:
            logits = self.head(compressed.to(self.quantum_device))
            return logits.to(self.classical_device)
        return self.head(x_c)

    def forward(
        self,
        images: torch.Tensor,
        modality_ids: torch.Tensor,
        return_attention: bool = False,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is not None:
            return self.forward_from_features(features)

        images = images.to(self.classical_device)
        modality_ids = modality_ids.to(self.classical_device)
        compressed = self.forward_features(images, modality_ids, return_attention)
        return self.forward_from_features(compressed)


class ClassicalBreastCancerModel(HybridBreastCancerModel):
    """Unified classical model without quantum head."""

    def __init__(self, **kwargs):
        kwargs["use_quantum"] = False
        kwargs["use_fusion"] = False
        super().__init__(**kwargs)
