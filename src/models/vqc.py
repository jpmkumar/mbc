"""Variational Quantum Circuit head — Benedetti et al. (2019) aligned design."""

import math

import pennylane as qml
import torch
import torch.nn as nn


class AngleEncoder(nn.Module):
    """Map normalized classical features to [0, pi] rotation angles."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * math.pi


def build_vqc_layer(n_qubits: int, n_layers: int, entanglement: str = "linear"):
    """Hardware-efficient variational circuit (RY, RZ, linear CNOT)."""

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            if entanglement == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            elif entanglement == "circular":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

        # Full readout: all qubit Z expectations (Benedetti post-measurement layer)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class VQCHead(nn.Module):
    """
    Quantum head with:
    - LayerNorm on compressed features (classical pre-encoding)
    - Angle encoding
    - Shallow hardware-efficient ansatz
    - Full qubit readout + linear classifier
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        num_classes: int = 2,
        entanglement: str = "linear",
        feature_norm: bool = True,
        full_readout: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_norm = nn.LayerNorm(n_qubits) if feature_norm else nn.Identity()
        self.angle_encoder = AngleEncoder()
        self.quantum_layer = build_vqc_layer(n_qubits, n_layers, entanglement)
        readout_dim = n_qubits if full_readout else min(n_qubits, 2)
        self.classifier = nn.Linear(readout_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.n_qubits:
            x = x[..., : self.n_qubits]
        x = self.feature_norm(x)
        angles = self.angle_encoder(x)
        q_out = self.quantum_layer(angles)
        if q_out.dim() == 1:
            q_out = q_out.unsqueeze(0)
        if q_out.size(-1) > self.classifier.in_features:
            q_out = q_out[..., : self.classifier.in_features]
        return self.classifier(q_out)
