"""Variational Quantum Circuit head — Benedetti et al. (2019) aligned design.

Expressivity ablations (CLI / config):
- entanglement: none | linear | circular
- n_layers: ansatz depth
- encoding: angle_x | angle_y | angle_z
- data_reuploading: encode once (False) or before each layer (True)
"""

import math

import pennylane as qml
import torch
import torch.nn as nn

ENCODING_ROTATION = {
    "angle_x": "X",
    "angle_y": "Y",
    "angle_z": "Z",
    # Aliases
    "x": "X",
    "y": "Y",
    "z": "Z",
}


class AngleEncoder(nn.Module):
    """Map normalized classical features to [0, pi] rotation angles."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * math.pi


def _apply_entanglement(n_qubits: int, entanglement: str):
    if entanglement in (None, "none", "no", "off"):
        return
    if entanglement == "linear":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        return
    if entanglement == "circular":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
        return
    raise ValueError(
        f"Unknown entanglement={entanglement!r}. "
        "Use none|linear|circular."
    )


def _apply_encoding(inputs, n_qubits: int, encoding: str):
    key = encoding.lower()
    if key not in ENCODING_ROTATION:
        raise ValueError(
            f"Unknown encoding={encoding!r}. "
            f"Use one of: {sorted(set(ENCODING_ROTATION))}."
        )
    qml.AngleEmbedding(
        inputs, wires=range(n_qubits), rotation=ENCODING_ROTATION[key]
    )


def _variational_layer(weights_layer, n_qubits: int, entanglement: str):
    for q in range(n_qubits):
        qml.RY(weights_layer[q, 0], wires=q)
        qml.RZ(weights_layer[q, 1], wires=q)
    _apply_entanglement(n_qubits, entanglement)


def _circuit_definition(
    inputs,
    weights,
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    encoding: str = "angle_y",
    data_reuploading: bool = False,
):
    """Hardware-efficient ansatz with optional data re-uploading.

    Standard (data_reuploading=False):
        encode once -> [RY/RZ + entangle] x n_layers

    Re-uploading (data_reuploading=True; Pérez-Salinas et al.):
        for each layer: encode -> RY/RZ + entangle
    """
    if data_reuploading:
        for layer in range(n_layers):
            _apply_encoding(inputs, n_qubits, encoding)
            _variational_layer(weights[layer], n_qubits, entanglement)
    else:
        _apply_encoding(inputs, n_qubits, encoding)
        for layer in range(n_layers):
            _variational_layer(weights[layer], n_qubits, entanglement)
    return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]


def build_vqc_layer(
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
    encoding: str = "angle_y",
    data_reuploading: bool = False,
    backend: str | None = None,
    diff_method: str | None = None,
):
    """Hardware-efficient variational circuit (RY, RZ, optional CNOT).

    Backend order matters for speed. Benchmarked at 8 qubits, batch 64:
    default.qubit+backprop is ~7x faster than lightning.qubit+adjoint
    because it vectorizes the batch dimension, whereas lightning's adjoint
    path loops sample-by-sample. Prefer default.qubit for batched training;
    override via config for large-qubit or single-sample regimes.
    """

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    attempts: list[tuple[str, str]] = []

    if backend is not None:
        attempts.append((backend, diff_method or "backprop"))
    else:
        # Vectorized batch path first (fastest for training small circuits).
        attempts.append(("default.qubit", "backprop"))
        try:
            import pennylane_lightning  # noqa: F401

            attempts.append(("lightning.qubit", "adjoint"))
        except ImportError:
            pass

    last_error: Exception | None = None
    for dev_name, diff in attempts:
        try:
            dev = qml.device(dev_name, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff)
            def circuit(inputs, weights):
                return _circuit_definition(
                    inputs,
                    weights,
                    n_qubits,
                    n_layers,
                    entanglement,
                    encoding=encoding,
                    data_reuploading=data_reuploading,
                )

            layer = qml.qnn.TorchLayer(circuit, weight_shapes)
            print(
                f"VQC backend: {dev_name} ({diff}) | "
                f"enc={encoding} ent={entanglement} "
                f"L={n_layers} reup={data_reuploading}"
            )
            return layer
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Could not initialize a PennyLane VQC backend. "
        "Install pennylane-lightning or use default.qubit."
    ) from last_error


class VQCHead(nn.Module):
    """
    Quantum head with:
    - LayerNorm on compressed features (classical pre-encoding)
    - Angle encoding (axis configurable)
    - Shallow hardware-efficient ansatz (+ optional data re-uploading)
    - Full qubit readout + linear classifier
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        num_classes: int = 2,
        entanglement: str = "linear",
        encoding: str = "angle_y",
        data_reuploading: bool = False,
        feature_norm: bool = True,
        full_readout: bool = True,
        backend: str | None = None,
        diff_method: str | None = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.data_reuploading = data_reuploading
        self.entanglement = entanglement
        self.n_layers = n_layers
        self.feature_norm = nn.LayerNorm(n_qubits) if feature_norm else nn.Identity()
        self.angle_encoder = AngleEncoder()
        self.quantum_layer = build_vqc_layer(
            n_qubits,
            n_layers,
            entanglement,
            encoding=encoding,
            data_reuploading=data_reuploading,
            backend=backend,
            diff_method=diff_method,
        )
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
