#!/usr/bin/env python3
"""Generate VQC circuit diagram text for paper supplementary."""

import sys
from pathlib import Path

import numpy as np
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]

N_QUBITS = 8
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RY(weights[layer, q, 0], wires=q)
            qml.RZ(weights[layer, q, 1], wires=q)
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
    return [qml.expval(qml.PauliZ(q)) for q in range(2)]


x = np.zeros(N_QUBITS)
w = np.zeros((N_LAYERS, N_QUBITS, 2))
_ = circuit(x, w)

out_path = ROOT / "figures" / "vqc_circuit_diagram.txt"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    f.write(qml.draw(circuit)(x, w))
print(f"Circuit diagram saved to {out_path}")
