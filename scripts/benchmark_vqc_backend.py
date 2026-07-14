#!/usr/bin/env python3
"""Benchmark VQC backends for batched training.

Compares default.qubit (backprop) vs lightning.qubit (adjoint) on a
training-style forward+backward step. At small qubit counts default.qubit
vectorizes the batch dimension and is typically several times faster;
lightning's adjoint path loops sample-by-sample. Run once per environment
(ideally on the target GPU) to pick the fastest backend for configs.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pennylane as qml  # noqa: E402

from src.models.vqc import _circuit_definition  # noqa: E402


def build_layer(backend: str, diff_method: str, n_qubits: int, n_layers: int):
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        return _circuit_definition(inputs, weights, n_qubits, n_layers, "linear")

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 2)})


def bench(backend, diff_method, n_qubits, n_layers, batch, iters):
    layer = build_layer(backend, diff_method, n_qubits, n_layers)
    head = torch.nn.Sequential(layer, torch.nn.Linear(n_qubits, 2))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    x = torch.rand(batch, n_qubits)
    y = torch.randint(0, 2, (batch,))
    for _ in range(3):  # warmup
        opt.zero_grad()
        lossf(head(x), y).backward()
        opt.step()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        lossf(head(x), y).backward()
        opt.step()
    return (time.time() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()

    candidates = [("default.qubit", "backprop")]
    try:
        import pennylane_lightning  # noqa: F401

        candidates.append(("lightning.qubit", "adjoint"))
    except ImportError:
        print("lightning.qubit not installed; benchmarking default.qubit only")

    print(
        f"VQC backend benchmark | qubits={args.n_qubits} layers={args.n_layers} "
        f"batch={args.batch} iters={args.iters}"
    )
    results = {}
    for backend, diff in candidates:
        try:
            ms = bench(
                backend, diff, args.n_qubits, args.n_layers, args.batch, args.iters
            )
            results[(backend, diff)] = ms
            print(f"  {backend:18s} {diff:9s}: {ms:8.1f} ms / train-step")
        except Exception as exc:
            print(f"  {backend:18s} {diff:9s}: FAILED ({exc})")

    if results:
        best = min(results, key=results.get)
        print(f"\nFastest: {best[0]} ({best[1]}) -> set in configs/histopath.yaml")


if __name__ == "__main__":
    main()
