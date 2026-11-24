#!/usr/bin/env python3
"""Paper A: morphology-block VQC + kernel target alignment (KTA).

Novelty vs PCA-blind QML on WBCD:
  - 10 qubits align with the 10 Wisconsin morphometry attributes; each wire
    encodes (mean, SE, worst) via sequential RY/RZ/RX (indices i, i+10, i+20).
  - KTA (centered kernel-target alignment) on the train fold predicts which
    fidelity kernels are separable before fitting an QSVM.

Same 5-fold seed 42, malignant = positive, train-fold-only scaling.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "papers/paper_a/results"
MALIGNANT = 0
N_ATTR = 10


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def malignant_metrics(y_true, y_pred, p_mal) -> dict:
    y_mal = (y_true == MALIGNANT).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_malignant": float(
            precision_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)
        ),
        "recall_malignant": float(
            recall_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)
        ),
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)),
        "auc": float(roc_auc_score(y_mal, p_mal)),
    }


def rec_row(name, y_te, pred, p_mal, fold, t0, **extra) -> dict:
    out = malignant_metrics(y_te, pred, p_mal)
    out.update(model=name, fold=fold, train_s=round(time.time() - t0, 3), **extra)
    return out


def p_mal_from_dec(scores) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.asarray(scores, dtype=float)))


def center_kernel(k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def kernel_target_alignment(k: np.ndarray, y: np.ndarray) -> float:
    """Centered alignment between a kernel and ideal same-class structure."""
    y_signed = np.where(y == MALIGNANT, 1.0, -1.0)
    t = np.outer(y_signed, y_signed)
    kc = center_kernel(k)
    tc = center_kernel(t)
    num = float(np.sum(kc * tc))
    den = float(np.linalg.norm(kc) * np.linalg.norm(tc))
    return num / den if den > 1e-12 else 0.0


def qml_device(n: int):
    try:
        return qml.device("lightning.qubit", wires=n)
    except Exception:
        return qml.device("default.qubit", wires=n)


def encode_z(x, wires, reps=2):
    for _ in range(reps):
        for i, w in enumerate(wires):
            qml.Hadamard(wires=w)
            qml.RZ(2.0 * x[i], wires=w)


def encode_zz(x, wires, reps=2):
    n = len(wires)
    for _ in range(reps):
        for i, w in enumerate(wires):
            qml.Hadamard(wires=w)
            qml.RZ(2.0 * x[i], wires=w)
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.RZ(2.0 * x[i] * x[i + 1], wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def encode_angle(x, wires, reps=1):
    for _ in range(reps):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")


def fidelity_gram(x_tr_q: np.ndarray, x_te_q: np.ndarray, n_wires: int, map_name: str):
    dev = qml_device(n_wires)
    wires = list(range(n_wires))
    enc = {"z": encode_z, "zz": encode_zz, "angle": encode_angle}[map_name]

    @qml.qnode(dev)
    def psi(x):
        enc(x, wires)
        return qml.state()

    def row(i):
        return psi(x_tr_q[i])

    states_tr = [row(i) for i in range(len(x_tr_q))]
    k_tr = np.zeros((len(x_tr_q), len(x_tr_q)))
    for i in range(len(states_tr)):
        for j in range(i, len(states_tr)):
            val = float(np.abs(np.vdot(states_tr[i], states_tr[j])) ** 2)
            k_tr[i, j] = k_tr[j, i] = val

    states_te = [psi(x_te_q[i]) for i in range(len(x_te_q))]
    k_te = np.zeros((len(x_te_q), len(x_tr_q)))
    for i, st in enumerate(states_te):
        for j, st_tr in enumerate(states_tr):
            k_te[i, j] = float(np.abs(np.vdot(st, st_tr)) ** 2)
    return k_tr, k_te


class MorphVQCHead(nn.Module):
    """10-qubit block encoding: wire i gets mean/error/worst for attribute i."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        n_qubits = N_ATTR
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.05)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, weights):
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
                qml.RZ(x[q + N_ATTR], wires=q)
                qml.RX(x[q + 2 * N_ATTR], wires=q)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.circuit(xi, self.weights) for xi in x])


class MatchedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train_morph_vqc(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seeds(seed)
    model = MorphVQCHead(n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    target = torch.tensor(2.0 * y_tr - 1.0, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(xt), batch_size, rng):
            opt.zero_grad()
            z = model(xt[b])
            loss = ((z - target[b]) ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x_te, dtype=torch.float32)).cpu().numpy()
    pred = (z > 0.0).astype(int)
    p_mal = (1.0 - z) / 2.0
    return pred, p_mal


def train_matched_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seeds(seed)
    model = MatchedMLP(x_tr.shape[1], hidden=10)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(xt), batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_te, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    return pred, prob[:, 0]


def scale_morph_angles(x_tr_s: np.ndarray, x_te_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MinMax each of 30 columns to [0, pi] on train only (morphology-native)."""
    mm = MinMaxScaler(feature_range=(0.0, np.pi))
    return mm.fit_transform(x_tr_s), mm.transform(x_te_s)


def scale_pca4_angles(x_tr_s: np.ndarray, x_te_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=4, random_state=0)
    tr = pca.fit_transform(x_tr_s)
    te = pca.transform(x_te_s)
    mm = MinMaxScaler(feature_range=(0.0, np.pi))
    return mm.fit_transform(tr), mm.transform(te)


def summarize(rows: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    out = {}
    for name, rs in by_model.items():
        keys = [k for k in rs[0] if k not in ("model", "fold", "train_s")]
        out[name] = {
            k: {
                "mean": float(np.mean([r[k] for r in rs])),
                "std": float(np.std([r[k] for r in rs], ddof=1)),
            }
            for k in keys
            if isinstance(rs[0][k], (int, float))
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--vqc-epochs", type=int, default=30)
    p.add_argument("--mlp-epochs", type=int, default=80)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="2 folds only")
    args = p.parse_args()

    data = load_breast_cancer()
    x = data.data.astype(np.float64)
    y = data.target.astype(int)
    n_splits = 2 if args.quick else args.n_splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    rows: list[dict] = []
    kta_rows: list[dict] = []

    print(f"WBCD morph-block VQC + KTA | {n_splits} folds | seed={args.seed}")
    for fold, (tr, te) in enumerate(cv.split(x, y)):
        t0 = time.time()
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x[tr])
        x_te_s = scaler.transform(x[te])
        y_tr, y_te = y[tr], y[te]

        x_tr_morph, x_te_morph = scale_morph_angles(x_tr_s, x_te_s)
        x_tr_q4, x_te_q4 = scale_pca4_angles(x_tr_s, x_te_s)

        for map_name in ("zz", "angle"):
            k_tr, k_te = fidelity_gram(x_tr_q4, x_te_q4, 4, map_name)
            kta = kernel_target_alignment(k_tr, y_tr)
            kta_rows.append(
                {
                    "fold": fold,
                    "kernel": map_name,
                    "kta_train": kta,
                }
            )
            svc = SVC(C=10.0, kernel="precomputed")
            svc.fit(k_tr, y_tr)
            pred = svc.predict(k_te)
            dec = svc.decision_function(k_te)
            label = f"QSVM_{map_name.upper()}4"
            if map_name == "zz":
                label = "QSVM_ZZ4"
            else:
                label = "QSVM_ANGLE4"
            rows.append(
                rec_row(
                    label,
                    y_te,
                    pred,
                    p_mal_from_dec(dec),
                    fold,
                    t0,
                    kta_train=kta,
                )
            )

        pred, p_mal = train_morph_vqc(
            x_tr_morph,
            y_tr,
            x_te_morph,
            n_layers=args.n_layers,
            epochs=args.vqc_epochs,
            batch_size=32,
            lr=0.05,
            seed=args.seed + fold,
        )
        rows.append(rec_row("VQC_MORPH10", y_te, pred, p_mal, fold, t0))

        pred, p_mal = train_matched_mlp(
            x_tr_morph,
            y_tr,
            x_te_morph,
            epochs=args.mlp_epochs,
            batch_size=32,
            lr=1e-3,
            seed=args.seed + fold,
        )
        rows.append(rec_row("MLP_MORPH30", y_te, pred, p_mal, fold, t0))

        svc = SVC(C=10.0, kernel="rbf", gamma="scale")
        svc.fit(x_tr_s, y_tr)
        pred = svc.predict(x_te_s)
        rows.append(
            rec_row(
                "SVM_RBF",
                y_te,
                pred,
                p_mal_from_dec(svc.decision_function(x_te_s)),
                fold,
                t0,
            )
        )
        print(
            f"  fold {fold}: VQC_MORPH10 bal={rows[-3]['balanced_accuracy']:.3f} "
            f"MLP_MORPH={rows[-2]['balanced_accuracy']:.3f} "
            f"KTA zz={kta_rows[-2]['kta_train']:.3f} ang={kta_rows[-1]['kta_train']:.3f}"
        )

    summary = summarize(rows)
    kta_by_kernel: dict[str, list[float]] = {}
    kta_bal: dict[str, list[tuple[float, float]]] = {"zz": [], "angle": []}
    for r in rows:
        if r.get("kta_train") is not None:
            kta_bal[r["model"].replace("QSVM_", "").replace("4", "").lower()].append(
                (r["kta_train"], r["balanced_accuracy"])
            )

    payload = {
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "paper_a_tabular_wbcd",
        "sweep": "morph_block_vqc_and_kta",
        "positive_class": "malignant (label 0)",
        "note": (
            "VQC_MORPH10: qubit i encodes mean/error/worst for attribute i. "
            "KTA computed on train Gram only."
        ),
        "summary": summary,
        "kta_folds": kta_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "morph_kta.json"
    out_csv = OUT / "morph_kta_folds.csv"
    out_json.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_json}")
    for name, stats in summary.items():
        ba = stats["balanced_accuracy"]
        print(f"  {name:18s} bal_acc {ba['mean']:.3f} +/- {ba['std']:.3f}")


if __name__ == "__main__":
    main()