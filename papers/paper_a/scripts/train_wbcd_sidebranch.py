#!/usr/bin/env python3
"""Paper A: hybrids that are not histopath / Mari transfer learning.

Histopath and the old WBCD hybrid both do: classical net compresses → VQC is
the classifier. That two-stage recipe lost on WBCD.

This script keeps the native 30-D clinical vector as the main path and uses
quantum only as extra features or a side branch:

  QFEAT_SVM     30-D + 7 fixed quantum expectations → SVM-RBF
  CLASS_NL_SVM  30-D + 7 classical nonlinear extras (matched width) → SVM-RBF
  SIDEBRANCH    MLP(30-D) hidden concatenated with VQC(PCA-8) → linear head
  SVM_RBF       30-D baseline (do not drop)

Same folds, train-fold-only scaling/PCA, malignant = positive.
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


def qml_device(n: int):
    try:
        return qml.device("lightning.qubit", wires=n)
    except Exception:
        return qml.device("default.qubit", wires=n)


def make_qfeat_fn(n_qubits: int = 4):
    """Fixed (untrained) feature map: angle-Y + ring CNOT, Z and adjacent ZZ."""
    dev = qml_device(n_qubits)
    wires = list(range(n_qubits))

    obs = [qml.PauliZ(w) for w in wires]
    obs += [qml.PauliZ(wires[i]) @ qml.PauliZ(wires[i + 1]) for i in range(n_qubits - 1)]

    @qml.qnode(dev)
    def qfeat(x):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[-1], wires[0]])
        return [qml.expval(o) for o in obs]

    return qfeat, len(obs)


def embed_qfeat(fn, x: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(fn(row), dtype=np.float64) for row in x])


def classical_nonlinear_extras(pca4: np.ndarray) -> np.ndarray:
    """Width-matched classical extras: sin of 4 PCs + 3 adjacent products."""
    s = np.sin(pca4)
    prod = np.stack(
        [pca4[:, i] * pca4[:, i + 1] for i in range(pca4.shape[1] - 1)], axis=1
    )
    return np.concatenate([s, prod], axis=1)


class SideBranch(nn.Module):
    """Main path: MLP on native 30-D. Side path: VQC on PCA-8. Concatenate."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 2, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(30),
            nn.Linear(30, hidden),
            nn.GELU(),
        )
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})
        self.head = nn.Linear(hidden + n_qubits, 2)

    def forward(self, x30: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x30)
        z = self.qlayer(xq)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return self.head(torch.cat([h, z], dim=-1))


def train_sidebranch(model, x30_tr, xq_tr, y_tr, x30_te, xq_te, epochs, batch, lr, seed):
    set_seeds(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x30 = torch.tensor(x30_tr, dtype=torch.float32)
    xq = torch.tensor(xq_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(x30))
        for start in range(0, len(x30), batch):
            b = idx[start : start + batch]
            opt.zero_grad()
            loss = loss_fn(model(x30[b], xq[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(x30_te, dtype=torch.float32),
            torch.tensor(xq_te, dtype=torch.float32),
        )
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pred, prob[:, 0], npar


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_splits = 2
        args.epochs = 4
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    x, y = data.data, data.target
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    qfeat_fn, n_qfeat = make_qfeat_fn(4)
    rows = []

    print(
        "Paper A side-branch hybrid (not Mari/histopath last-layer VQC). "
        f"qfeat_dim={n_qfeat}",
        flush=True,
    )

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        sc = StandardScaler().fit(x[tr])
        xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
        ytr, yte = y[tr], y[te]
        pca4 = PCA(n_components=4).fit(xtr)
        pca8 = PCA(n_components=8).fit(xtr)
        mm4 = MinMaxScaler((0.0, np.pi)).fit(pca4.transform(xtr))
        mm8 = MinMaxScaler((0.0, np.pi)).fit(pca8.transform(xtr))
        qtr4 = mm4.transform(pca4.transform(xtr))
        qte4 = mm4.transform(pca4.transform(xte))
        qtr8 = mm8.transform(pca8.transform(xtr))
        qte8 = mm8.transform(pca8.transform(xte))
        print(f"\n=== fold {fold} n_tr={len(tr)} n_te={len(te)} ===", flush=True)

        t0 = time.time()
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=args.seed + fold)
        svm.fit(xtr, ytr)
        rows.append(
            rec_row(
                "SVM_RBF",
                yte,
                svm.predict(xte),
                p_mal_from_dec(svm.decision_function(xte)),
                fold,
                t0,
            )
        )
        print(
            f"  {'SVM_RBF':16s} bal={rows[-1]['balanced_accuracy']:.3f} "
            f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}",
            flush=True,
        )

        t0 = time.time()
        qtr_f, qte_f = embed_qfeat(qfeat_fn, qtr4), embed_qfeat(qfeat_fn, qte4)
        xtr_q = np.concatenate([xtr, qtr_f], axis=1)
        xte_q = np.concatenate([xte, qte_f], axis=1)
        svm_q = SVC(kernel="rbf", C=10, gamma="scale", random_state=args.seed + fold)
        svm_q.fit(xtr_q, ytr)
        rows.append(
            rec_row(
                "QFEAT_SVM",
                yte,
                svm_q.predict(xte_q),
                p_mal_from_dec(svm_q.decision_function(xte_q)),
                fold,
                t0,
                n_qfeat=n_qfeat,
            )
        )
        print(
            f"  {'QFEAT_SVM':16s} bal={rows[-1]['balanced_accuracy']:.3f} "
            f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}",
            flush=True,
        )

        t0 = time.time()
        ctr = classical_nonlinear_extras(pca4.transform(xtr))
        cte = classical_nonlinear_extras(pca4.transform(xte))
        xtr_c = np.concatenate([xtr, ctr], axis=1)
        xte_c = np.concatenate([xte, cte], axis=1)
        svm_c = SVC(kernel="rbf", C=10, gamma="scale", random_state=args.seed + fold)
        svm_c.fit(xtr_c, ytr)
        rows.append(
            rec_row(
                "CLASS_NL_SVM",
                yte,
                svm_c.predict(xte_c),
                p_mal_from_dec(svm_c.decision_function(xte_c)),
                fold,
                t0,
            )
        )
        print(
            f"  {'CLASS_NL_SVM':16s} bal={rows[-1]['balanced_accuracy']:.3f} "
            f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}",
            flush=True,
        )

        t0 = time.time()
        model = SideBranch()
        pred, p_mal, npar = train_sidebranch(
            model,
            xtr,
            qtr8,
            ytr,
            xte,
            qte8,
            args.epochs,
            32,
            1e-3,
            args.seed + fold,
        )
        rows.append(
            rec_row("SIDEBRANCH", yte, pred, p_mal, fold, t0, n_params=npar)
        )
        print(
            f"  {'SIDEBRANCH':16s} bal={rows[-1]['balanced_accuracy']:.3f} "
            f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    summary = {}
    for name, g in df.groupby("model"):
        summary[name] = {
            m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1) if len(g) > 1 else 0.0)}
            for m in [
                "accuracy",
                "balanced_accuracy",
                "precision_malignant",
                "recall_malignant",
                "f1_malignant",
                "auc",
            ]
        }
    payload = {
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "paper_a_tabular_wbcd",
        "sweep": "sidebranch_early_fusion",
        "positive_class": "malignant (label 0)",
        "quick": bool(args.quick),
        "note": (
            "Not Mari/histopath: quantum is extra features or a side branch; "
            "the 30-D clinical vector is never discarded."
        ),
        "summary": summary,
    }
    df.to_csv(OUT / "sidebranch_folds.csv", index=False)
    (OUT / "sidebranch.json").write_text(json.dumps(payload, indent=2))
    print("\n=== mean ± std balanced accuracy ===", flush=True)
    for name, s in sorted(
        summary.items(), key=lambda kv: -kv[1]["balanced_accuracy"]["mean"]
    ):
        print(
            f"  {name:16s} {s['balanced_accuracy']['mean']:.3f}±"
            f"{s['balanced_accuracy']['std']:.3f}  rec_mal="
            f"{s['recall_malignant']['mean']:.3f}",
            flush=True,
        )
    print(f"Wrote {OUT / 'sidebranch.json'}", flush=True)


if __name__ == "__main__":
    main()
