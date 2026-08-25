#!/usr/bin/env python3
"""Sweep hybrid classical–quantum heads on Paper A confirmatory folds."""

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
from sklearn.neural_network import MLPClassifier
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


def last_hidden(mlp: MLPClassifier, x: np.ndarray) -> np.ndarray:
    h = x
    for i in range(len(mlp.coefs_) - 1):
        h = np.maximum(h @ mlp.coefs_[i] + mlp.intercepts_[i], 0.0)
    return h


class VQCFull(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})
        self.readout = nn.Linear(n_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.qlayer(x)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return self.readout(z)


class HybridE2E(nn.Module):
    """30-D MLP encoder → (0,π) → VQC full readout. Joint CE training."""

    def __init__(self, in_dim: int, n_qubits: int, n_layers: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_qubits),
        )
        self.vqc = VQCFull(n_qubits, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.encoder(x)) * np.pi
        return self.vqc(z)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, n_qubits: int, n_layers: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.compress = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, n_qubits),
        )
        self.vqc = VQCFull(n_qubits, n_layers)
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit_alpha)
        q = torch.sigmoid(self.compress(x)) * np.pi
        return alpha * self.mlp(x) + (1.0 - alpha) * self.vqc(q)


def train_torch(model, x_tr, y_tr, x_te, epochs, batch, lr, seed):
    set_seeds(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(xt))
        for start in range(0, len(xt), batch):
            b = idx[start : start + batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_te, dtype=torch.float32))
        if isinstance(model, FusionHead):
            alpha = float(torch.sigmoid(model.logit_alpha).item())
        else:
            alpha = None
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pred, prob[:, 0], npar, alpha


def encode_angle(x, wires, reps=1):
    for _ in range(reps):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")


def qsvm_angle(x_tr, y_tr, x_te, n_qubits: int, c: float = 10.0):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def state(row):
        encode_angle(row, wires, 1)
        return qml.state()

    psi_tr = np.stack([np.asarray(state(r), dtype=np.complex128) for r in x_tr])
    psi_te = np.stack([np.asarray(state(r), dtype=np.complex128) for r in x_te])
    k_tr = np.real(np.abs(psi_tr.conj() @ psi_tr.T) ** 2)
    k_te = np.real(np.abs(psi_te.conj() @ psi_tr.T) ** 2)
    clf = SVC(kernel="precomputed", C=c)
    clf.fit(k_tr, y_tr)
    return clf.predict(k_te), p_mal_from_dec(clf.decision_function(k_te))


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
        args.epochs = 5
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    x, y = data.data, data.target
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        sc = StandardScaler().fit(x[tr])
        xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
        ytr, yte = y[tr], y[te]
        pca = PCA(n_components=8).fit(xtr)
        qtr = MinMaxScaler((0, np.pi)).fit_transform(pca.transform(xtr))
        qte = MinMaxScaler((0, np.pi)).fit(pca.transform(xtr)).transform(pca.transform(xte))
        print(f"\n=== fold {fold} n_tr={len(tr)} n_te={len(te)} ===")

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
        print(f"  {'SVM_RBF':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

        t0 = time.time()
        mlp = nn.Sequential(
            nn.LayerNorm(30), nn.Linear(30, 32), nn.GELU(), nn.Linear(32, 2)
        )
        pred, p_mal, npar, _ = train_torch(
            mlp, xtr, ytr, xte, args.epochs, 32, 1e-3, args.seed + fold
        )
        rows.append(rec_row("MLP_30D", yte, pred, p_mal, fold, t0, n_params=npar))
        print(f"  {'MLP_30D':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

        t0 = time.time()
        ext = MLPClassifier(
            hidden_layer_sizes=(32, 8),
            activation="relu",
            solver="adam",
            max_iter=200 if args.quick else 500,
            random_state=args.seed + fold,
            early_stopping=True,
            validation_fraction=0.15,
        )
        ext.fit(xtr, ytr)
        htr, hte = last_hidden(ext, xtr), last_hidden(ext, xte)
        mm = MinMaxScaler((0, np.pi)).fit(htr)
        htr_q, hte_q = mm.transform(htr), mm.transform(hte)
        ext_pred = ext.predict(xte)
        ext_p = ext.predict_proba(xte)[:, list(ext.classes_).index(MALIGNANT)]
        rows.append(rec_row("MLP_extractor", yte, ext_pred, ext_p, fold, t0))
        print(f"  {'MLP_extractor':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

        t0 = time.time()
        vqc = VQCFull(8, 2)
        pred, p_mal, npar, _ = train_torch(
            vqc, htr_q, ytr, hte_q, args.epochs, 32, 0.05, args.seed + fold
        )
        rows.append(
            rec_row("HYBRID_2STAGE_FULL", yte, pred, p_mal, fold, t0, n_params=npar)
        )
        print(f"  {'HYBRID_2STAGE_FULL':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

        t0 = time.time()
        pred, p_mal = qsvm_angle(htr_q, ytr, hte_q, 8)
        rows.append(rec_row("HYBRID_QSVM_ANGLE", yte, pred, p_mal, fold, t0))
        print(f"  {'HYBRID_QSVM_ANGLE':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

        t0 = time.time()
        e2e = HybridE2E(30, 8, 2)
        pred, p_mal, npar, _ = train_torch(
            e2e, xtr, ytr, xte, args.epochs, 32, 1e-3, args.seed + fold
        )
        rows.append(rec_row("HYBRID_E2E", yte, pred, p_mal, fold, t0, n_params=npar))
        print(f"  {'HYBRID_E2E':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

        t0 = time.time()
        fus = FusionHead(30, 8, 2)
        pred, p_mal, npar, alpha = train_torch(
            fus, xtr, ytr, xte, args.epochs, 32, 1e-3, args.seed + fold
        )
        rows.append(
            rec_row(
                "HYBRID_FUSION",
                yte,
                pred,
                p_mal,
                fold,
                t0,
                n_params=npar,
                fusion_alpha=alpha,
            )
        )
        print(f"  {'HYBRID_FUSION':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f} "
              f"alpha={alpha:.3f}")

        t0 = time.time()
        vqc8 = VQCFull(8, 2)
        pred, p_mal, npar, _ = train_torch(
            vqc8, qtr, ytr, qte, args.epochs, 32, 0.05, args.seed + fold
        )
        rows.append(rec_row("VQC_FULL8", yte, pred, p_mal, fold, t0, n_params=npar))
        print(f"  {'VQC_FULL8':20s} bal={rows[-1]['balanced_accuracy']:.3f} "
              f"rec={rows[-1]['recall_malignant']:.3f} auc={rows[-1]['auc']:.3f}")

    df = pd.DataFrame(rows)
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision_malignant",
        "recall_malignant",
        "f1_malignant",
        "auc",
    ]
    summary = {}
    for model, g in df.groupby("model"):
        summary[model] = {
            m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1))}
            for m in metrics
        }
        if "fusion_alpha" in g:
            summary[model]["mean_fusion_alpha"] = float(g["fusion_alpha"].mean())
    payload = {
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper": "paper_a_tabular_wbcd",
        "sweep": "hybrid_algorithms",
        "positive_class": "malignant (label 0)",
        "quick": bool(args.quick),
        "summary": summary,
    }
    out = OUT / "hybrid_sweep.json"
    df.to_csv(OUT / "hybrid_sweep_folds.csv", index=False)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n======== hybrid sweep mean ± std ========")
    for model in sorted(
        summary, key=lambda m: summary[m]["balanced_accuracy"]["mean"], reverse=True
    ):
        s = summary[model]
        extra = ""
        if "mean_fusion_alpha" in s:
            extra = f"  alpha={s['mean_fusion_alpha']:.3f}"
        print(
            f"{model:20s}  bal {s['balanced_accuracy']['mean']:.3f}±"
            f"{s['balanced_accuracy']['std']:.3f}  "
            f"rec {s['recall_malignant']['mean']:.3f}±"
            f"{s['recall_malignant']['std']:.3f}  "
            f"auc {s['auc']['mean']:.3f}±{s['auc']['std']:.3f}{extra}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
