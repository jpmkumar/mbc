#!/usr/bin/env python3
"""Sweep standard quantum classifiers on the Paper A confirmatory folds.

Same leakage-free 5-fold protocol and malignant-positive metrics as
train_wbcd_cv.py. Classical SVM/MLP stay in the table.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "papers/paper_a/results"
MALIGNANT = 0


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def device(n: int):
    try:
        return qml.device("lightning.qubit", wires=n)
    except Exception:
        return qml.device("default.qubit", wires=n)


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


def p_mal_from_decision(scores) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.asarray(scores, dtype=float)))


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


def encode_iqp(x, wires, reps=1):
    n = len(wires)
    for _ in range(reps):
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.RZ(2.0 * x[i], wires=w)
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(2.0 * x[i] * x[j], wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


def state_fn(n: int, encoder, reps: int):
    dev = device(n)
    wires = list(range(n))

    @qml.qnode(dev)
    def _state(x):
        encoder(x, wires, reps)
        return qml.state()

    return _state


def states(fn, x: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(fn(row), dtype=np.complex128) for row in x])


def fidelity_kernel(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    return np.real(np.abs(psi_a.conj() @ psi_b.T) ** 2)


def fit_precomputed_svm(k_tr, y_tr, k_te, c: float):
    clf = SVC(kernel="precomputed", C=c)
    clf.fit(k_tr, y_tr)
    pred = clf.predict(k_te)
    p_mal = p_mal_from_decision(clf.decision_function(k_te))
    return pred, p_mal


def projected_features(n: int, encoder, reps: int, x: np.ndarray) -> np.ndarray:
    dev = device(n)
    wires = list(range(n))

    @qml.qnode(dev)
    def exps(row):
        encoder(row, wires, reps)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return np.stack([np.asarray(exps(row), dtype=np.float64) for row in x])


class VQCReadout(nn.Module):
    """Angle-Y + SEL; all-qubit Pauli-Z readout + linear classifier."""

    def __init__(self, n_qubits: int, n_layers: int, reup: bool):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            for layer in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n_qubits))
                if not reup:
                    break
            if not reup:
                for extra in range(1, n_layers):
                    qml.StronglyEntanglingLayers(
                        weights[extra : extra + 1], wires=range(n_qubits)
                    )
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})
        self.readout = nn.Linear(n_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.qlayer(x)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return self.readout(z)


def train_vqc_ce(x_tr, y_tr, x_te, *, n_qubits, n_layers, reup, epochs, batch, lr, seed):
    set_seeds(seed)
    model = VQCReadout(n_qubits, n_layers, reup=reup)
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
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pred, prob[:, 0], npar


def prepare(x_tr, x_te, n4=4, n8=8):
    sc = StandardScaler().fit(x_tr)
    tr_s, te_s = sc.transform(x_tr), sc.transform(x_te)
    pca4 = PCA(n_components=n4).fit(tr_s)
    pca8 = PCA(n_components=n8).fit(tr_s)
    tr4, te4 = pca4.transform(tr_s), pca4.transform(te_s)
    tr8, te8 = pca8.transform(tr_s), pca8.transform(te_s)
    mm4 = MinMaxScaler((0.0, np.pi)).fit(tr4)
    mm8 = MinMaxScaler((0.0, np.pi)).fit(tr8)
    return {
        "s_tr": tr_s,
        "s_te": te_s,
        "pca4_tr": tr4,
        "pca4_te": te4,
        "q4_tr": mm4.transform(tr4),
        "q4_te": mm4.transform(te4),
        "q8_tr": mm8.transform(tr8),
        "q8_te": mm8.transform(te8),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vqc-epochs", type=int, default=30)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_splits = 2
        args.vqc_epochs = 3
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    x, y = data.data, data.target
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []

    kernel_specs = [
        ("QSVM_Z4", encode_z, 4, 2),
        ("QSVM_ANGLE4", encode_angle, 4, 1),
        ("QSVM_ZZ4", encode_zz, 4, 2),
        ("QSVM_IQP4", encode_iqp, 4, 1),
        ("QSVM_ZZ8", encode_zz, 8, 1),
    ]

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        d = prepare(x[tr], x[te])
        ytr, yte = y[tr], y[te]
        print(f"\n=== fold {fold} n_tr={len(tr)} n_te={len(te)} ===")

        t0 = time.time()
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=args.seed + fold)
        svm.fit(d["s_tr"], ytr)
        pred = svm.predict(d["s_te"])
        rec = rec_row(
            "SVM_RBF",
            yte,
            pred,
            p_mal_from_decision(svm.decision_function(d["s_te"])),
            fold,
            t0,
        )
        rows.append(rec)
        print(f"  {rec['model']:16s} bal={rec['balanced_accuracy']:.3f} "
              f"rec={rec['recall_malignant']:.3f} auc={rec['auc']:.3f}")

        t0 = time.time()
        lin = SVC(kernel="linear", C=10, random_state=args.seed + fold)
        lin.fit(d["pca4_tr"], ytr)
        pred = lin.predict(d["pca4_te"])
        rec = rec_row(
            "SVM_LIN_PCA4",
            yte,
            pred,
            p_mal_from_decision(lin.decision_function(d["pca4_te"])),
            fold,
            t0,
        )
        rows.append(rec)
        print(f"  {rec['model']:16s} bal={rec['balanced_accuracy']:.3f} "
              f"rec={rec['recall_malignant']:.3f} auc={rec['auc']:.3f}")

        for name, enc, nq, reps in kernel_specs:
            t0 = time.time()
            qtr = d["q4_tr"] if nq == 4 else d["q8_tr"]
            qte = d["q4_te"] if nq == 4 else d["q8_te"]
            fn = state_fn(nq, enc, reps)
            psi_tr, psi_te = states(fn, qtr), states(fn, qte)
            pred, p_mal = fit_precomputed_svm(
                fidelity_kernel(psi_tr, psi_tr),
                ytr,
                fidelity_kernel(psi_te, psi_tr),
                c=10.0,
            )
            rec = rec_row(name, yte, pred, p_mal, fold, t0)
            rows.append(rec)
            print(f"  {rec['model']:16s} bal={rec['balanced_accuracy']:.3f} "
                  f"rec={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
                  f"({rec['train_s']}s)")

        t0 = time.time()
        feat_tr = projected_features(4, encode_zz, 2, d["q4_tr"])
        feat_te = projected_features(4, encode_zz, 2, d["q4_te"])
        proj = SVC(kernel="rbf", C=10, gamma="scale", random_state=args.seed + fold)
        proj.fit(feat_tr, ytr)
        pred = proj.predict(feat_te)
        rec = rec_row(
            "PROJ_ZZ4_SVM",
            yte,
            pred,
            p_mal_from_decision(proj.decision_function(feat_te)),
            fold,
            t0,
        )
        rows.append(rec)
        print(f"  {rec['model']:16s} bal={rec['balanced_accuracy']:.3f} "
              f"rec={rec['recall_malignant']:.3f} auc={rec['auc']:.3f}")

        t0 = time.time()
        fn = state_fn(4, encode_zz, 2)
        k_te = fidelity_kernel(states(fn, d["q4_te"]), states(fn, d["q4_tr"]))
        knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
        # sklearn kNN precomputed expects distances; convert fidelity → distance
        d_tr = 1.0 - fidelity_kernel(states(fn, d["q4_tr"]), states(fn, d["q4_tr"]))
        d_te = 1.0 - k_te
        np.fill_diagonal(d_tr, 0.0)
        knn.fit(d_tr, ytr)
        pred = knn.predict(d_te)
        neigh = knn.kneighbors(d_te, return_distance=False)
        p_mal = np.mean(ytr[neigh] == MALIGNANT, axis=1)
        rec = rec_row("QKNN5_ZZ4", yte, pred, p_mal, fold, t0)
        rows.append(rec)
        print(f"  {rec['model']:16s} bal={rec['balanced_accuracy']:.3f} "
              f"rec={rec['recall_malignant']:.3f} auc={rec['auc']:.3f}")

        for name, reup in (("VQC_FULL8", False), ("VQC_REUP8", True)):
            t0 = time.time()
            pred, p_mal, npar = train_vqc_ce(
                d["q8_tr"],
                ytr,
                d["q8_te"],
                n_qubits=8,
                n_layers=2,
                reup=reup,
                epochs=args.vqc_epochs,
                batch=32,
                lr=0.05,
                seed=args.seed + fold,
            )
            rec = rec_row(name, yte, pred, p_mal, fold, t0, n_params=npar)
            rows.append(rec)
            print(f"  {rec['model']:16s} bal={rec['balanced_accuracy']:.3f} "
                  f"rec={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
                  f"params={npar} ({rec['train_s']}s)")

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
    payload = {
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper": "paper_a_tabular_wbcd",
        "sweep": "quantum_algorithms",
        "positive_class": "malignant (label 0)",
        "quick": bool(args.quick),
        "summary": summary,
    }
    out_json = OUT / "quantum_sweep.json"
    df.to_csv(OUT / "quantum_sweep_folds.csv", index=False)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n======== sweep mean ± std (malignant = positive) ========")
    order = sorted(
        summary,
        key=lambda m: summary[m]["balanced_accuracy"]["mean"],
        reverse=True,
    )
    for model in order:
        s = summary[model]
        print(
            f"{model:16s}  bal {s['balanced_accuracy']['mean']:.3f}±"
            f"{s['balanced_accuracy']['std']:.3f}  "
            f"rec {s['recall_malignant']['mean']:.3f}±"
            f"{s['recall_malignant']['std']:.3f}  "
            f"auc {s['auc']['mean']:.3f}±{s['auc']['std']:.3f}"
        )
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
