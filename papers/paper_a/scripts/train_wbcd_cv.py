#!/usr/bin/env python3
"""Paper A confirmatory CV: leakage-free WBCD hybrid QML.

Malignant is the positive class (sklearn WBCD: 0 = malignant, 1 = benign).
Scaler / PCA / MinMax are fit on each training fold only.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "papers/paper_a/results"

# sklearn WBCD: 0 = malignant (clinical positive), 1 = benign
MALIGNANT = 0


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def vqc_param_count(n_qubits: int, n_layers: int) -> int:
    return n_layers * n_qubits * 3


def malignant_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_mal: np.ndarray) -> dict:
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


def last_hidden_relu(mlp: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """Activations of the last hidden layer (before the output layer)."""
    h = x
    for i in range(len(mlp.coefs_) - 1):
        h = np.maximum(h @ mlp.coefs_[i] + mlp.intercepts_[i], 0.0)
    return h


class MatchedMLP(nn.Module):
    """LayerNorm + GELU MLP with equal-or-greater capacity than a VQC head."""

    def __init__(self, in_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQCHead(nn.Module):
    """Angle-Y + StronglyEntanglingLayers; Pauli-Z on qubit 0 (reference study)."""

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(
            qnode, {"weights": (n_layers, n_qubits, 3)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train_mlp_torch(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    set_seeds(seed)
    model = MatchedMLP(x_tr.shape[1], hidden=hidden)
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
    return pred, prob[:, 0], count_params(model)


def train_vqc(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    set_seeds(seed)
    model = VQCHead(n_qubits, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    # Reference study: target = 2y - 1  (+1 benign, -1 malignant)
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
    return pred, p_mal, count_params(model)


@dataclass
class FoldPrep:
    x_tr_s: np.ndarray
    x_te_s: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    x_tr_pca: np.ndarray
    x_te_pca: np.ndarray
    x_tr_q: np.ndarray
    x_te_q: np.ndarray
    x_tr_pca4: np.ndarray
    x_te_pca4: np.ndarray
    x_tr_q4: np.ndarray
    x_te_q4: np.ndarray
    pca_var: float
    pca4_var: float


def prepare_fold(
    x_tr, x_te, y_tr, y_te, n_qubits: int, n_qsvm: int = 4
) -> FoldPrep:
    scaler = StandardScaler().fit(x_tr)
    x_tr_s = scaler.transform(x_tr)
    x_te_s = scaler.transform(x_te)
    pca = PCA(n_components=n_qubits).fit(x_tr_s)
    x_tr_pca = pca.transform(x_tr_s)
    x_te_pca = pca.transform(x_te_s)
    mm = MinMaxScaler(feature_range=(0.0, np.pi)).fit(x_tr_pca)
    pca4 = PCA(n_components=n_qsvm).fit(x_tr_s)
    x_tr_pca4 = pca4.transform(x_tr_s)
    x_te_pca4 = pca4.transform(x_te_s)
    mm4 = MinMaxScaler(feature_range=(0.0, np.pi)).fit(x_tr_pca4)
    return FoldPrep(
        x_tr_s=x_tr_s,
        x_te_s=x_te_s,
        y_tr=y_tr,
        y_te=y_te,
        x_tr_pca=x_tr_pca,
        x_te_pca=x_te_pca,
        x_tr_q=mm.transform(x_tr_pca),
        x_te_q=mm.transform(x_te_pca),
        x_tr_pca4=x_tr_pca4,
        x_te_pca4=x_te_pca4,
        x_tr_q4=mm4.transform(x_tr_pca4),
        x_te_q4=mm4.transform(x_te_pca4),
        pca_var=float(pca.explained_variance_ratio_.sum()),
        pca4_var=float(pca4.explained_variance_ratio_.sum()),
    )


def _zz_feature_map(x, wires, reps: int = 2) -> None:
    """Qiskit-style ZZFeatureMap used by Jose / Yadav QSVM papers."""
    n = len(wires)
    for _ in range(reps):
        for i, w in enumerate(wires):
            qml.Hadamard(wires=w)
            qml.RZ(2.0 * x[i], wires=w)
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.RZ(2.0 * x[i] * x[i + 1], wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def _make_state_qnode(n_qubits: int, reps: int = 2):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def state(x):
        _zz_feature_map(x, wires, reps=reps)
        return qml.state()

    return state


def _fidelity_kernel_from_states(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    """K_ij = |<psi_a_i | psi_b_j>|^2. Same kernel as pairwise overlap QSVM."""
    gram = psi_a.conj() @ psi_b.T
    return np.abs(gram) ** 2


def train_qsvm(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    n_qubits: int,
    reps: int,
    c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """ZZ fidelity-kernel SVM (Jose / Yadav / Vashisth) via statevectors."""
    set_seeds(seed)
    state = _make_state_qnode(n_qubits, reps=reps)
    psi_tr = np.stack([np.asarray(state(x), dtype=np.complex128) for x in x_tr])
    psi_te = np.stack([np.asarray(state(x), dtype=np.complex128) for x in x_te])
    k_tr = np.real(_fidelity_kernel_from_states(psi_tr, psi_tr))
    k_te = np.real(_fidelity_kernel_from_states(psi_te, psi_tr))
    clf = SVC(kernel="precomputed", C=c)
    clf.fit(k_tr, y_tr)
    pred = clf.predict(k_te)
    scores = clf.decision_function(k_te)
    p_mal = 1.0 / (1.0 + np.exp(scores))
    n_sv = int(clf.n_support_.sum()) if hasattr(clf, "n_support_") else 0
    return pred, p_mal, n_sv


def run_classical_sklearn(name: str, model, x_tr, y_tr, x_te, y_te) -> dict:
    t0 = time.time()
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        p_mal = model.predict_proba(x_te)[:, classes.index(MALIGNANT)]
    elif hasattr(model, "decision_function"):
        # Higher decision score → class 1 (benign). Monotonic map for AUC.
        p_mal = 1.0 / (1.0 + np.exp(model.decision_function(x_te)))
    else:
        p_mal = (pred == MALIGNANT).astype(float)
    out = malignant_metrics(y_te, pred, p_mal)
    out.update(model=name, train_s=round(time.time() - t0, 3), n_params=None)
    return out


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = a - b
    out = {
        "n": int(len(a)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta": float(delta.mean()),
    }
    if len(a) >= 2 and np.std(delta) > 0:
        t_stat, t_p = stats.ttest_rel(a, b)
        out["paired_t"] = float(t_stat)
        out["paired_t_p"] = float(t_p)
    else:
        out["paired_t"] = None
        out["paired_t_p"] = None
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="pratt")
        out["wilcoxon"] = float(w_stat)
        out["wilcoxon_p"] = float(w_p)
    except ValueError:
        out["wilcoxon"] = None
        out["wilcoxon_p"] = None
    return out


def summarize(rows: list[dict]) -> dict:
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
        summary[model]["n_folds"] = int(len(g))
        if g["n_params"].notna().any():
            summary[model]["n_params"] = int(g["n_params"].dropna().iloc[0])
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper A confirmatory WBCD CV")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--hybrid-qubits", type=int, default=6)
    p.add_argument("--vqc-epochs", type=int, default=30)
    p.add_argument("--mlp-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--vqc-lr", type=float, default=0.05)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--qsvm-qubits", type=int, default=4)
    p.add_argument("--qsvm-reps", type=int, default=2)
    p.add_argument("--qsvm-c", type=float, default=10.0)
    p.add_argument(
        "--qsvm-only",
        action="store_true",
        help="Run QSVM + matched PCA-4 classical controls (skip VQC/hybrid)",
    )
    p.add_argument("--no-qsvm", action="store_true", help="Skip QSVM arm")
    p.add_argument(
        "--quick",
        action="store_true",
        help="2 folds, 3 VQC epochs — smoke test only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_splits = 2
        args.n_repeats = 1
        args.vqc_epochs = 3
        args.mlp_epochs = 5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    data = load_breast_cancer()
    x, y = data.data, data.target
    print(
        f"WBCD n={len(y)} features={x.shape[1]} "
        f"malignant={(y == 0).sum()} benign={(y == 1).sum()}"
    )
    print(
        f"CV: {args.n_repeats}x{args.n_splits} stratified | "
        f"VQC {args.n_qubits}q x {args.n_layers}L | epochs={args.vqc_epochs}"
    )

    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )
    rows: list[dict] = []
    hidden_8 = max(args.n_qubits, 8)
    hidden_6 = max(args.hybrid_qubits, 8)

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        prep = prepare_fold(
            x[tr], x[te], y[tr], y[te], args.n_qubits, n_qsvm=args.qsvm_qubits
        )
        print(f"\n=== fold {fold}  n_tr={len(tr)} n_te={len(te)}  "
              f"PCA var={prep.pca_var:.3f} ===")

        # --- classical on 30-D ---
        for name, clf in [
            (
                "SVM_RBF",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    random_state=args.seed + fold,
                ),
            ),
            (
                "SVM_LIN",
                SVC(kernel="linear", C=10, random_state=args.seed + fold),
            ),
            (
                "RF",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=args.seed + fold,
                ),
            ),
        ]:
            rec = run_classical_sklearn(
                name, clf, prep.x_tr_s, prep.y_tr, prep.x_te_s, prep.y_te
            )
            rec["fold"] = fold
            rows.append(rec)
            print(f"  {name:16s} bal_acc={rec['balanced_accuracy']:.3f} "
                  f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f}")

        if not args.no_qsvm:
            for name, clf in [
                (
                    "SVM_RBF_PCA4",
                    SVC(
                        kernel="rbf",
                        C=args.qsvm_c,
                        gamma="scale",
                        random_state=args.seed + fold,
                    ),
                ),
                (
                    "SVM_LIN_PCA4",
                    SVC(
                        kernel="linear",
                        C=args.qsvm_c,
                        random_state=args.seed + fold,
                    ),
                ),
            ]:
                rec = run_classical_sklearn(
                    name, clf, prep.x_tr_pca4, prep.y_tr, prep.x_te_pca4, prep.y_te
                )
                rec["fold"] = fold
                rec["pca4_var"] = prep.pca4_var
                rows.append(rec)
                print(
                    f"  {name:16s} bal_acc={rec['balanced_accuracy']:.3f} "
                    f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f}"
                )

            t0 = time.time()
            pred, p_mal, npar = train_mlp_torch(
                prep.x_tr_q4,
                prep.y_tr,
                prep.x_te_q4,
                hidden=max(args.qsvm_qubits, 8),
                epochs=args.mlp_epochs,
                batch_size=args.batch_size,
                lr=args.mlp_lr,
                seed=args.seed + fold,
            )
            rec = malignant_metrics(prep.y_te, pred, p_mal)
            rec.update(
                model="MLP_PCA4",
                fold=fold,
                train_s=round(time.time() - t0, 3),
                n_params=npar,
            )
            rows.append(rec)
            print(
                f"  {'MLP_PCA4':16s} bal_acc={rec['balanced_accuracy']:.3f} "
                f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
                f"params={npar}"
            )

            t0 = time.time()
            pred, p_mal, n_sv = train_qsvm(
                prep.x_tr_q4,
                prep.y_tr,
                prep.x_te_q4,
                n_qubits=args.qsvm_qubits,
                reps=args.qsvm_reps,
                c=args.qsvm_c,
                seed=args.seed + fold,
            )
            rec = malignant_metrics(prep.y_te, pred, p_mal)
            rec.update(
                model="QSVM_ZZ4",
                fold=fold,
                train_s=round(time.time() - t0, 3),
                n_params=n_sv,
                pca4_var=prep.pca4_var,
            )
            rows.append(rec)
            print(
                f"  {'QSVM_ZZ4':16s} bal_acc={rec['balanced_accuracy']:.3f} "
                f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
                f"n_sv={n_sv} ({rec['train_s']}s)"
            )

        if args.qsvm_only:
            continue

        t0 = time.time()
        pred, p_mal, npar = train_mlp_torch(
            prep.x_tr_s,
            prep.y_tr,
            prep.x_te_s,
            hidden=32,
            epochs=args.mlp_epochs,
            batch_size=args.batch_size,
            lr=args.mlp_lr,
            seed=args.seed + fold,
        )
        rec = malignant_metrics(prep.y_te, pred, p_mal)
        rec.update(
            model="MLP_30D",
            fold=fold,
            train_s=round(time.time() - t0, 3),
            n_params=npar,
        )
        rows.append(rec)
        print(f"  {'MLP_30D':16s} bal_acc={rec['balanced_accuracy']:.3f} "
              f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
              f"params={npar}")

        # --- matched MLP vs VQC on identical 8-D quantum features ---
        t0 = time.time()
        pred, p_mal, npar = train_mlp_torch(
            prep.x_tr_q,
            prep.y_tr,
            prep.x_te_q,
            hidden=hidden_8,
            epochs=args.mlp_epochs,
            batch_size=args.batch_size,
            lr=args.mlp_lr,
            seed=args.seed + fold,
        )
        rec = malignant_metrics(prep.y_te, pred, p_mal)
        rec.update(
            model="MLP_PCA8",
            fold=fold,
            train_s=round(time.time() - t0, 3),
            n_params=npar,
        )
        rows.append(rec)
        print(f"  {'MLP_PCA8':16s} bal_acc={rec['balanced_accuracy']:.3f} "
              f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
              f"params={npar}")

        t0 = time.time()
        pred, p_mal, npar = train_vqc(
            prep.x_tr_q,
            prep.y_tr,
            prep.x_te_q,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            epochs=args.vqc_epochs,
            batch_size=args.batch_size,
            lr=args.vqc_lr,
            seed=args.seed + fold,
        )
        rec = malignant_metrics(prep.y_te, pred, p_mal)
        rec.update(
            model="VQC_8",
            fold=fold,
            train_s=round(time.time() - t0, 3),
            n_params=npar,
            pca_var=prep.pca_var,
        )
        rows.append(rec)
        print(f"  {'VQC_8':16s} bal_acc={rec['balanced_accuracy']:.3f} "
              f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
              f"params={npar} ({rec['train_s']}s)")

        # --- two-stage hybrid: MLP 30→32→6, then VQC-6 vs matched MLP-6 ---
        t0 = time.time()
        extractor = MLPClassifier(
            hidden_layer_sizes=(32, args.hybrid_qubits),
            activation="relu",
            solver="adam",
            max_iter=200 if args.quick else 500,
            random_state=args.seed + fold,
            early_stopping=True,
            validation_fraction=0.15,
        )
        extractor.fit(prep.x_tr_s, prep.y_tr)
        h_tr = last_hidden_relu(extractor, prep.x_tr_s)
        h_te = last_hidden_relu(extractor, prep.x_te_s)
        mm6 = MinMaxScaler(feature_range=(0.0, np.pi)).fit(h_tr)
        h_tr_q = mm6.transform(h_tr)
        h_te_q = mm6.transform(h_te)

        ext_pred = extractor.predict(prep.x_te_s)
        ext_p = extractor.predict_proba(prep.x_te_s)
        ext_classes = list(extractor.classes_)
        ext_rec = malignant_metrics(
            prep.y_te, ext_pred, ext_p[:, ext_classes.index(MALIGNANT)]
        )
        ext_rec.update(
            model="MLP_extractor",
            fold=fold,
            train_s=round(time.time() - t0, 3),
            n_params=int(sum(c.size for c in extractor.coefs_)
                         + sum(b.size for b in extractor.intercepts_)),
        )
        rows.append(ext_rec)
        print(f"  {'MLP_extractor':16s} bal_acc={ext_rec['balanced_accuracy']:.3f} "
              f"rec_mal={ext_rec['recall_malignant']:.3f} auc={ext_rec['auc']:.3f}")

        t0 = time.time()
        pred, p_mal, npar = train_mlp_torch(
            h_tr_q,
            prep.y_tr,
            h_te_q,
            hidden=hidden_6,
            epochs=args.mlp_epochs,
            batch_size=args.batch_size,
            lr=args.mlp_lr,
            seed=args.seed + fold,
        )
        rec = malignant_metrics(prep.y_te, pred, p_mal)
        rec.update(
            model="MLP_H6",
            fold=fold,
            train_s=round(time.time() - t0, 3),
            n_params=npar,
        )
        rows.append(rec)
        print(f"  {'MLP_H6':16s} bal_acc={rec['balanced_accuracy']:.3f} "
              f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
              f"params={npar}")

        t0 = time.time()
        pred, p_mal, npar = train_vqc(
            h_tr_q,
            prep.y_tr,
            h_te_q,
            n_qubits=args.hybrid_qubits,
            n_layers=args.n_layers,
            epochs=args.vqc_epochs,
            batch_size=args.batch_size,
            lr=args.vqc_lr,
            seed=args.seed + fold,
        )
        rec = malignant_metrics(prep.y_te, pred, p_mal)
        rec.update(
            model="HYBRID_MLP_VQC",
            fold=fold,
            train_s=round(time.time() - t0, 3),
            n_params=npar,
        )
        rows.append(rec)
        print(f"  {'HYBRID_MLP_VQC':16s} bal_acc={rec['balanced_accuracy']:.3f} "
              f"rec_mal={rec['recall_malignant']:.3f} auc={rec['auc']:.3f} "
              f"params={npar} ({rec['train_s']}s)")

    df = pd.DataFrame(rows)
    summary = summarize(rows)
    pairs = {
        "VQC_8_vs_MLP_PCA8": ("VQC_8", "MLP_PCA8"),
        "VQC_8_vs_SVM_RBF": ("VQC_8", "SVM_RBF"),
        "HYBRID_vs_MLP_H6": ("HYBRID_MLP_VQC", "MLP_H6"),
        "HYBRID_vs_MLP_30D": ("HYBRID_MLP_VQC", "MLP_30D"),
        "HYBRID_vs_extractor": ("HYBRID_MLP_VQC", "MLP_extractor"),
        "QSVM_ZZ4_vs_SVM_RBF_PCA4": ("QSVM_ZZ4", "SVM_RBF_PCA4"),
        "QSVM_ZZ4_vs_SVM_LIN_PCA4": ("QSVM_ZZ4", "SVM_LIN_PCA4"),
        "QSVM_ZZ4_vs_MLP_PCA4": ("QSVM_ZZ4", "MLP_PCA4"),
        "QSVM_ZZ4_vs_SVM_RBF": ("QSVM_ZZ4", "SVM_RBF"),
        "QSVM_ZZ4_vs_SVM_LIN": ("QSVM_ZZ4", "SVM_LIN"),
    }
    stats_out = {}
    for key, (a, b) in pairs.items():
        stats_out[key] = {}
        for metric in ("balanced_accuracy", "recall_malignant", "auc"):
            sa = df.loc[df.model == a].sort_values("fold")[metric].to_numpy()
            sb = df.loc[df.model == b].sort_values("fold")[metric].to_numpy()
            if len(sa) == 0 or len(sb) == 0 or len(sa) != len(sb):
                continue
            stats_out[key][metric] = paired_tests(sa, sb)

    payload = {
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper": "paper_a_tabular_wbcd",
        "protocol": "papers/paper_a/PROTOCOL.md",
        "positive_class": "malignant (label 0)",
        "quick": bool(args.quick),
        "cv": {
            "n_splits": args.n_splits,
            "n_repeats": args.n_repeats,
            "seed": args.seed,
        },
        "vqc": {
            "n_qubits": args.n_qubits,
            "n_layers": args.n_layers,
            "epochs": args.vqc_epochs,
            "theoretical_params": vqc_param_count(args.n_qubits, args.n_layers),
        },
        "summary": summary,
        "stats": stats_out,
    }

    folds_csv = args.out_dir / "confirmatory_folds.csv"
    out_json = args.out_dir / "confirmatory_cv.json"
    stats_json = args.out_dir / "confirmatory_stats.json"
    df.to_csv(folds_csv, index=False)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    stats_json.write_text(json.dumps(stats_out, indent=2) + "\n")

    print("\n======== mean ± std (malignant = positive) ========")
    for model, s in summary.items():
        print(
            f"{model:16s}  bal_acc {s['balanced_accuracy']['mean']:.3f}±"
            f"{s['balanced_accuracy']['std']:.3f}  "
            f"rec_mal {s['recall_malignant']['mean']:.3f}±"
            f"{s['recall_malignant']['std']:.3f}  "
            f"auc {s['auc']['mean']:.3f}±{s['auc']['std']:.3f}"
        )
    print(f"\nWrote {out_json}")
    print(f"Wrote {folds_csv}")
    print(f"Wrote {stats_json}")


if __name__ == "__main__":
    main()
