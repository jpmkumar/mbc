#!/usr/bin/env python3
"""Paper A: why quantum arms lose — mechanistic diagnostics on confirmatory folds.

On the locked seed-42 splits we compute:
  - centred kernel-target alignment (KTA) vs test balanced accuracy (Spearman)
  - VQC depth ablation L=1..4 with matched MLP on PCA-8
  - train-fold gradient L2 norm at epoch 0 and after last epoch (VQC-8)
  - malignant false-negative overlap (Jaccard) among VQC-8, MLP-PCA8, SVM-RBF
  - Brier score for probability outputs

No test labels are used for KTA or gradients beyond standard training.
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
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss
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
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_mal = (y_true == MALIGNANT).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_malignant": float(
            recall_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)
        ),
        "brier_malignant": float(brier_score_loss(y_mal, p_mal)),
        "auc": float(roc_auc_score(y_mal, p_mal)),
    }


def center_kernel(k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def kernel_target_alignment(k: np.ndarray, y: np.ndarray) -> float:
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


def encode_iqp(x, wires, reps=1):
    """IQP map identical to train_wbcd_quantum_sweep.py (QSVM_IQP4).

    Revision fix: the earlier version applied data-independent CNOTs only, so
    its fidelity kernel reduced to the product-cosine angle kernel and
    duplicated the angle rows in the KTA analysis.
    """
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


def fidelity_gram_train(x_q: np.ndarray, map_name: str) -> np.ndarray:
    n_wires = x_q.shape[1]
    dev = qml.device("default.qubit", wires=n_wires)
    wires = list(range(n_wires))
    enc = {"z": encode_z, "zz": encode_zz, "angle": encode_angle, "iqp": encode_iqp}[map_name]

    @qml.qnode(dev)
    def psi(x):
        enc(x, wires)
        return qml.state()

    states = [psi(x_q[i]) for i in range(len(x_q))]
    k = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        for j in range(i, len(states)):
            val = float(np.abs(np.vdot(states[i], states[j])) ** 2)
            k[i, j] = k[j, i] = val
    return k


class MatchedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQCHead(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.05)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.circuit(xi, self.weights) for xi in x])


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train_vqc_with_grad_norm(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    set_seeds(seed)
    model = VQCHead(n_qubits, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    target = torch.tensor(2.0 * y_tr - 1.0, dtype=torch.float32)
    rng = np.random.default_rng(seed)

    def step_grad_norm() -> float:
        opt.zero_grad()
        z = model(xt[: min(64, len(xt))])
        loss = ((z - target[: min(64, len(xt))]) ** 2).mean()
        loss.backward()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.pow(2).sum())
        return float(np.sqrt(total))

    grad0 = step_grad_norm()
    model.train()
    for _ in range(epochs):
        for b in _batches(len(xt), 32, rng):
            opt.zero_grad()
            z = model(xt[b])
            loss = ((z - target[b]) ** 2).mean()
            loss.backward()
            opt.step()
    grad_end = step_grad_norm()
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x_te, dtype=torch.float32)).cpu().numpy()
    pred = (z > 0.0).astype(int)
    p_mal = (1.0 - z) / 2.0
    return pred, p_mal, grad0, grad_end


def train_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seeds(seed)
    model = MatchedMLP(x_tr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(xt), 32, rng):
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


def malignant_fn_mask(y_true, y_pred) -> set[int]:
    return {i for i, (a, b) in enumerate(zip(y_true, y_pred)) if a == MALIGNANT and b != MALIGNANT}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def prep_pca8(x_tr_s, x_te_s):
    pca = PCA(n_components=8, random_state=0)
    tr = pca.fit_transform(x_tr_s)
    te = pca.transform(x_te_s)
    mm = MinMaxScaler(feature_range=(0.0, np.pi))
    return mm.fit_transform(tr), mm.transform(te), float(pca.explained_variance_ratio_.sum())


def prep_pca4(x_tr_s, x_te_s):
    pca = PCA(n_components=4, random_state=0)
    tr = pca.fit_transform(x_tr_s)
    te = pca.transform(x_te_s)
    mm = MinMaxScaler(feature_range=(0.0, np.pi))
    return mm.fit_transform(tr), mm.transform(te)


def summarize_key(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if key in r]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    n_splits = 2 if args.quick else args.n_splits
    layer_grid = (1, 2) if args.quick else (1, 2, 3, 4)

    data = load_breast_cancer()
    x = data.data.astype(np.float64)
    y = data.target.astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    kta_rows: list[dict] = []
    depth_rows: list[dict] = []
    grad_rows: list[dict] = []
    overlap_rows: list[dict] = []
    brier_rows: list[dict] = []

    print(f"Quantum diagnosis | {n_splits} folds | seed={args.seed}")

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        t0 = time.time()
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x[tr])
        x_te_s = scaler.transform(x[te])
        y_tr, y_te = y[tr], y[te]

        x_tr_q8, x_te_q8, _ = prep_pca8(x_tr_s, x_te_s)
        x_tr_q4, x_te_q4 = prep_pca4(x_tr_s, x_te_s)

        for map_name in ("zz", "angle", "iqp", "z"):
            k_tr = fidelity_gram_train(x_tr_q4, map_name)
            kta = kernel_target_alignment(k_tr, y_tr)
            svc = SVC(C=10.0, kernel="precomputed")
            svc.fit(k_tr, y_tr)
            k_te = np.zeros((len(x_te_q4), len(x_tr_q4)))
            # test gram: reuse statevector approach in two steps (train states cached in k_tr diagonal block)
            dev = qml.device("default.qubit", wires=4)
            wires = list(range(4))
            enc = {"z": encode_z, "zz": encode_zz, "angle": encode_angle, "iqp": encode_iqp}[map_name]

            @qml.qnode(dev)
            def psi(vec):
                enc(vec, wires)
                return qml.state()

            st_tr = [psi(x_tr_q4[i]) for i in range(len(x_tr_q4))]
            for i in range(len(x_te_q4)):
                st = psi(x_te_q4[i])
                for j, stj in enumerate(st_tr):
                    k_te[i, j] = float(np.abs(np.vdot(st, stj)) ** 2)
            pred = svc.predict(k_te)
            p_mal = 1.0 / (1.0 + np.exp(svc.decision_function(k_te)))
            m = malignant_metrics(y_te, pred, p_mal)
            kta_rows.append(
                {
                    "fold": fold,
                    "kernel": map_name,
                    "kta_train": kta,
                    **m,
                }
            )

        for n_layers in layer_grid:
            pred, p_mal, _, _ = train_vqc_with_grad_norm(
                x_tr_q8,
                y_tr,
                x_te_q8,
                n_qubits=8,
                n_layers=n_layers,
                epochs=30,
                seed=args.seed + fold + 10 * n_layers,
            )
            mv = malignant_metrics(y_te, pred, p_mal)
            depth_rows.append({"fold": fold, "n_layers": n_layers, "model": "VQC-8", **mv})
            mp, p_mal_m = train_mlp(
                x_tr_q8,
                y_tr,
                x_te_q8,
                hidden=8,
                epochs=80,
                seed=args.seed + fold + 10 * n_layers,
            )
            mm = malignant_metrics(y_te, mp, p_mal_m)
            depth_rows.append({"fold": fold, "n_layers": n_layers, "model": "MLP-PCA8", **mm})

        pred_v, p_v, g0, g1 = train_vqc_with_grad_norm(
            x_tr_q8, y_tr, x_te_q8, n_qubits=8, n_layers=2, epochs=30, seed=args.seed + fold
        )
        pred_m, p_m = train_mlp(x_tr_q8, y_tr, x_te_q8, hidden=8, epochs=80, seed=args.seed + fold)
        svc = SVC(C=10.0, kernel="rbf", gamma="scale")
        svc.fit(x_tr_s, y_tr)
        pred_s = svc.predict(x_te_s)
        p_s = 1.0 / (1.0 + np.exp(svc.decision_function(x_te_s)))

        grad_rows.append(
            {
                "fold": fold,
                "grad_l2_epoch0": g0,
                "grad_l2_epoch30": g1,
                "grad_ratio_end_over_start": g1 / g0 if g0 > 1e-12 else float("nan"),
            }
        )

        fn_v = malignant_fn_mask(y_te, pred_v)
        fn_m = malignant_fn_mask(y_te, pred_m)
        fn_s = malignant_fn_mask(y_te, pred_s)
        overlap_rows.append(
            {
                "fold": fold,
                "fn_vqc": len(fn_v),
                "fn_mlp": len(fn_m),
                "fn_svm": len(fn_s),
                "jaccard_vqc_mlp": jaccard(fn_v, fn_m),
                "jaccard_vqc_svm": jaccard(fn_v, fn_s),
                "jaccard_mlp_svm": jaccard(fn_m, fn_s),
                "vqc_only_fn": len(fn_v - fn_m - fn_s),
            }
        )

        for name, pred, p_mal in (
            ("VQC-8", pred_v, p_v),
            ("MLP-PCA8", pred_m, p_m),
            ("SVM-RBF", pred_s, p_s),
        ):
            brier_rows.append(
                {"fold": fold, "model": name, **malignant_metrics(y_te, pred, p_mal)}
            )

        print(
            f"  fold {fold} done in {time.time()-t0:.0f}s | "
            f"grad {g0:.4e}->{g1:.4e} | fn V/M/S {len(fn_v)}/{len(fn_m)}/{len(fn_s)}"
        )

    kta_df = pd.DataFrame(kta_rows)
    spearman = {}
    for kernel in kta_df["kernel"].unique():
        sub = kta_df[kta_df["kernel"] == kernel]
        if len(sub) >= 3:
            rho, p = stats.spearmanr(sub["kta_train"], sub["balanced_accuracy"])
            spearman[kernel] = {"rho": float(rho), "p": float(p), "n": int(len(sub))}
    pooled = kta_df
    rho_all, p_all = stats.spearmanr(pooled["kta_train"], pooled["balanced_accuracy"])

    depth_summary = {}
    for layers in layer_grid:
        for model in ("VQC-8", "MLP-PCA8"):
            sub = [r for r in depth_rows if r["n_layers"] == layers and r["model"] == model]
            key = f"L{layers}_{model}"
            depth_summary[key] = {
                "balanced_accuracy": summarize_key(sub, "balanced_accuracy"),
                "recall_malignant": summarize_key(sub, "recall_malignant"),
            }

    payload = {
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "paper_a_tabular_wbcd",
        "sweep": "quantum_diagnosis",
        "spearman_kta_vs_bal_acc_pooled": {"rho": float(rho_all), "p": float(p_all), "n": len(pooled)},
        "spearman_by_kernel": spearman,
        "grad_norm": {
            "grad_l2_epoch0": summarize_key(grad_rows, "grad_l2_epoch0"),
            "grad_l2_epoch30": summarize_key(grad_rows, "grad_l2_epoch30"),
            "grad_ratio": summarize_key(grad_rows, "grad_ratio_end_over_start"),
        },
        "malignant_fn_overlap": {
            "fn_vqc_mean": summarize_key(overlap_rows, "fn_vqc"),
            "fn_mlp_mean": summarize_key(overlap_rows, "fn_mlp"),
            "fn_svm_mean": summarize_key(overlap_rows, "fn_svm"),
            "jaccard_vqc_mlp": summarize_key(overlap_rows, "jaccard_vqc_mlp"),
            "jaccard_vqc_svm": summarize_key(overlap_rows, "jaccard_vqc_svm"),
            "vqc_only_fn": summarize_key(overlap_rows, "vqc_only_fn"),
        },
        "brier_malignant": {
            m: summarize_key([r for r in brier_rows if r["model"] == m], "brier_malignant")
            for m in ("VQC-8", "MLP-PCA8", "SVM-RBF")
        },
        "depth_ablation_summary": depth_summary,
        "kta_folds": kta_rows,
        "depth_folds": depth_rows,
        "grad_folds": grad_rows,
        "overlap_folds": overlap_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "quantum_diagnosis.json"
    out.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(kta_rows).to_csv(OUT / "quantum_diagnosis_kta.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(OUT / "quantum_diagnosis_depth.csv", index=False)
    print(f"Wrote {out}")
    print(f"  Spearman(KTA, bal_acc) pooled rho={rho_all:.3f} p={p_all:.4g}")
    for layers in layer_grid:
        v = depth_summary[f"L{layers}_VQC-8"]["balanced_accuracy"]
        m = depth_summary[f"L{layers}_MLP-PCA8"]["balanced_accuracy"]
        print(f"  depth L={layers}: VQC {v['mean']:.3f}  MLP {m['mean']:.3f}")


if __name__ == "__main__":
    main()