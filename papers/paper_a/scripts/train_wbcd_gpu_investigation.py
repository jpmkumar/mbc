#!/usr/bin/env python3
"""GPU-friendly Paper A investigation: why VQC loses on WBCD.

Arms (same folds as train_wbcd_cv.py):
  1. Sample efficiency: train on 25/50/75/100% of each train fold
  2. Random labels: memorisation of shuffled y on train (expressivity test)
  3. Entangle ablation: StronglyEntangling vs basic entangler (entanglement on/off)
  4. LR grid: small search with fixed epoch budget for VQC vs MLP
  5. Product-state fidelity: mean overlap to unentangled |+>^n after encoding

VQC training uses default.qubit + backprop (same as train_wbcd_cv.py).
Lightning + adjoint is version-fragile with TorchLayer (no grad_fn on older
PennyLane); WBCD 8-qubit circuits are small enough that default.qubit is fine
even on a GPU box. Optional --device lightning.qubit|lightning.gpu for newer PL.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
MALIGNANT = 0


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_diff_method(dev_name: str) -> str:
    if str(dev_name).startswith("lightning"):
        return "adjoint"
    return "backprop"


def pick_device(n_wires: int, requested: str = "auto") -> str:
    """Resolve PennyLane device. Default: default.qubit (reliable Torch grads)."""
    _ = n_wires
    if requested and requested != "auto":
        return requested
    # Match confirmatory CV: TorchLayer + backprop needs a state sim device.
    return "default.qubit"


def assert_torch_grads(dev_name: str) -> None:
    """Fail fast if QNode+TorchLayer cannot backprop (avoids mid-epoch crash)."""
    n = 2
    dev = qml.device(dev_name, wires=n)
    diff = pick_diff_method(dev_name)

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n))
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff)
    layer = qml.qnn.TorchLayer(qnode, {"weights": (1, n, 3)})
    x = torch.rand(4, n)
    loss = layer(x).pow(2).mean()
    loss.backward()
    grad = next(layer.parameters()).grad
    if grad is None:
        raise RuntimeError(
            f"No torch grad with device={dev_name} diff={diff}. "
            "Use --device default.qubit or upgrade pennylane."
        )


def prep_pca8(x_tr_s: np.ndarray, x_te_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=8, random_state=0)
    tr = pca.fit_transform(x_tr_s)
    te = pca.transform(x_te_s)
    mm = MinMaxScaler(feature_range=(0.0, np.pi))
    return mm.fit_transform(tr), mm.transform(te)


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


def build_vqc(n_qubits: int, n_layers: int, entangle: str, dev_name: str):
    dev = qml.device(dev_name, wires=n_qubits)
    diff = pick_diff_method(dev_name)

    if entangle == "none":
        def circuit(inputs):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            return qml.expval(qml.PauliZ(0))

        return qml.QNode(circuit, dev, interface="torch", diff_method=diff), 0

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        if entangle == "strong":
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        else:
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff)
    # StronglyEntanglingLayers: (L, n, 3); BasicEntanglerLayers: (L, n)
    shape = (
        (max(n_layers, 1), n_qubits, 3)
        if entangle == "strong"
        else (max(n_layers, 1), n_qubits)
    )
    return qml.qnn.TorchLayer(qnode, {"weights": shape}), shape[0]


class VQCModule(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, entangle: str, dev_name: str):
        super().__init__()
        self.entangle = entangle
        built, self.effective_layers = build_vqc(n_qubits, n_layers, entangle, dev_name)
        if entangle == "none":
            self.qlayer = None
            self._encode = built
        else:
            self.qlayer = built
            self._encode = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qlayer is None:
            # Batched encoding-only expectation
            outs = [self._encode(row) for row in x]
            return torch.stack(outs)
        return self.qlayer(x)


def train_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    set_seeds(seed)
    model = MatchedMLP(x_tr.shape[1], hidden=8)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(xt))
        for start in range(0, len(xt), 32):
            b = idx[start : start + 32]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_te, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    return _metrics(y_te, pred, prob[:, 0])


def train_vqc(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_layers: int,
    entangle: str,
    dev_name: str,
    epochs: int,
    lr: float,
    seed: int,
    loss_name: str = "mse",
    return_train_pred: bool = False,
) -> dict:
    set_seeds(seed)
    model = VQCModule(8, n_layers, entangle, dev_name)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    target = torch.tensor(2.0 * y_tr - 1.0, dtype=torch.float32)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    # Encoding-only ablation has no trainable weights — evaluate, do not backward.
    if entangle != "none" and any(p.requires_grad for p in model.parameters()):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            idx = rng.permutation(len(xt))
            for start in range(0, len(xt), 32):
                b = idx[start : start + 32]
                opt.zero_grad()
                z = model(xt[b])
                if loss_name == "ce":
                    logits = torch.stack((-z, z), dim=-1)
                    loss = ce(logits, yt[b])
                else:
                    loss = ((z - target[b]) ** 2).mean()
                loss.backward()
                opt.step()
    model.eval()
    with torch.no_grad():
        z_te = model(torch.tensor(x_te, dtype=torch.float32)).cpu().numpy()
        z_tr = model(torch.tensor(x_tr, dtype=torch.float32)).cpu().numpy()
    pred = (z_te > 0.0).astype(int)
    p_mal = (1.0 - z_te) / 2.0
    out = _metrics(y_te, pred, p_mal)
    if return_train_pred:
        out["train_bal_acc"] = float(
            balanced_accuracy_score(y_tr, (z_tr > 0.0).astype(int))
        )
    return out


def _metrics(y_te, pred, p_mal) -> dict:
    y_mal = (y_te == MALIGNANT).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "recall_malignant": float((pred == MALIGNANT)[y_te == MALIGNANT].mean())
        if (y_te == MALIGNANT).any()
        else 0.0,
        "auc": float(roc_auc_score(y_mal, p_mal)),
    }


def mean_product_state_fidelity(x_batch: np.ndarray, n_qubits: int, dev_name: str) -> float:
    """Mean |<psi(x)|+...+>|^2 after angle encoding (entanglement diagnostic)."""
    # state() needs a statevector device; always use default.qubit here
    _ = dev_name
    dev = qml.device("default.qubit", wires=n_qubits)
    plus = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)

    @qml.qnode(dev)
    def enc(x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        return qml.state()

    fids = []
    for x in x_batch[: min(64, len(x_batch))]:
        st = np.asarray(enc(x))
        fids.append(float(np.abs(np.vdot(st, plus)) ** 2))
    return float(np.mean(fids))


def summarize(rows: list[dict]) -> dict:
    out: dict[str, list] = {}
    for r in rows:
        # Keep models separate under the same arm (SVM lacks recall_malignant, etc.)
        arm = r.get("arm", "unknown")
        model = r.get("model")
        key = f"{arm}|{model}" if model else arm
        out.setdefault(key, []).append(r)
    metric_keys = (
        "balanced_accuracy",
        "recall_malignant",
        "auc",
        "train_acc",
        "train_bal_acc",
        "mean_fidelity_to_plus",
    )
    summ = {}
    for key, rs in out.items():
        block = {}
        for k in metric_keys:
            vals = [float(x[k]) for x in rs if k in x]
            if not vals:
                continue
            block[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "n": len(vals),
            }
        if block:
            summ[key] = block
    return summ


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "papers/paper_a/results/gpu_investigation")
    p.add_argument("--vqc-epochs", type=int, default=30)
    p.add_argument("--mlp-epochs", type=int, default=80)
    p.add_argument(
        "--device",
        default="auto",
        help="PennyLane device: auto (=default.qubit), default.qubit, lightning.qubit, lightning.gpu",
    )
    args = p.parse_args()

    data = load_breast_cancer()
    x = data.data.astype(np.float64)
    y = data.target.astype(int)
    # StratifiedKFold requires n_splits >= 2
    n_splits = 2 if args.quick else 5
    fractions = [0.5, 1.0] if args.quick else [0.25, 0.5, 0.75, 1.0]
    if args.quick:
        args.vqc_epochs = min(args.vqc_epochs, 10)
        args.mlp_epochs = min(args.mlp_epochs, 30)
    dev_name = pick_device(8, args.device)
    assert_torch_grads(dev_name)
    lrs_vqc = [0.01, 0.05, 0.1] if not args.quick else [0.05]
    lrs_mlp = [1e-3, 3e-3] if not args.quick else [1e-3]

    rows: list[dict] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    print(
        f"GPU investigation | device={dev_name} | diff={pick_diff_method(dev_name)} "
        f"| cuda={torch.cuda.is_available()} | folds={n_splits} | pennylane={qml.__version__}"
    )

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x[tr])
        x_te_s = scaler.transform(x[te])
        y_tr, y_te = y[tr], y[te]
        x_tr_q, x_te_q = prep_pca8(x_tr_s, x_te_s)

        fid = mean_product_state_fidelity(x_tr_q, 8, dev_name)
        rows.append(
            {
                "arm": "encoding_product_fidelity",
                "fold": fold,
                "mean_fidelity_to_plus": fid,
            }
        )

        for frac in fractions:
            n_use = max(32, int(len(y_tr) * frac))
            sel = np.random.default_rng(args.seed + fold).choice(len(y_tr), n_use, replace=False)
            x_sub, y_sub = x_tr_q[sel], y_tr[sel]

            m_mlp = train_mlp(
                x_sub, y_sub, x_te_q, y_te, epochs=args.mlp_epochs, lr=1e-3, seed=args.seed + fold
            )
            rows.append(
                {
                    "arm": "sample_efficiency",
                    "model": "MLP-PCA8",
                    "fold": fold,
                    "train_fraction": frac,
                    **m_mlp,
                }
            )
            m_vqc = train_vqc(
                x_sub,
                y_sub,
                x_te_q,
                y_te,
                n_layers=2,
                entangle="strong",
                dev_name=dev_name,
                epochs=args.vqc_epochs,
                lr=0.05,
                seed=args.seed + fold,
            )
            rows.append(
                {
                    "arm": "sample_efficiency",
                    "model": "VQC-8",
                    "fold": fold,
                    "train_fraction": frac,
                    **m_vqc,
                }
            )

        y_shuffle = y_tr.copy()
        rng = np.random.default_rng(args.seed + 99 + fold)
        rng.shuffle(y_shuffle)
        m_vqc_r = train_vqc(
            x_tr_q,
            y_shuffle,
            x_te_q,
            y_te,
            n_layers=2,
            entangle="strong",
            dev_name=dev_name,
            epochs=args.vqc_epochs * 2,
            lr=0.05,
            seed=args.seed + fold,
            return_train_pred=True,
        )
        m_mlp_r = train_mlp(
            x_tr_q, y_shuffle, x_te_q, y_te, epochs=args.mlp_epochs * 2, lr=1e-3, seed=args.seed + fold
        )
        set_seeds(args.seed + fold)
        model = MatchedMLP(x_tr_q.shape[1], 8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xt = torch.tensor(x_tr_q, dtype=torch.float32)
        ys = torch.tensor(y_shuffle, dtype=torch.long)
        for _ in range(args.mlp_epochs * 2):
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(xt), ys).backward()
            opt.step()
        with torch.no_grad():
            train_pred = model(xt).argmax(dim=1).cpu().numpy()
        rows.append(
            {
                "arm": "random_labels_train_only",
                "model": "VQC-8",
                "fold": fold,
                "train_bal_acc": m_vqc_r.pop("train_bal_acc"),
                **m_vqc_r,
            }
        )
        rows.append(
            {
                "arm": "random_labels_train_only",
                "model": "MLP-PCA8",
                "fold": fold,
                "train_bal_acc": float(balanced_accuracy_score(y_shuffle, train_pred)),
                **m_mlp_r,
            }
        )

        for ent in ("none", "basic", "strong"):
            m = train_vqc(
                x_tr_q,
                y_tr,
                x_te_q,
                y_te,
                n_layers=2,
                entangle=ent,
                dev_name=dev_name,
                epochs=args.vqc_epochs,
                lr=0.05,
                seed=args.seed + fold,
            )
            rows.append(
                {"arm": "entangle_ablation", "entangle": ent, "fold": fold, "model": "VQC-8", **m}
            )

        best_vqc, best_mlp = -1.0, -1.0
        for lr in lrs_vqc:
            m = train_vqc(
                x_tr_q,
                y_tr,
                x_te_q,
                y_te,
                n_layers=2,
                entangle="strong",
                dev_name=dev_name,
                epochs=args.vqc_epochs,
                lr=lr,
                seed=args.seed + fold,
            )
            best_vqc = max(best_vqc, m["balanced_accuracy"])
        for lr in lrs_mlp:
            m = train_mlp(x_tr_q, y_tr, x_te_q, y_te, epochs=args.mlp_epochs, lr=lr, seed=args.seed + fold)
            best_mlp = max(best_mlp, m["balanced_accuracy"])
        rows.append(
            {
                "arm": "lr_grid_best",
                "fold": fold,
                "model": "VQC-8",
                "balanced_accuracy": best_vqc,
            }
        )
        rows.append(
            {
                "arm": "lr_grid_best",
                "fold": fold,
                "model": "MLP-PCA8",
                "balanced_accuracy": best_mlp,
            }
        )

        svc = SVC(C=10.0, kernel="rbf", gamma="scale")
        svc.fit(x_tr_s, y_tr)
        rows.append(
            {
                "arm": "sample_efficiency",
                "model": "SVM-RBF",
                "fold": fold,
                "train_fraction": 1.0,
                "balanced_accuracy": float(balanced_accuracy_score(y_te, svc.predict(x_te_s))),
                "auc": float(roc_auc_score((y_te == MALIGNANT).astype(int), svc.decision_function(x_te_s))),
            }
        )
        print(f"  fold {fold}: plus-fidelity={fid:.3f} best_lr VQC={best_vqc:.3f} MLP={best_mlp:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize([r for r in rows if "balanced_accuracy" in r])
    payload = {
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "device": dev_name,
        "cuda": torch.cuda.is_available(),
        "summary": summary,
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(rows).to_csv(args.out_dir / "rows.csv", index=False)
    print(f"Wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

