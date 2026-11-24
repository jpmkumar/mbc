#!/usr/bin/env python3
"""VQC boost sweep — close the gap to MLP-PCA8 on WBCD.

Highest-leverage changes vs confirmatory VQC_8 (⟨Z0⟩ + MSE + L=2):
  - measure all ⟨Zi⟩ and a classical linear CE head
  - CE loss, more epochs
  - deeper SEL and optional data re-uploading

Matched controls: Linear-PCA8, MLP-PCA8 (same folds).
"""

from __future__ import annotations

import argparse
import json
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

ROOT = Path(__file__).resolve().parents[3]
MALIGNANT = 0


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def prep_pca8(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=8, random_state=0)
    tr = pca.fit_transform(x_tr)
    te = pca.transform(x_te)
    mm = MinMaxScaler(feature_range=(0.0, np.pi))
    return mm.fit_transform(tr), mm.transform(te)


def metrics(y_te: np.ndarray, pred: np.ndarray, p_mal: np.ndarray) -> dict:
    y_mal = (y_te == MALIGNANT).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "recall_malignant": float((pred == MALIGNANT)[y_te == MALIGNANT].mean())
        if (y_te == MALIGNANT).any()
        else 0.0,
        "auc": float(roc_auc_score(y_mal, p_mal)),
    }


class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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


class VQCBoost(nn.Module):
    """Angle embedding + SEL blocks (+ optional reupload) → multi-Z → linear CE."""

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_reupload: int = 1,
        readout: str = "allz",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_reupload = max(1, n_reupload)
        self.readout = readout
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            # weights: (n_reupload, n_layers, n_qubits, 3)
            for r in range(self.n_reupload):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[r], wires=range(n_qubits))
            if readout == "z0":
                return qml.expval(qml.PauliZ(0))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(
            qnode, {"weights": (self.n_reupload, n_layers, n_qubits, 3)}
        )
        out_dim = 1 if readout == "z0" else n_qubits
        self.head = nn.Linear(out_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.qlayer(x)
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        elif self.readout == "z0" and z.ndim == 1:
            z = z.view(-1, 1)
        if self.readout == "z0" and z.shape[-1] != 1:
            z = z.unsqueeze(-1)
        return self.head(z)


def train_torch_classifier(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    epochs: int,
    lr: float,
    seed: int,
    batch_size: int = 32,
) -> dict:
    set_seeds(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(xt))
        for start in range(0, len(xt), batch_size):
            b = idx[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_te, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    out = metrics(y_te, pred, prob[:, MALIGNANT])
    out["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return out


def train_vqc_z0_mse(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_layers: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """Legacy confirmatory-style: ⟨Z0⟩ + MSE (no classical head)."""
    set_seeds(seed)
    n_qubits = 8
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
    model = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    target = torch.tensor(2.0 * y_tr - 1.0, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(xt))
        for start in range(0, len(xt), 32):
            b = idx[start : start + 32]
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
    out = metrics(y_te, pred, p_mal)
    out["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "papers/paper_a/results/vqc_boost",
    )
    args = p.parse_args()

    data = load_breast_cancer()
    x = data.data.astype(np.float64)
    y = data.target.astype(int)
    n_splits = 2 if args.quick else 5
    epochs_short = 15 if args.quick else 30
    epochs_long = 25 if args.quick else 80

    configs = [
        ("baseline_z0_mse_L2", "legacy", dict(n_layers=2, epochs=epochs_short, lr=0.05)),
        ("vqc_allz_ce_L2", "boost", dict(n_layers=2, n_reupload=1, readout="allz", epochs=epochs_long, lr=1e-2)),
        ("vqc_allz_ce_L4", "boost", dict(n_layers=4, n_reupload=1, readout="allz", epochs=epochs_long, lr=1e-2)),
        ("vqc_reup2_allz_ce_L2", "boost", dict(n_layers=2, n_reupload=2, readout="allz", epochs=epochs_long, lr=1e-2)),
        ("vqc_z0_ce_head_L2", "boost", dict(n_layers=2, n_reupload=1, readout="z0", epochs=epochs_long, lr=1e-2)),
        ("linear_pca8", "classical", dict(epochs=epochs_long, lr=1e-2)),
        ("mlp_pca8", "classical", dict(epochs=epochs_long, lr=1e-3)),
    ]
    if args.quick:
        configs = [c for c in configs if c[0] in (
            "baseline_z0_mse_L2",
            "vqc_allz_ce_L2",
            "mlp_pca8",
            "linear_pca8",
        )]

    rows: list[dict] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    print(f"VQC boost | folds={n_splits} | pennylane={qml.__version__}")

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x[tr])
        x_te_s = scaler.transform(x[te])
        y_tr, y_te = y[tr], y[te]
        x_tr_q, x_te_q = prep_pca8(x_tr_s, x_te_s)

        for name, family, kw in configs:
            seed = args.seed + fold
            if name == "baseline_z0_mse_L2":
                m = train_vqc_z0_mse(
                    x_tr_q, y_tr, x_te_q, y_te,
                    n_layers=kw["n_layers"], epochs=kw["epochs"], lr=kw["lr"], seed=seed,
                )
            elif name == "linear_pca8":
                m = train_torch_classifier(
                    LinearHead(8), x_tr_q, y_tr, x_te_q, y_te,
                    epochs=kw["epochs"], lr=kw["lr"], seed=seed,
                )
            elif name == "mlp_pca8":
                m = train_torch_classifier(
                    MatchedMLP(8, 8), x_tr_q, y_tr, x_te_q, y_te,
                    epochs=kw["epochs"], lr=kw["lr"], seed=seed,
                )
            else:
                model = VQCBoost(
                    n_layers=kw["n_layers"],
                    n_reupload=kw["n_reupload"],
                    readout=kw["readout"],
                )
                m = train_torch_classifier(
                    model, x_tr_q, y_tr, x_te_q, y_te,
                    epochs=kw["epochs"], lr=kw["lr"], seed=seed,
                )
            row = {"model": name, "family": family, "fold": fold, **m}
            rows.append(row)
            print(
                f"  fold {fold} {name}: bal={m['balanced_accuracy']:.3f} "
                f"rec={m['recall_malignant']:.3f} auc={m['auc']:.3f} params={m['n_params']}"
            )

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "rows.csv", index=False)
    summary = {}
    for model, g in df.groupby("model"):
        summary[model] = {
            k: {"mean": float(g[k].mean()), "std": float(g[k].std(ddof=0)), "n": int(len(g))}
            for k in ("balanced_accuracy", "recall_malignant", "auc", "n_params")
            if k in g
        }
    payload = {
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "n_splits": n_splits,
        "quick": bool(args.quick),
        "summary": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out_dir / 'summary.json'}")
    # ranked by mean bal.acc
    ranked = sorted(
        ((k, v["balanced_accuracy"]["mean"]) for k, v in summary.items()),
        key=lambda t: -t[1],
    )
    print("Ranking (bal.acc mean):")
    for k, v in ranked:
        print(f"  {v:.3f}  {k}")


if __name__ == "__main__":
    main()
