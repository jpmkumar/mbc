#!/usr/bin/env python3
"""Paper D T&F secondaries S1–S3 from saved predictions + summaries.

S1: per-magnification case-balanced AUPRC
S2: ECE + reliability diagrams (F-MLP, F-VQC)
S3: wall-clock train_time_s + parameter counts

Does not retrain. Aligns test_predictions.npz to runtime test.csv row order.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

_TRAIN = ROOT / "papers" / "paper_d" / "scripts" / "train_breakhis_cv.py"
_spec = importlib.util.spec_from_file_location("train_breakhis_cv", _TRAIN)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ARMS = _mod.ARMS
build_model = _mod.build_model
case_balanced_auprc = _mod.case_balanced_auprc

RAW = ROOT / "results" / "breakhis_paper_d"
SPLITS = ROOT / "data" / "splits" / "breakhis" / "runtime"
OUT = ROOT / "papers" / "paper_d" / "results"
FIG = ROOT / "papers" / "paper_d" / "manuscript" / "figures"
CONFIG = ROOT / "papers" / "paper_d" / "config" / "breakhis.yaml"
MAGS = ("40X", "100X", "200X", "400X")
ARM_LIST = ("F-MLP", "F-VQC", "T-MLP", "T-VQC")
CAL_ARMS = ("F-MLP", "F-VQC")
N_BINS = 15


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, dict]:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_conf, bin_acc, bin_frac = [], [], []
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        frac = float(mask.mean()) if n else 0.0
        if mask.any():
            conf = float(probs[mask].mean())
            acc = float(labels[mask].mean())
            ece += abs(acc - conf) * frac
        else:
            conf, acc = float("nan"), float("nan")
        bin_conf.append(conf)
        bin_acc.append(acc)
        bin_frac.append(frac)
    return float(ece), {
        "bin_edges": edges.tolist(),
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_fraction": bin_frac,
    }


def mean_std(values: list[float]) -> dict:
    vals = [float(v) for v in values]
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "values": vals}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)),
        "values": vals,
    }


def load_aligned(fold: int, arm: str) -> tuple[pd.DataFrame, dict]:
    csv = pd.read_csv(SPLITS / f"fold_{fold}" / "test.csv")
    path = RAW / f"fold{fold}_{arm}" / "test_predictions.npz"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    npz = np.load(path, allow_pickle=True)
    labels = np.asarray(npz["labels"])
    probs = np.asarray(npz["probs"])
    case_ids = np.asarray(npz["case_ids"]).astype(str)
    if len(csv) != len(labels):
        raise SystemExit(f"fold {fold} {arm}: csv {len(csv)} != preds {len(labels)}")
    if not np.all(csv["patient_id"].astype(str).values == case_ids):
        raise SystemExit(f"fold {fold} {arm}: case_id order mismatch vs test.csv")
    return csv, {"labels": labels, "probs": probs, "case_ids": case_ids}


def param_counts() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = {}
    for arm, spec in ARMS.items():
        model = build_model(config, spec["head"])
        total = sum(p.numel() for p in model.parameters())
        # Regime-aware trainable count (same rules as train_breakhis_cv.apply_regime)
        encoder = model.encoder
        if spec["regime"] == "frozen":
            for p in encoder.parameters():
                p.requires_grad = False
        else:
            n = len(encoder.features)
            freeze_until = max(0, n - int(config["training"].get("finetune_last_blocks", 1)))
            encoder.freeze_early(num_blocks=freeze_until)
        if spec["head"] == "vqc" and hasattr(model, "set_classical_head_trainable"):
            model.set_classical_head_trainable(False)
            model.set_vqc_head_trainable(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out[arm] = {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "regime": spec["regime"],
            "head": spec["head"],
        }
        del model
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    per_mag: dict = {arm: {m: [] for m in MAGS} for arm in ARM_LIST}
    ece_by_arm: dict = {arm: [] for arm in CAL_ARMS}
    reliability_pooled: dict = {}
    wall: dict = {arm: [] for arm in ARM_LIST}

    for fold in range(5):
        for arm in ARM_LIST:
            csv, pred = load_aligned(fold, arm)
            summary = json.loads(
                (RAW / f"fold{fold}_{arm}" / "summary.json").read_text(encoding="utf-8")
            )
            wall[arm].append(float(summary["train_time_s"]))
            for mag in MAGS:
                mask = (csv["magnification"].astype(str).str.upper() == mag).to_numpy()
                if mask.sum() == 0:
                    per_mag[arm][mag].append(float("nan"))
                    continue
                score = case_balanced_auprc(
                    pred["labels"][mask],
                    pred["probs"][mask],
                    pred["case_ids"][mask],
                )
                per_mag[arm][mag].append(score)

        for arm in CAL_ARMS:
            _, pred = load_aligned(fold, arm)
            ece, _ = expected_calibration_error(pred["labels"], pred["probs"])
            ece_by_arm[arm].append(ece)

    # Pooled reliability for F arms (all folds concatenated)
    for arm in CAL_ARMS:
        labs, probs = [], []
        for fold in range(5):
            _, pred = load_aligned(fold, arm)
            labs.append(pred["labels"])
            probs.append(pred["probs"])
        labs = np.concatenate(labs)
        probs = np.concatenate(probs)
        ece, bins = expected_calibration_error(labs, probs)
        reliability_pooled[arm] = {"ece_pooled": ece, **bins}

    params = param_counts()

    payload = {
        "amendment_date": "2026-09-22",
        "venue": "Connection Science",
        "s1_per_magnification": {
            arm: {mag: mean_std(vals) for mag, vals in mags.items()}
            for arm, mags in per_mag.items()
        },
        "s2_calibration": {
            "n_bins": N_BINS,
            "per_fold_ece": {arm: mean_std(vals) for arm, vals in ece_by_arm.items()},
            "pooled": reliability_pooled,
        },
        "s3_cost": {
            "train_time_s": {arm: mean_std(vals) for arm, vals in wall.items()},
            "parameters": params,
        },
        "s4_idc_to_breakhis": {
            "status": "skipped",
            "reason": "optional; not required for Connection Science filing",
        },
    }
    dest = OUT / "tf_secondaries.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")

    # Fig 03 — per-magnification bars
    colours = {
        "F-MLP": "#1f4e79",
        "F-VQC": "#c45911",
        "T-MLP": "#2e7d32",
        "T-VQC": "#7b1fa2",
    }
    x = np.arange(len(MAGS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for i, arm in enumerate(ARM_LIST):
        means = [payload["s1_per_magnification"][arm][m]["mean"] for m in MAGS]
        stds = [payload["s1_per_magnification"][arm][m]["std"] for m in MAGS]
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            label=arm,
            color=colours[arm],
            edgecolor="black",
            linewidth=0.4,
            error_kw={"linewidth": 0.6},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(MAGS)
    ax.set_ylabel("Case-balanced AUPRC")
    ax.set_title("Per-magnification held-out AUPRC (mean ± fold std)")
    ax.set_ylim(0.75, 1.02)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.savefig(FIG / "fig03_per_magnification.pdf", dpi=300)
    fig.savefig(FIG / "fig03_per_magnification.png", dpi=300)
    plt.close(fig)
    print("wrote fig03_per_magnification")

    # Fig 04 — reliability diagrams
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, arm in zip(axes, CAL_ARMS):
        bins = reliability_pooled[arm]
        conf = np.asarray(bins["bin_confidence"], dtype=float)
        acc = np.asarray(bins["bin_accuracy"], dtype=float)
        mask = np.isfinite(conf) & np.isfinite(acc)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.plot(
            conf[mask],
            acc[mask],
            "o-",
            color=colours[arm],
            label=f"{arm} (ECE={bins['ece_pooled']:.3f})",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(arm)
        ax.legend(frameon=False, fontsize=7)
        ax.set_aspect("equal")
    fig.savefig(FIG / "fig04_reliability.pdf", dpi=300)
    fig.savefig(FIG / "fig04_reliability.png", dpi=300)
    plt.close(fig)
    print("wrote fig04_reliability")

    for arm in CAL_ARMS:
        print(arm, "ECE mean±std", mean_std(ece_by_arm[arm]))
    for arm in ARM_LIST:
        print(arm, "train_s", mean_std(wall[arm]), "params", params[arm])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
