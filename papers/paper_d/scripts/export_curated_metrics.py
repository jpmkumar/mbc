#!/usr/bin/env python3
"""Promote Paper D BreakHis summaries into curated JSON + a two-panel figure."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_vqc_stage_b_folds import corrected_interval

RAW = ROOT / "results" / "breakhis_paper_d"
OUT = ROOT / "papers" / "paper_d" / "results"
FIG = ROOT / "papers" / "paper_d" / "manuscript" / "figures"
ARMS = ("F-MLP", "F-VQC", "T-MLP", "T-VQC")
MARGIN = 0.01


def load_cells() -> dict:
    cells = {}
    for fold in range(5):
        for arm in ARMS:
            path = RAW / f"fold{fold}_{arm}" / "summary.json"
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            cells[(fold, arm)] = json.loads(path.read_text())
    return cells


def mean_std(values: list[float]) -> dict:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
        "values": [float(v) for v in values],
    }


def holm(p_values: list[tuple[str, float]]) -> dict[str, dict]:
    ordered = sorted(p_values, key=lambda x: x[1])
    m = len(ordered)
    out = {}
    hold = 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, hold)
        hold = adj
        out[name] = {"raw": p, "holm": adj}
    return out


def main() -> int:
    cells = load_cells()
    metrics = (
        "case_balanced_auprc",
        "auprc",
        "auc",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
    )
    per_arm = {}
    for arm in ARMS:
        per_arm[arm] = {
            key: mean_std([cells[(f, arm)]["test"][key] for f in range(5)])
            for key in metrics
        }
        per_arm[arm]["train_time_s"] = mean_std(
            [cells[(f, arm)]["train_time_s"] for f in range(5)]
        )

    n_test = [17, 17, 16, 16, 16]
    n_train = [82 - n for n in n_test]
    ratio = float(np.mean([t / tr for t, tr in zip(n_test, n_train)]))

    contrasts = {}
    p_pairs = []
    for name, vqc, mlp in (
        ("frozen", "F-VQC", "F-MLP"),
        ("finetune", "T-VQC", "T-MLP"),
    ):
        deltas = [
            cells[(f, vqc)]["test"]["case_balanced_auprc"]
            - cells[(f, mlp)]["test"]["case_balanced_auprc"]
            for f in range(5)
        ]
        interval = corrected_interval(deltas, ratio, 0.95)
        se = interval["corrected_standard_error"]
        t_stat = interval["mean"] / se if se else 0.0
        from scipy import stats as scipy_stats

        p_two = float(2 * scipy_stats.t.sf(abs(t_stat), 4))
        lo, hi = interval["lower"], interval["upper"]
        if hi < -MARGIN:
            verdict = "classical_gain"
        elif lo > MARGIN:
            verdict = "vqc_gain"
        elif lo >= -MARGIN and hi <= MARGIN:
            verdict = "no_effect_of_interest"
        else:
            verdict = "inconclusive"
        contrasts[name] = {
            "deltas": [float(d) for d in deltas],
            "corrected_interval_95": interval,
            "p_two_sided": p_two,
            "verdict": verdict,
        }
        p_pairs.append((name, p_two))

    holm_adj = holm(p_pairs)
    for name, rec in holm_adj.items():
        contrasts[name]["p_holm"] = rec["holm"]

    payload = {
        "n_cells": 20,
        "primary_metric": "case_balanced_auprc",
        "smallest_effect_of_interest": MARGIN,
        "test_train_ratio": ratio,
        "per_arm": per_arm,
        "per_fold": {
            f"{fold}_{arm}": {
                "test": cells[(fold, arm)]["test"],
                "best_epoch": cells[(fold, arm)]["best_epoch"],
                "best_val_case_balanced_auprc": cells[(fold, arm)][
                    "best_val_case_balanced_auprc"
                ],
            }
            for fold in range(5)
            for arm in ARMS
        },
        "coprimary": contrasts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "curated_metrics.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")

    FIG.mkdir(parents=True, exist_ok=True)
    folds = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    colours = {
        "F-MLP": "#1f4e79",
        "F-VQC": "#c45911",
        "T-MLP": "#2e7d32",
        "T-VQC": "#7b1fa2",
    }
    width = 0.18
    ax = axes[0]
    for i, arm in enumerate(ARMS):
        vals = [cells[(f, arm)]["test"]["case_balanced_auprc"] for f in range(5)]
        ax.bar(
            folds + (i - 1.5) * width,
            vals,
            width=width,
            label=arm,
            color=colours[arm],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_ylim(0.82, 1.0)
    ax.set_ylabel("Case-balanced AUPRC")
    ax.set_title("Held-out test, all four arms")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.axhline(0.95, color="0.7", linewidth=0.6, linestyle=":")

    ax = axes[1]
    frozen = contrasts["frozen"]["deltas"]
    fine = contrasts["finetune"]["deltas"]
    ax.plot(folds, frozen, "o-", color=colours["F-VQC"], label="Frozen (F-VQC − F-MLP)")
    ax.plot(folds, fine, "s-", color=colours["T-VQC"], label="Fine-tune (T-VQC − T-MLP)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(MARGIN, color="0.45", linestyle="--", linewidth=0.8)
    ax.axhline(-MARGIN, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_ylabel("Paired Δ AUPRC")
    ax.set_title("VQC minus matched MLP")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(-0.04, 0.04)

    fig_path = FIG / "fig02_breakhis_auprc.pdf"
    fig.savefig(fig_path, dpi=300)
    fig.savefig(FIG / "fig02_breakhis_auprc.png", dpi=300)
    plt.close(fig)
    print(f"wrote {fig_path.relative_to(ROOT)}")
    print("frozen", contrasts["frozen"]["verdict"], contrasts["frozen"]["p_holm"])
    print("finetune", contrasts["finetune"]["verdict"], contrasts["finetune"]["p_holm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
