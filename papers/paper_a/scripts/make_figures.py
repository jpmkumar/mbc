#!/usr/bin/env python3
"""Generate the manuscript figures from confirmatory_cv.json and oof_scores.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "confirmatory_cv.json"
OOF = ROOT / "results" / "oof_scores.json"
OUT = ROOT / "figures"

# Print-friendly palette (colourblind-safe-ish, no purple mega-theme)
C_BOX = "#F4F6F8"
C_EDGE = "#2C3E50"
C_CLASS = "#1B4F72"
C_MATCH = "#196F3D"
C_QUANT = "#922B21"
C_ACC = "#1F618D"
C_REC = "#B9770E"


def _box(ax, xy, w, h, text, *, fc=C_BOX, ec=C_EDGE, fontsize=8.5, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=C_EDGE,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=C_EDGE,
            shrinkA=1,
            shrinkB=1,
        )
    )


def make_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Column x positions
    x0, x1, x2, x3 = 0.3, 2.6, 5.3, 8.6
    w0, w1, w2, w3 = 2.0, 2.3, 2.8, 3.1
    h = 0.85

    _box(ax, (x0, 2.5), w0, h, "WBCD\n30 features\n$n=569$", weight="bold")
    _box(ax, (x1, 2.5), w1, h, "Fold-wise\nStandardScaler\n(+ PCA / $[0,\\pi]$)")

    # Branch heads
    y_top, y_mid, y_low, y_bot = 4.5, 3.15, 1.85, 0.55
    _box(ax, (x2, y_top), w2, h, "Classical 30-D\nSVM-RBF / RF / MLP", ec=C_CLASS)
    _box(ax, (x2, y_mid), w2, h, "PCA-8 → VQC-8\n+ matched MLP-PCA8", ec=C_QUANT)
    _box(ax, (x2, y_low), w2, h, "MLP $30\\!\\to\\!32\\!\\to\\!6$\n→ VQC-6 / MLP-H6", ec=C_QUANT)
    _box(ax, (x2, y_bot), w2, h, "PCA-4 → QSVM-ZZ\n+ matched SVM/MLP", ec=C_QUANT)

    _box(
        ax,
        (x3, 2.35),
        w3,
        1.15,
        "5-fold metrics\nbal.\u00a0acc., malignant\nrecall, ROC-AUC",
        weight="bold",
        fc="#EAF2F8",
    )

    # Arrows: data → preprocess
    _arrow(ax, (x0 + w0, 2.5 + h / 2), (x1, 2.5 + h / 2))
    # preprocess → branches
    cx = x1 + w1
    mid = 2.5 + h / 2
    for y in (y_top, y_mid, y_low, y_bot):
        _arrow(ax, (cx, mid), (x2, y + h / 2))
    # branches → metrics
    for y in (y_top, y_mid, y_low, y_bot):
        _arrow(ax, (x2 + w2, y + h / 2), (x3, 2.35 + 1.15 / 2))

    ax.text(
        6.0,
        5.75,
        "Train-fold fits only (no test leakage)",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
        color="#566573",
    )

    fig.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def make_performance(path: Path, summary: dict) -> None:
    # † marks quantum models (VQC / quantum kernel) — no background shading
    order = [
        ("SVM_RBF", "SVM-RBF"),
        ("RF", "RF"),
        ("MLP_30D", "MLP-30D"),
        ("MLP_PCA8", "MLP-PCA8"),
        ("VQC_8", "VQC-8†"),
        ("MLP_H6", "MLP-H6"),
        ("HYBRID_MLP_VQC", "Hybrid†"),
        ("QSVM_ZZ4", "QSVM-ZZ4†"),
    ]
    labels = [lab for _, lab in order]
    ba_m = np.array([summary[k]["balanced_accuracy"]["mean"] for k, _ in order])
    ba_s = np.array([summary[k]["balanced_accuracy"]["std"] for k, _ in order])
    rec_m = np.array([summary[k]["recall_malignant"]["mean"] for k, _ in order])
    rec_s = np.array([summary[k]["recall_malignant"]["std"] for k, _ in order])

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=300)
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.10, right=0.78)

    b1 = ax.bar(
        x - width / 2,
        ba_m,
        width,
        yerr=[ba_s, np.minimum(ba_s, 1.0 - ba_m)],  # clip upper bar at 1.0
        capsize=2.5,
        label="Balanced accuracy",
        color=C_ACC,
        edgecolor="white",
        linewidth=0.4,
        error_kw={"elinewidth": 0.9, "ecolor": "#34495E"},
    )
    b2 = ax.bar(
        x + width / 2,
        rec_m,
        width,
        yerr=[rec_s, np.minimum(rec_s, 1.0 - rec_m)],  # clip upper bar at 1.0
        capsize=2.5,
        label="Malignant recall",
        color=C_REC,
        edgecolor="white",
        linewidth=0.4,
        error_kw={"elinewidth": 0.9, "ecolor": "#34495E"},
    )

    ax.set_ylim(0.55, 1.05)
    ax.set_ylabel("Score (mean ± std)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#D5D8DC")

    fig.legend(
        handles=[b1, b2],
        labels=["Balanced accuracy", "Malignant recall"],
        loc="center left",
        bbox_to_anchor=(0.80, 0.55),
        ncol=1,
        frameon=False,
        fontsize=9,
        labelspacing=1.15,
        handlelength=1.8,
        handletextpad=0.8,
        borderaxespad=0.0,
    )
    # Footnote for dagger labels (outside axes, under legend column)
    fig.text(
        0.80,
        0.22,
        "† Quantum model\n(VQC or kernel)",
        fontsize=8,
        color="#566573",
        va="bottom",
        ha="left",
        linespacing=1.35,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    fig.savefig(
        path.with_suffix(".png"),
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
        pad_inches=0.15,
    )
    plt.close(fig)


def make_roc_pr(path: Path, oof: dict) -> None:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    # Pooled out-of-fold scores; malignant = positive (label 0)
    order = [
        ("SVM_RBF", "SVM-RBF", C_CLASS),
        ("MLP_PCA8", "MLP-PCA8", C_MATCH),
        ("VQC_8", "VQC-8", C_QUANT),
        ("HYBRID_MLP_VQC", "Hybrid", "#6C3483"),
        ("QSVM_ZZ4", "QSVM-ZZ4", "#B9770E"),
    ]
    pooled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, _, _ in order:
        ys, ps = [], []
        for row in oof["rows"]:
            if row["model"] != key:
                continue
            y = np.asarray(row["y_true"], dtype=int)
            ys.append((y == 0).astype(int))
            ps.append(np.asarray(row["p_mal"], dtype=float))
        pooled[key] = (np.concatenate(ys), np.concatenate(ps))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), dpi=300)
    fig.subplots_adjust(wspace=0.32, bottom=0.18, top=0.90, left=0.10, right=0.98)

    ax_roc, ax_pr = axes
    for key, lab, color in order:
        y, p = pooled[key]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        ax_roc.plot(fpr, tpr, color=color, lw=1.4, label=f"{lab} ({auc:.3f})")
        ax_pr.plot(rec, prec, color=color, lw=1.4, label=f"{lab} ({ap:.3f})")

    prev = float(np.mean(pooled["SVM_RBF"][0]))
    ax_roc.plot([0, 1], [0, 1], ls=":", lw=0.8, color="#95A5A6")
    ax_pr.axhline(prev, ls=":", lw=0.8, color="#95A5A6")

    ax_roc.set_xlabel("False-positive rate", fontsize=8)
    ax_roc.set_ylabel("True-positive rate", fontsize=8)
    ax_roc.set_title("ROC (malignant +)", fontsize=9)
    ax_pr.set_xlabel("Recall", fontsize=8)
    ax_pr.set_ylabel("Precision", fontsize=8)
    ax_pr.set_title("Precision–recall (malignant +)", fontsize=9)
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
    ax_roc.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax_pr.legend(frameon=False, fontsize=6.5, loc="lower left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(
        path.with_suffix(".png"),
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
        pad_inches=0.12,
    )
    plt.close(fig)


def main() -> None:
    data = json.loads(RESULTS.read_text())
    summary = data["summary"]
    OUT.mkdir(parents=True, exist_ok=True)
    make_pipeline(OUT / "fig01_pipeline.pdf")
    make_performance(OUT / "fig02_performance.pdf", summary)
    print(f"Wrote {OUT / 'fig01_pipeline.pdf'}")
    print(f"Wrote {OUT / 'fig02_performance.pdf'}")
    if OOF.exists():
        make_roc_pr(OUT / "fig03_roc_pr.pdf", json.loads(OOF.read_text()))
        print(f"Wrote {OUT / 'fig03_roc_pr.pdf'}")
    else:
        print(f"Skip Fig 3 (missing {OOF})")
    print("Also wrote PNG previews alongside PDFs.")


if __name__ == "__main__":
    main()
