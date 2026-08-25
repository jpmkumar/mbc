#!/usr/bin/env python3
"""Generate the histopath manuscript data figures.

Schematics are TikZ sources under figures/ and are built by
figures/build_figures.sh; the raster drafts kept here are deprecated.
"""

from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

PAPER_B = Path(__file__).resolve().parents[1]
ROOT = PAPER_B.parents[1]
FIG = PAPER_B / "figures" / "export"

# Figures are drawn once on a 372 pt design measure and shared by both target
# venues, which set them at different printed widths:
#   Frontiers in AI (primary)  \textwidth 504 pt (180 mm), 12 pt body
#   Springer sn-jnl (fallback) \textwidth 372 pt,          10 pt body
# In-figure type therefore prints at ~9.5 pt in Frontiers and ~7 pt in Springer,
# below body size in both. Set MBC_FIG_WIDTH_PT to redraw natively at another
# measure, which is what a production 180 mm export needs if Frontiers asks for
# one at acceptance.
PRINT_W = float(os.environ.get("MBC_FIG_WIDTH_PT", "372.0")) / 72.27
COL_W = PRINT_W
TEXT_W = PRINT_W
FS_TICK = 6.5
FS_LABEL = 7.5
FS_TITLE = 8.0
FS_LEGEND = 6.5

# Matplotlib defaults to Type 3 subsets in PDF, which several publishers reject
# outright and which some viewers render poorly; 42 selects TrueType instead.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Every series that is distinguished by colour also carries a second, achromatic
# channel, so the figures survive greyscale printing and common colour-vision
# deficiencies: hatch patterns for bars, marker shape and dash pattern for lines.
# Index order matches the arm order E2, E2b, E3 used throughout.
ARM_HATCH = ["", "///", "xxx"]
ARM_DASH = ["-", "--", "-."]


def save_figure(fig, stem: str) -> Path:
    """Write the PNG preview and the vector PDF at exactly the drawn size.

    ``bbox_inches="tight"`` is deliberately not used: it retrims the canvas, so
    the PDF would no longer be exactly PRINT_W wide and pdflatex would rescale
    it again.
    """
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png

# Long arm names do not fit a single-column legend at a print-legible size, so
# the compact form is used in-figure and the full form stays in the captions.
SHORT_ARM = {
    "E2 (linear)": "E2 linear",
    "E2b (capacity-control MLP)": "E2b MLP",
    "E3 (staged; Stage A selected)": "E3 staged",
}


def tidy(ax, grid_axis: str = "y") -> None:
    ax.tick_params(labelsize=FS_TICK, length=3, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle=":", alpha=0.4)


def load_arm(arm_glob_key: str, exp_key: str) -> list[dict]:
    rows: list[dict] = []
    for fold in range(5):
        pattern_dirs = list((ROOT / "results").glob(f"histopath_kaggle_fold{fold}_{arm_glob_key}*"))
        if arm_glob_key in ("e2_v2", "e3_v2"):
            pattern_dirs = [d for d in pattern_dirs if arm_glob_key in d.name]
        elif arm_glob_key == "e2b":
            pattern_dirs = [d for d in pattern_dirs if "_e2b" in d.name]
        mets = None
        for d in pattern_dirs:
            for p in d.rglob("cv_summary.json"):
                s = json.loads(p.read_text())
                results = s.get("results") or {}
                arr = results.get(exp_key)
                if not arr and results:
                    arr = next(iter(results.values()))
                if not arr:
                    continue
                t = arr[0].get("test_metrics") or arr[0].get("train_metrics")
                if t and "f1" in t:
                    mets = t
                    break
            if mets:
                break
        if not mets:
            raise SystemExit(f"Missing {exp_key} fold {fold}")
        rows.append(mets)
    return rows


def series(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def plot_fig5(arms: dict[str, list[dict]], colors: list[str]) -> Path:
    metrics = ["balanced_accuracy", "f1", "auprc"]
    metric_labels = ["Balanced\naccuracy", "F1", "AUPRC"]
    fig, ax = plt.subplots(figsize=(PRINT_W, 2.5), dpi=600)
    x = np.arange(len(metrics))
    width = 0.25
    for i, (name, rows) in enumerate(arms.items()):
        means = [st.mean(series(rows, m)) for m in metrics]
        stds = [st.stdev(series(rows, m)) for m in metrics]
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            label=SHORT_ARM[name],
            color=colors[i],
            edgecolor="black",
            hatch=ARM_HATCH[i],
            linewidth=0.5,
            capsize=2,
            error_kw={"linewidth": 0.7},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.74, 0.96)
    ax.set_ylabel("Test score, five-fold mean", fontsize=FS_LABEL)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=FS_LEGEND,
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.1)
    tidy(ax)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, "fig5_histopath_cv_means")


def plot_fig6(arms: dict[str, list[dict]], colors: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(PRINT_W, 2.5), dpi=600)
    folds = np.arange(5)
    markers = ["o", "s", "^"]
    for i, (name, rows) in enumerate(arms.items()):
        ax.plot(
            folds,
            series(rows, "f1"),
            marker=markers[i],
            linestyle=ARM_DASH[i],
            color=colors[i],
            label=SHORT_ARM[name],
            linewidth=1.3,
            markersize=4.5,
        )
    ax.set_xticks(folds)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_ylim(0.72, 0.88)
    ax.set_xlabel("Cross-validation fold", fontsize=FS_LABEL)
    ax.set_ylabel("Test F1", fontsize=FS_LABEL)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=FS_LEGEND,
              handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.1)
    tidy(ax)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, "fig6_histopath_perfold_f1")


def plot_fig1() -> Path:
    """Deprecated raster draft. Canonical Fig.1 is figures/Fig01_Architecture.tikz."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Hybrid histopathology pipeline: common backbone with staged output routes",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    def box(x, y, w, h, text, facecolor, fontsize=8.5):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#222222",
            facecolor=facecolor,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4),
        )

    box(0.02, 0.42, 0.11, 0.22, "IDC patch\n224×224", "#eef2f7", 8)
    arrow(0.13, 0.53, 0.16, 0.53)
    box(0.16, 0.38, 0.14, 0.30, "EfficientNet-B0\n1280-D", "#d6eaf8", 8)
    arrow(0.30, 0.53, 0.33, 0.53)
    box(0.33, 0.38, 0.14, 0.30, "Transformer\n2L / 4H\n2048-D", "#d5f5e3", 8)
    arrow(0.47, 0.53, 0.50, 0.53)
    box(0.50, 0.38, 0.13, 0.30, "Compression\n2048→128→\n32→8", "#fcf3cf", 8)

    ax.text(
        0.82,
        0.94,
        "Classification heads\n(only this block differs)",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    box(0.67, 0.72, 0.30, 0.12, "E2  Linear(8→2)", "#aed6f1", 8.5)
    box(0.67, 0.55, 0.30, 0.12, "E2b  LN→Linear→GELU→Linear", "#82e0aa", 8.2)
    box(0.67, 0.38, 0.30, 0.12, "E3  VQC (8q, 2L) → Linear", "#f5b041", 8.5)
    box(0.67, 0.21, 0.30, 0.12, "E4  α·MLP + (1−α)·VQC  (optional)", "#e59866", 8)
    for y in [0.78, 0.61, 0.44, 0.27]:
        ax.annotate(
            "",
            xy=(0.67, y),
            xytext=(0.63, 0.53),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9),
        )
    box(0.67, 0.04, 0.30, 0.12, "Softmax + val-tuned Fβ\n(+ TTA at test)", "#f5eef8", 8)
    ax.annotate(
        "",
        xy=(0.82, 0.16),
        xytext=(0.82, 0.21),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    )
    ax.text(
        0.02,
        0.12,
        "Case-ID-grouped 5-fold StratifiedGroupKFold · focal loss · strong aug · TTA · Fβ (β=1.5)",
        fontsize=8,
        style="italic",
        color="#444444",
    )
    out = FIG / "fig1_histopath_architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig1_histopath_architecture.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _box(ax, x, y, w, h, text, fc, fs=8.5, ec="#222"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def plot_fig2() -> Path:
    """Deprecated raster draft. Canonical Fig.3 is figures/Fig02_VQC_Head.tikz."""
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Variational quantum classification head (default E3): 8 qubits, 2 layers",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    _box(ax, 0.02, 0.38, 0.12, 0.24, "8-D embedding\n(from compression)", "#fcf3cf", 8)
    _box(ax, 0.17, 0.38, 0.12, 0.24, "LayerNorm\n+ angle-Y\nencoding", "#d6eaf8", 8)
    panel = FancyBboxPatch(
        (0.32, 0.12),
        0.42,
        0.76,
        boxstyle="round,pad=0.01",
        linewidth=1.3,
        edgecolor="#444",
        facecolor="#fbfcfd",
    )
    ax.add_patch(panel)
    ax.text(0.53, 0.84, "Hardware-efficient ansatz", ha="center", fontsize=9, fontweight="bold")
    ys = np.linspace(0.72, 0.22, 4)
    labels = ["q0", "q1", "…", "q7"]
    for i, (y, lab) in enumerate(zip(ys, labels)):
        ax.plot([0.36, 0.70], [y, y], color="#333", lw=1.2)
        ax.text(0.34, y, lab, ha="right", va="center", fontsize=8, fontfamily="monospace")
        for x0 in (0.40, 0.56):
            if lab == "…":
                continue
            _box(ax, x0, y - 0.035, 0.055, 0.07, "RY", "#f9e79f", 6.5)
            _box(ax, x0 + 0.06, y - 0.035, 0.055, 0.07, "RZ", "#fad7a0", 6.5)
        if i < 3 and labels[i] != "…" and labels[i + 1] != "…":
            y2 = ys[i + 1]
            for xc in (0.52, 0.68):
                ax.plot([xc, xc], [y, y2], color="#1a5276", lw=1.0)
                ax.plot(xc, y, "o", color="#1a5276", markersize=5)
                ax.plot(xc, y2, "o", color="#1a5276", markersize=8, fillstyle="none", mew=1.2)
        elif lab == "…":
            ax.text(
                0.53,
                y,
                "linear CNOT · layer ×2",
                ha="center",
                va="center",
                fontsize=7,
                style="italic",
                color="#555",
            )
    ax.text(0.53, 0.15, "Layer 1                         Layer 2", ha="center", fontsize=8, color="#333")
    _box(ax, 0.77, 0.38, 0.10, 0.24, "Pauli-Z\n⟨Z⟩×8", "#d5f5e3", 8)
    _box(ax, 0.89, 0.38, 0.09, 0.24, "Linear\n8→2", "#aed6f1", 8)
    for x1, x2 in [(0.14, 0.17), (0.29, 0.32), (0.74, 0.77), (0.87, 0.89)]:
        ax.annotate(
            "",
            xy=(x2, 0.50),
            xytext=(x1, 0.50),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.3),
        )
    ax.text(
        0.02,
        0.06,
        "Default: encoding=angle_Y · entanglement=linear · n_layers=2 · n_qubits=8 · "
        "no data re-uploading · PennyLane default.qubit + backprop",
        fontsize=7.5,
        style="italic",
        color="#444",
    )
    out = FIG / "fig2_histopath_vqc.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig2_histopath_vqc.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_fig3() -> Path:
    """Deprecated raster draft. Canonical Fig.4 is figures/Fig04_Stages.tikz."""
    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Staged hybrid training and checkpoint selection", fontsize=12, fontweight="bold", pad=10)
    stages = [
        (
            0.04,
            "Stage A — Classical warmup",
            "Train backbone + classical head\n(E2 / E2b path; E3/E4 warmup)\n≤25 epochs · early stop",
            "#d6eaf8",
            "TRAINABLE backbone\n+ classical head",
        ),
        (
            0.36,
            "Stage B — VQC only",
            "Freeze backbone · cache features\nTrain VQC / fusion quantum path\n≤15 epochs · early stop",
            "#fdebd0",
            "FROZEN backbone\nTRAINABLE VQC",
        ),
        (
            0.68,
            "Stage C — Joint fine-tune",
            "Unfreeze lightly · lower LR\nShort joint update\n≤3 epochs",
            "#d5f5e3",
            "Both paths\n(low LR)",
        ),
    ]
    for x, title, body, fc, tag in stages:
        _box(ax, x, 0.38, 0.28, 0.42, "", fc, 8)
        ax.text(x + 0.14, 0.72, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + 0.14, 0.55, body, ha="center", va="center", fontsize=7.8)
        ax.text(x + 0.14, 0.42, tag, ha="center", va="center", fontsize=7.5, style="italic", color="#1a5276")
    for x1, x2 in [(0.32, 0.36), (0.64, 0.68)]:
        ax.annotate(
            "",
            xy=(x2, 0.59),
            xytext=(x1, 0.59),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.6),
        )
    _box(ax, 0.20, 0.08, 0.60, 0.20, "", "#f5eef8", 8)
    ax.text(0.50, 0.22, "Model selection (all hybrid arms)", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        0.50,
        0.14,
        "Keep checkpoint with best validation balanced accuracy across stages (best_stage).\n"
        "In all E3 folds (and E4 / ablations on fold 0), best_stage = stage_a.",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.annotate(
        "",
        xy=(0.50, 0.28),
        xytext=(0.50, 0.38),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.3),
    )
    ax.text(
        0.02,
        0.02,
        "Adam · focal loss (γ=2) · selection_metric=balanced_accuracy · LR 1e-4 (A/B), 1e-5 (C)",
        fontsize=7.5,
        style="italic",
        color="#444",
    )
    out = FIG / "fig3_histopath_training_stages.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig3_histopath_training_stages.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_fig4() -> Path:
    """Deprecated raster draft. Canonical Fig.2 is figures/Fig03_PatientCV.tikz."""
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Case-ID-grouped stratified 5-fold cross-validation",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    _box(ax, 0.05, 0.72, 0.90, 0.18, "", "#eef2f7", 8)
    ax.text(
        0.50,
        0.86,
        "Public case-ID pool — 279 identifiers · 277,524 patches",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.78,
        "Bin patients by IDC-positive patch ratio → StratifiedGroupKFold (groups = patient IDs)",
        ha="center",
        fontsize=8,
    )
    fold_y = [0.58, 0.46, 0.34, 0.22, 0.10]
    for i, y in enumerate(fold_y):
        ax.text(0.02, y + 0.03, f"Fold {i}", fontsize=8, fontweight="bold", va="center")
        _box(ax, 0.12, y, 0.50, 0.08, "Train patients (patches only from these IDs)", "#82e0aa", 6.5)
        _box(ax, 0.63, y, 0.14, 0.08, "Val", "#f9e79f", 7)
        _box(ax, 0.78, y, 0.17, 0.08, "Test (held out)", "#f5b7b1", 7)
    _box(ax, 0.12, 0.01, 0.83, 0.07, "", "#fbfcfd", 7)
    ax.text(
        0.535,
        0.045,
        "Rule: no patient appears in more than one of {train, val, test} within a fold. "
        "Seed 42 · manifests archived as CSV.",
        ha="center",
        va="center",
        fontsize=7.8,
    )
    ax.text(0.12, 0.68, "■ train", color="#1e8449", fontsize=8, fontweight="bold")
    ax.text(0.22, 0.68, "■ val (from train pool)", color="#b7950b", fontsize=8, fontweight="bold")
    ax.text(0.42, 0.68, "■ test (never in train/val)", color="#922b21", fontsize=8, fontweight="bold")
    out = FIG / "fig4_histopath_patient_cv.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig4_histopath_patient_cv.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_fig7(probs_dir: Path | None = None) -> Path:
    """Fold-0 ROC and PR overlays for E2 / E2b / E3 from exported NPZ files."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    probs_dir = probs_dir or (ROOT / "results" / "fold0_probs")
    specs = [
        ("E2", "E2 linear", "#2c6eaf"),
        ("E2b", "E2b MLP", "#3a9d5c"),
        ("E3", "E3 staged", "#c45c26"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_W, 2.6), dpi=600)
    ax_roc, ax_pr = axes

    for i, (arm, label, color) in enumerate(specs):
        path = probs_dir / f"fold0_{arm}_test_probs.npz"
        if not path.is_file():
            raise SystemExit(f"Missing {path} — unpack mbc_fold0_probs_*.zip first")
        data = np.load(path)
        y = np.asarray(data["labels"], dtype=int)
        p = np.asarray(data["probs"], dtype=float)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        auc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        # The three curves nearly coincide, so the dash pattern, not the colour,
        # is what lets a reader separate them in print.
        dash = ARM_DASH[i]
        ax_roc.plot(fpr, tpr, color=color, ls=dash, lw=1.3, label=f"{label} (AUC={auc:.3f})")
        ax_pr.plot(rec, prec, color=color, ls=dash, lw=1.3, label=f"{label} (AUPRC={auprc:.3f})")

    ax_roc.plot([0, 1], [0, 1], ls=":", color="#888", lw=0.8)
    ax_roc.set_xlabel("False positive rate", fontsize=FS_LABEL)
    ax_roc.set_ylabel("True positive rate", fontsize=FS_LABEL)
    ax_roc.set_title("(A) ROC, fold-0 test", fontsize=FS_TITLE, fontweight="bold")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right", handlelength=1.6)
    tidy(ax_roc, grid_axis="both")

    prevalence = None
    e2 = probs_dir / "fold0_E2_test_probs.npz"
    if e2.is_file():
        y0 = np.asarray(np.load(e2)["labels"], dtype=float)
        prevalence = float(y0.mean())
        ax_pr.axhline(prevalence, ls=":", color="#888", lw=0.8, label=f"Prevalence ({prevalence:.2f})")
    ax_pr.set_xlabel("Recall", fontsize=FS_LABEL)
    ax_pr.set_ylabel("Precision", fontsize=FS_LABEL)
    ax_pr.set_title("(B) Precision–recall, fold-0 test", fontsize=FS_TITLE, fontweight="bold")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    # Opaque backing: the prevalence rule sits at 0.29 and would otherwise run
    # straight through the legend text.
    ax_pr.legend(
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        fontsize=FS_LEGEND,
        loc="lower left",
        handlelength=1.6,
    )
    tidy(ax_pr, grid_axis="both")

    fig.tight_layout(pad=0.5, w_pad=1.2)
    return save_figure(fig, "fig7_histopath_fold0_roc_pr")


def plot_fig8(probs_dir: Path | None = None) -> Path:
    """Fold-0 confusion matrices for E2 / E2b / E3 from exported NPZ files."""
    from sklearn.metrics import confusion_matrix

    probs_dir = probs_dir or (ROOT / "results" / "fold0_probs")
    specs = [
        ("E2", "E2 linear"),
        ("E2b", "E2b MLP"),
        ("E3", "E3 staged"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(PRINT_W, 2.15), dpi=600)
    for ax, (arm, title) in zip(axes, specs):
        path = probs_dir / f"fold0_{arm}_test_probs.npz"
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        data = np.load(path)
        y = np.asarray(data["labels"], dtype=int)
        p = np.asarray(data["probs"], dtype=float)
        thr = float(data["threshold"])
        pred = (p >= thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1], ["Pred\nbenign", "Pred\nIDC+"], fontsize=FS_TICK)
        ax.set_yticks([0, 1], ["True\nbenign", "True\nIDC+"], fontsize=FS_TICK)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({100 * cm_norm[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=FS_TICK,
                    color="white" if cm_norm[i, j] > 0.55 else "black",
                )
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        # Three panels across a 372 pt column leave about 1.7 in each, so the
        # title and axis label are shortened rather than scaled below 6.5 pt.
        ax.set_title(f"{title} (thr {thr:.2f})", fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel(f"sens {sens:.3f}, spec {spec:.3f}", fontsize=FS_TICK)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return save_figure(fig, "fig8_histopath_fold0_confusion")


STAGE_B_AGGREGATES = {
    "locked": {
        "primary": "vqc_stage_b_crossfold_v2_final.json",
        "sensitivity": "vqc_stage_b_crossfold_v2_sensitivity_folds1_4.json",
    },
    "nested": {
        "primary": "vqc_stage_b_crossfold_v3_final.json",
        "sensitivity": "vqc_stage_b_crossfold_v3_sensitivity_folds1_4.json",
    },
}


def load_stage_b() -> dict[str, dict[str, dict]]:
    base = ROOT / "results" / "histopath"
    out: dict[str, dict[str, dict]] = {}
    for protocol, files in STAGE_B_AGGREGATES.items():
        out[protocol] = {}
        for scope, name in files.items():
            path = base / name
            if not path.is_file():
                raise SystemExit(f"Missing Stage-B aggregate {path}")
            out[protocol][scope] = json.loads(path.read_text())
    return out


def plot_fig9_stage_b() -> Path:
    """Frozen-head Stage-B comparison: VQC minus MLP test AUPRC under both LR protocols."""
    agg = load_stage_b()
    margin = 0.01
    mlp_color = "#2c6eaf"
    vqc_color = "#c45c26"
    # Drawn at the Springer single-column text width so the PDF is placed at
    # scale 1 and in-figure type keeps its nominal point size in print. Three
    # panels abreast would leave 1.7 in each, which the fold labels and the
    # interval annotations cannot take, so the equivalence panel gets its own row.
    fig = plt.figure(figsize=(PRINT_W, 4.3), dpi=600)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.55, wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(labelsize=6.5, length=3, width=0.7)

    folds = [f["fold"] for f in agg["locked"]["primary"]["folds"]]
    x = np.arange(len(folds))

    ax_a.axhspan(-margin, margin, color="#e8eaed", zorder=0)
    ax_a.axhline(0.0, color="#555", lw=0.9, ls="-", zorder=1)
    specs_a = [
        ("locked", "Locked LR (primary)", vqc_color, "o", "--"),
        ("nested", "Nested LR (secondary)", mlp_color, "s", "-"),
    ]
    for protocol, label, color, marker, ls in specs_a:
        deltas = [f["mean_delta"] for f in agg[protocol]["primary"]["folds"]]
        ax_a.plot(x, deltas, marker=marker, ls=ls, color=color, lw=1.3, markersize=4.5, label=label, zorder=3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"F{f}" for f in folds])
    ax_a.set_ylim(-0.014, 0.021)
    ax_a.set_xlabel("Cross-validation fold", fontsize=7.5)
    ax_a.set_ylabel("VQC − MLP test AUPRC", fontsize=7.5)
    ax_a.set_title("(A) Per-fold frozen-head gap", fontsize=8, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=2.2)
    ax_a.text(
        0.97,
        0.03,
        f"shaded: ±{margin:.2f} margin",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        style="italic",
        color="#555",
    )

    width = 0.27
    # The two MLP series must be separable in greyscale as well as colour, so
    # they differ in fill lightness and hatch rather than in hue alone.
    bars = [
        (-1, [f["seed_stability"]["mlp"]["spread"] for f in agg["locked"]["primary"]["folds"]],
         "MLP, locked LR", mlp_color, None),
        (0, [f["seed_stability"]["mlp"]["spread"] for f in agg["nested"]["primary"]["folds"]],
         "MLP, nested LR", "#a8c4e0", "///"),
        (1, [f["seed_stability"]["vqc"]["spread"] for f in agg["locked"]["primary"]["folds"]],
         "VQC (LR unchanged)", vqc_color, None),
    ]
    for offset, values, label, color, hatch in bars:
        ax_b.bar(
            x + offset * width,
            values,
            width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
    ax_b.axhline(margin, color="#555", lw=0.8, ls=":")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"F{f}" for f in folds])
    ax_b.set_ylim(0, 0.047)
    ax_b.set_xlabel("Cross-validation fold", fontsize=7.5)
    ax_b.set_ylabel("Seed spread in test AUPRC", fontsize=7.5)
    ax_b.set_title("(B) Convergence reliability", fontsize=8, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.4, handletextpad=0.5)

    rows = [
        ("nested", "sensitivity", "Nested LR\nfolds 1–4", mlp_color),
        ("nested", "primary", "Nested LR\nall folds", mlp_color),
        ("locked", "sensitivity", "Locked LR\nfolds 1–4", vqc_color),
        ("locked", "primary", "Locked LR\nall folds", vqc_color),
    ]
    ax_c.axvspan(-margin, margin, color="#e8eaed", zorder=0)
    ax_c.axvline(0.0, color="#555", lw=0.8, zorder=1)
    for i, (protocol, scope, label, color) in enumerate(rows):
        d = agg[protocol][scope]
        ci = d["corrected_interval_90"]
        equivalent = d["equivalence_test"]["equivalent"]
        ax_c.errorbar(
            ci["mean"],
            i,
            xerr=[[ci["mean"] - ci["lower"]], [ci["upper"] - ci["mean"]]],
            fmt="s" if protocol == "nested" else "o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
            markersize=4.5,
            zorder=3,
        )
        ax_c.text(
            0.0125,
            i + 0.30,
            f"$p$={d['equivalence_test']['p_value']:.3f}, "
            f"{'equivalent' if equivalent else 'inconclusive'}",
            va="bottom",
            fontsize=6,
            color=color,
        )
    ax_c.set_yticks(range(len(rows)))
    ax_c.set_yticklabels([r[2] for r in rows], fontsize=6.5)
    ax_c.set_ylim(-0.6, len(rows) - 0.25)
    ax_c.set_xlim(-0.013, 0.036)
    ax_c.set_xticks([-0.01, 0.0, 0.01, 0.02, 0.03])
    ax_c.set_xlabel("Pooled VQC − MLP AUPRC (corrected 90% CI)", fontsize=7.5)
    ax_c.set_title("(C) Cross-fold equivalence test", fontsize=8, fontweight="bold")

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y" if ax is not ax_c else "x", linestyle=":", alpha=0.4, zorder=0)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.94, bottom=0.09)
    return save_figure(fig, "fig9_histopath_stage_b_equivalence")


def plot_fig10_width_matrix() -> Path:
    """Circuit-width matrix: per-cell scores and the paired width contrasts.

    Panel (A) is the twelve-cell server matrix, the only arm whose q8 reference
    was rerun inside the same immutable image, so its contrasts are paired
    within fold on identical test patients. Panel (B) puts both arms on the
    pre-declared decision scale: a contrast is a difference only if the
    corrected 95% interval excludes zero *and* the mean exceeds the +-0.01
    AUPRC practical margin, and it is equivalence only if the corrected 90%
    interval sits inside that margin. Every interval here does neither, which
    is what the panel is drawn to show.
    """
    data = json.loads(
        (ROOT / "results/histopath/width_matrix_analysis.json").read_text()
    )
    margin = data["practical_auprc_margin"]
    server = data["server_primary"]
    folds = [1, 2, 3, 4]
    widths = [4, 8, 12]
    marks = {4: ("o", "#2c6eaf"), 8: ("s", "#3a9d5c"), 12: ("^", "#c45c26")}

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(PRINT_W, 2.30), dpi=600, gridspec_kw={"width_ratios": [1.0, 1.22]}
    )

    # ---------------------------------------------------------------- panel (A)
    # Folds are separate partitions, so widths are dodged within each fold and
    # never joined across folds; the light rule shows the within-fold spread.
    offsets = {4: -0.16, 8: 0.0, 12: 0.16}
    for fold in folds:
        scores = [server["selected_auprc_by_width"][f"q{w}"][f"fold{fold}"]
                  for w in widths]
        ax_a.plot([fold, fold], [min(scores), max(scores)],
                  color="#9e9e9e", linewidth=0.7, zorder=2)
    for width in widths:
        marker, color = marks[width]
        xs = [fold + offsets[width] for fold in folds]
        scores = [server["selected_auprc_by_width"][f"q{width}"][f"fold{f}"]
                  for f in folds]
        ax_a.plot(xs, scores, marker=marker, color=color, markersize=3.4,
                  linestyle="none", label=f"{width} qubits", zorder=3)

    # A hollow ring flags the two cells whose validation-selected checkpoint was
    # the quantum stage; every other cell deployed the classical Stage A route.
    for cell in server["quantum_stage_evidence"]["cells_selecting_quantum_stage"]:
        score = server["selected_auprc_by_width"][f"q{cell['width']}"][f"fold{cell['fold']}"]
        ax_a.plot(cell["fold"] + offsets[cell["width"]], score, marker="o",
                  markersize=7.0, markerfacecolor="none",
                  markeredgecolor="#1b1b1b", markeredgewidth=0.8,
                  linestyle="none", zorder=4,
                  label="_ring" if cell is not
                  server["quantum_stage_evidence"]["cells_selecting_quantum_stage"][0]
                  else "quantum stage selected")

    ax_a.set_xticks(folds)
    ax_a.set_xlim(0.55, 4.45)
    ax_a.set_xlabel("Fold", fontsize=FS_LABEL)
    ax_a.set_ylabel("Test AUPRC", fontsize=FS_LABEL)
    ax_a.set_title("(A) Server matrix, selected checkpoint",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_a.set_ylim(0.845, 0.945)
    ax_a.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left",
                handlelength=1.0, borderpad=0.2, labelspacing=0.22,
                handletextpad=0.4, ncol=2, columnspacing=0.8)
    tidy(ax_a)

    # ---------------------------------------------------------------- panel (B)
    rows = [
        ("Server, 4 vs 8", server["contrasts"]["q4_minus_q8"], "#2c6eaf"),
        ("Server, 12 vs 8", server["contrasts"]["q12_minus_q8"], "#c45c26"),
        ("Kaggle, 4 vs 8", data["kaggle_secondary"]["contrasts"]["q4_minus_q8"], "#2c6eaf"),
        ("Kaggle, 12 vs 8", data["kaggle_secondary"]["contrasts"]["q12_minus_q8"], "#c45c26"),
    ]
    ax_b.axvspan(-margin, margin, color="#d9d9d9", alpha=0.55, zorder=0)
    ax_b.axvline(0.0, color="#1b1b1b", linewidth=0.8, zorder=1)

    for index, (label, contrast, color) in enumerate(rows):
        y = len(rows) - 1 - index
        interval = contrast["aggregate"]["corrected_interval_90"]
        mean = contrast["mean_delta_auprc"]
        ax_b.plot([interval["lower"], interval["upper"]], [y, y],
                  color=color, linewidth=1.3, solid_capstyle="butt", zorder=3)
        ax_b.plot([mean], [y], marker="D", markersize=3.6, color=color, zorder=4)
        for bound in (interval["lower"], interval["upper"]):
            ax_b.plot([bound, bound], [y - 0.13, y + 0.13],
                      color=color, linewidth=1.3, zorder=3)
        n_folds = len(contrast["folds"])
        ax_b.text(-0.043, y + 0.17, f"{label} ({n_folds} folds)",
                  fontsize=FS_LEGEND, ha="left", va="bottom")

    ax_b.set_yticks([])
    ax_b.set_ylim(-0.6, len(rows) - 0.05)
    ax_b.set_xlim(-0.045, 0.045)
    ax_b.set_xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    ax_b.set_xlabel("Width $-$ 8-qubit AUPRC (corrected 90% CI)",
                    fontsize=FS_LABEL)
    ax_b.set_title("(B) Paired width contrasts", fontsize=FS_TITLE,
                   fontweight="bold")
    ax_b.text(0.0, -0.47, f"shaded: $\\pm${margin:g} practical margin",
              fontsize=FS_LEGEND, ha="center", va="center", color="#4d4d4d")
    ax_b.tick_params(labelsize=FS_TICK, length=3, width=0.7)
    for side in ("top", "right", "left"):
        ax_b.spines[side].set_visible(False)
    ax_b.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    fig.subplots_adjust(left=0.105, right=0.995, top=0.90, bottom=0.155,
                        wspace=0.16)
    return save_figure(fig, "fig10_histopath_width_matrix")


def main() -> None:
    """Regenerate the data figures.

    The four schematics (architecture, VQC head, staged training, patient CV)
    are no longer produced here. Their canonical sources are the TikZ files
    Fig01_Architecture, Fig02_VQC_Head, Fig03_PatientCV and Fig04_Stages, built
    by figures/build_figures.sh. The raster drafts below are kept only for
    reference and are not included by the manuscript.
    """
    FIG.mkdir(exist_ok=True)
    arms = {
        "E2 (linear)": load_arm("e2_v2", "E2"),
        "E2b (capacity-control MLP)": load_arm("e2b", "E2b"),
        "E3 (staged; Stage A selected)": load_arm("e3_v2", "E3"),
    }
    colors = ["#2c6eaf", "#3a9d5c", "#c45c26"]
    for name, rows in arms.items():
        ba = series(rows, "balanced_accuracy")
        print(
            f"{name}: bal_acc {st.mean(ba):.4f}±{st.stdev(ba):.4f} "
            f"F1 {st.mean(series(rows, 'f1')):.4f}"
        )
    print("Wrote", plot_fig5(arms, colors))
    print("Wrote", plot_fig6(arms, colors))
    print("Wrote", plot_fig7())
    print("Wrote", plot_fig8())
    print("Wrote", plot_fig9_stage_b())
    print("Wrote", plot_fig10_width_matrix())


if __name__ == "__main__":
    main()
