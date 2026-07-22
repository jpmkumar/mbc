#!/usr/bin/env python3
"""Generate publication figures and LaTeX/Markdown tables from results/publication_metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS_PATH = ROOT / "publication/publication_metrics.json"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "publication/tables"

# IEEE-friendly styling
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
    }
)

COLORS = {
    "baseline_stage_a": "#95a5a6",
    "benedetti_stage_b": "#e67e22",
    "enhanced_stage_a": "#2ecc71",
    "enhanced_stage_b": "#3498db",
}


def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


def _fmt(x: float, pct: bool = True) -> str:
    return f"{100 * x:.1f}\\%" if pct else f"{x:.3f}"


def plot_metric_comparison(data: dict, out: Path):
    exps = data["mammography_experiments"]
    labels = [e["label"] for e in exps]
    metrics = ["balanced_accuracy", "recall", "precision", "f1", "auc"]
    titles = ["Balanced Acc.", "Recall", "Precision", "F1", "AUC"]

    x = np.arange(len(labels))
    width = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: key clinical metrics
    ax = axes[0]
    for i, (m, title) in enumerate(zip(metrics[:3], titles[:3])):
        vals = [e["test"][m] for e in exps]
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=title,
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) CBIS-DDSM test set — clinical metrics")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: F1 and AUC by experiment (color-coded)
    ax = axes[1]
    ids = [e["id"] for e in exps]
    f1s = [e["test"]["f1"] for e in exps]
    aucs = [e["test"]["auc"] for e in exps]
    bar_colors = [COLORS.get(i, "#333") for i in ids]
    w = 0.35
    ax.bar(x - w / 2, f1s, w, label="F1", color=bar_colors, alpha=0.85)
    ax.bar(x + w / 2, aucs, w, label="AUC", color=bar_colors, alpha=0.55, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) F1 and AUC comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "fig_mammo_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: list, title: str, out_path: Path):
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Benign", "Pred Malignant"],
        yticklabels=["True Benign", "True Malignant"],
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    tn, fp, fn, tp = arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1]
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ax.text(
        0.5,
        -0.22,
        f"Sensitivity={sens:.1%}  |  Specificity={spec:.1%}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_panels(data: dict, out: Path):
    primary = next(
        e
        for e in data["mammography_experiments"]
        if e["id"] == data["recommended_reporting"]["primary_model"]
    )
    secondary = next(e for e in data["mammography_experiments"] if e["id"] == "enhanced_stage_b")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, exp, panel in zip(
        axes,
        [primary, secondary],
        ["(a) Stage A enhanced (recommended)", "(b) Stage B enhanced VQC"],
    ):
        arr = np.array(exp["test"]["confusion_matrix"])
        sns.heatmap(
            arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Benign", "Pred Malignant"],
            yticklabels=["True Benign", "True Malignant"],
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"{panel}\n{exp['label']}")
    plt.suptitle("CBIS-DDSM test confusion matrices (n=445)", y=1.02, fontsize=11)
    plt.tight_layout()
    fig.savefig(out / "fig_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    for exp in data["mammography_experiments"]:
        if "confusion_matrix" not in exp["test"]:
            continue
        slug = exp["id"]
        plot_confusion_matrix(
            exp["test"]["confusion_matrix"],
            f"{exp['label']} — test (n=445)",
            out / f"fig_confusion_{slug}.png",
        )


def plot_dataset_splits(data: dict, out: Path):
    ds = data["dataset"]
    fig, ax = plt.subplots(figsize=(6, 4))
    cats = ["Train", "Val", "Test", "Total"]
    vals = [ds["train"], ds["val"], ds["test"], ds["total_images"]]
    colors = ["#3498db", "#9b59b6", "#e74c3c", "#2c3e50"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.85)
    ax.set_ylabel("Images")
    ax.set_title("CBIS-DDSM mammography dataset splits")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30, str(v), ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.12)
    plt.tight_layout()
    fig.savefig(out / "fig_dataset_splits.png", bbox_inches="tight")
    plt.close(fig)


def plot_pilot_vs_imaging(data: dict, out: Path):
    pilot = data["pilot_wbcd_tabular"]["models"]
    enhanced = next(e for e in data["mammography_experiments"] if e["id"] == "enhanced_stage_a")

    labels = ["SVM\n(WBCD)", "Hybrid\n(WBCD)", "Stage A\n(CBIS-DDSM)"]
    acc = [
        pilot["SVM_RBF"]["accuracy"],
        pilot["Hybrid_MLP_VQC"]["accuracy"],
        enhanced["test"]["accuracy"],
    ]
    auc = [
        pilot["SVM_RBF"]["auc"],
        pilot["Hybrid_MLP_VQC"]["auc"],
        enhanced["test"]["auc"],
    ]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#8e44ad", alpha=0.9)
    ax.bar(x + w / 2, auc, w, label="AUC", color="#16a085", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tabular pilot (WBCD) vs mammography imaging (CBIS-DDSM)")
    ax.legend()
    ax.annotate(
        "Different tasks — not directly comparable",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "fig_pilot_vs_imaging.png", bbox_inches="tight")
    plt.close(fig)


def _draw_layer_diagram(
    out: Path,
    filename: str,
    title: str,
    layers: list[tuple[str, str, str]],
    legend_note: str,
):
    """Draw a vertical layer stack diagram.

    Each layer: (name, detail, state) where state is trainable|frozen|output.
    """
    state_style = {
        "trainable": ("#e8f8f0", "#27ae60", "solid"),
        "frozen": ("#f0f0f0", "#7f8c8d", "dashed"),
        "output": ("#eaf2fb", "#2980b9", "solid"),
    }

    n = len(layers)
    fig_h = max(7.5, 0.72 * n + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")

    box_w, box_h = 0.78, 0.52
    x0 = 0.11
    y_top = 0.92
    gap = 0.08

    ax.text(
        0.5,
        0.985,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    for i, (name, detail, state) in enumerate(layers):
        y = y_top - i * (box_h + gap)
        face, edge, linestyle = state_style.get(state, state_style["trainable"])
        patch = FancyBboxPatch(
            (x0, y - box_h),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=face,
            linestyle=linestyle,
        )
        ax.add_patch(patch)
        ax.text(
            x0 + box_w / 2,
            y - box_h / 2 + 0.07,
            name,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.text(
            x0 + box_w / 2,
            y - box_h / 2 - 0.08,
            detail,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
        if state == "trainable":
            tag = "TRAINABLE"
            tag_color = "#27ae60"
        elif state == "frozen":
            tag = "FROZEN"
            tag_color = "#7f8c8d"
        else:
            tag = "OUTPUT"
            tag_color = "#2980b9"
        ax.text(
            x0 + box_w - 0.02,
            y - 0.03,
            tag,
            ha="right",
            va="top",
            fontsize=7,
            color=tag_color,
            fontweight="bold",
        )

        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + box_w / 2, y - box_h - 0.005),
                    (x0 + box_w / 2, y - box_h - gap + 0.005),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.4,
                    color="#444444",
                )
            )

    ax.text(0.5, 0.04, legend_note, ha="center", va="bottom", fontsize=8.5, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)


def plot_stage_layer_diagrams(out: Path):
    """Stage A and Stage B layer architecture diagrams (separate figures)."""
    shared_backbone = [
        (
            "Input Image",
            "224×224 grayscale + CLAHE preprocessing",
            "trainable",
        ),
        (
            "Modality Token Embedding",
            "[MAMMO] / [ULTRA] / [THERMO] learnable tokens",
            "trainable",
        ),
        (
            "EfficientNet-B0 Encoder",
            "ImageNet-pretrained CNN → 1280-D feature vector",
            "trainable",
        ),
        (
            "Modality Transformer Encoder",
            "2 layers, 4 heads, token + feature → 2048-D",
            "trainable",
        ),
        (
            "Feature Compression MLP",
            "Linear+LayerNorm+ReLU+Dropout: 2048 → 128 → 32 → 8",
            "trainable",
        ),
    ]

    stage_a_layers = shared_backbone + [
        (
            "Classical Linear Head",
            "Fully connected 8 → 2 (benign / malignant)",
            "trainable",
        ),
        (
            "Classification Output",
            "Softmax probabilities — Stage A deployment head",
            "output",
        ),
    ]

    stage_b_layers = [
        (
            "Input Image",
            "224×224 grayscale + CLAHE preprocessing",
            "frozen",
        ),
        (
            "Modality Token Embedding",
            "[MAMMO] / [ULTRA] / [THERMO] tokens (Stage A weights)",
            "frozen",
        ),
        (
            "EfficientNet-B0 Encoder",
            "1280-D features — backbone frozen in Stage B",
            "frozen",
        ),
        (
            "Modality Transformer Encoder",
            "2048-D representation — weights frozen",
            "frozen",
        ),
        (
            "Feature Compression MLP",
            "2048 → 128 → 32 → 8 — weights frozen",
            "frozen",
        ),
        (
            "LayerNorm + Angle Encoding",
            "Normalize 8-D features; sigmoid × π → rotation angles",
            "trainable",
        ),
        (
            "Variational Quantum Circuit (VQC)",
            "8 qubits, 2 ansatz layers; RY/RZ + linear CNOT entanglement",
            "trainable",
        ),
        (
            "Pauli-Z Readout + Linear Classifier",
            "8 expectation values → linear 8 → 2 logits",
            "trainable",
        ),
        (
            "Classification Output",
            "Softmax probabilities — Stage B VQC head",
            "output",
        ),
    ]

    _draw_layer_diagram(
        out,
        "fig_stage_a_layers.png",
        "Stage A — Classical Hybrid Layer Stack (all backbone + head trainable)",
        stage_a_layers,
        "Stage A trains EfficientNet, Transformer, compression MLP, and classical head.",
    )
    _draw_layer_diagram(
        out,
        "fig_stage_b_layers.png",
        "Stage B — VQC Layer Stack (frozen backbone + trainable quantum head)",
        stage_b_layers,
        "Stage B freezes Stage A backbone; only LayerNorm, VQC ansatz, and classifier train.",
    )


def plot_training_pipeline(out: Path):
    """Two-stage training schematic."""
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    stages = [
        ("Stage A\nClassical head", "EfficientNet +\nTransformer", "#2ecc71"),
        ("Stage B\nVQC head", "Frozen backbone\n+ quantum", "#3498db"),
        ("Stage C\nJoint (opt.)", "Fine-tune all", "#95a5a6"),
    ]
    for i, (title, sub, color) in enumerate(stages):
        x = 0.12 + i * 0.32
        rect = plt.Rectangle((x, 0.35), 0.22, 0.45, facecolor=color, alpha=0.35, edgecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.11, 0.72, title, ha="center", va="center", fontweight="bold", fontsize=10)
        ax.text(x + 0.11, 0.48, sub, ha="center", va="center", fontsize=8)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 0.24, 0.57),
                xytext=(x + 0.30, 0.57),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            )
    ax.text(0.5, 0.12, "Best CBIS-DDSM result: Stage A enhanced (70.1% balanced accuracy)", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out / "fig_training_stages.png", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(data: dict, out: Path):
    exps = data["mammography_experiments"]
    ds = data["dataset"]

    # Table I — main mammography results
    lines = [
        "% Auto-generated by scripts/generate_publication.py",
        "\\begin{table}[!t]",
        "\\caption{CBIS-DDSM mammography classification on held-out test set ($n=445$). Enhanced Stage A uses CLAHE preprocessing and class-weighted training.}",
        "\\label{tab:mammo}",
        "\\centering",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Acc. & Bal. Acc. & Prec. & Rec. & F1 & AUC \\\\",
        "\\midrule",
    ]
    for e in exps:
        t = e["test"]
        lines.append(
            f"{e['label']} & "
            f"{_fmt(t['accuracy'])} & {_fmt(t['balanced_accuracy'])} & "
            f"{_fmt(t['precision'])} & {_fmt(t['recall'])} & "
            f"{_fmt(t['f1'])} & {_fmt(t['auc'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]

    # Table II — pilot vs imaging
    pilot = data["pilot_wbcd_tabular"]
    lines += [
        "\\begin{table}[!t]",
        "\\caption{EQML pilot study (WBCD tabular, $n=569$) vs CBIS-DDSM mammography (enhanced Stage A). Tasks differ in modality and difficulty.}",
        "\\label{tab:pilot}",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & Accuracy & F1 & AUC \\\\",
        "\\midrule",
    ]
    for name, key in [
        ("SVM (RBF)", "SVM_RBF"),
        ("VQC standalone", "VQC_standalone"),
        ("Hybrid MLP+VQC", "Hybrid_MLP_VQC"),
    ]:
        m = pilot["models"][key]
        lines.append(
            f"{name} & {_fmt(m['accuracy'])} & {_fmt(m['f1'])} & {_fmt(m['auc'])} \\\\"
        )
    best = next(e for e in exps if e["id"] == "enhanced_stage_a")
    t = best["test"]
    lines.append(
        f"Stage A enhanced (CBIS-DDSM) & {_fmt(t['accuracy'])} & {_fmt(t['f1'])} & {_fmt(t['auc'])} \\\\"
    )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # Table III — dataset
    lines += [
        "\\begin{table}[!t]",
        "\\caption{CBIS-DDSM dataset summary used for mammography experiments.}",
        "\\label{tab:dataset}",
        "\\centering",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Statistic & Count \\\\",
        "\\midrule",
        f"Total ROI images & {ds['total_images']} \\\\",
        f"Benign / Malignant & {ds['benign']} / {ds['malignant']} \\\\",
        f"Train / Val / Test & {ds['train']} / {ds['val']} / {ds['test']} \\\\",
        f"Image size & {ds['image_size']}$\\times${ds['image_size']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    tex = "\n".join(lines)
    (out / "publication_tables.tex").write_text(tex)

    # CSV for spreadsheet
    csv_lines = ["model,accuracy,balanced_accuracy,precision,recall,f1,auc,stage,preprocessing"]
    for e in exps:
        t = e["test"]
        csv_lines.append(
            f"\"{e['label']}\",{t['accuracy']:.4f},{t['balanced_accuracy']:.4f},"
            f"{t['precision']:.4f},{t['recall']:.4f},{t['f1']:.4f},{t['auc']:.4f},"
            f"{e.get('stage','')},\"{e.get('preprocessing', False)}\""
        )
    (out / "mammo_results.csv").write_text("\n".join(csv_lines) + "\n")


def write_markdown_summary(data: dict, out: Path):
    rec = data["recommended_reporting"]["primary_model"]
    best = next(e for e in data["mammography_experiments"] if e["id"] == rec)
    t = best["test"]
    cm = t["confusion_matrix"]

    md = f"""# Publication Results Summary

Auto-generated from `publication/publication_metrics.json`. Re-run: `python scripts/generate_publication.py`

## Primary result (recommended for paper)

**{best['label']}** on CBIS-DDSM test ($n={t['n_samples']}$):

| Metric | Value |
|--------|-------|
| Balanced accuracy | **{100*t['balanced_accuracy']:.1f}%** |
| Malignant recall (sensitivity) | **{100*t['recall']:.1f}%** |
| Precision | {100*t['precision']:.1f}% |
| F1 | {100*t['f1']:.1f}% |
| AUC | {100*t['auc']:.1f}% |

Confusion matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}

## Figures for manuscript

| File | Use in paper |
|------|----------------|
| `figures/fig_mammo_metrics_comparison.png` | Results section — metric bars |
| `figures/fig_confusion_matrices.png` | Results — Stage A vs B confusion |
| `figures/fig_dataset_splits.png` | Experimental setup |
| `figures/fig_pilot_vs_imaging.png` | Discussion — pilot bridge |
| `figures/fig_training_stages.png` | Method — training strategy |
| `figures/fig_stage_a_layers.png` | Method — Stage A layer stack |
| `figures/fig_stage_b_layers.png` | Method — Stage B layer stack |

## LaTeX

Include in `main.tex`:

```latex
\\input{{publication/tables/publication_tables.tex}}
```

Or copy tables from `publication/tables/publication_tables.tex`.

## Key claims (copy-ready)

> On CBIS-DDSM mammography (2,966 ROI images; 445-image test split), enhanced classical Stage A achieved **70.1% balanced accuracy**, **84.7% malignant recall**, and **0.763 AUC**, outperforming the initial pipeline (61.8%) and Benedetti VQC Stage B (69.0% balanced accuracy, 59.9% recall).

> Hybrid VQC Stage B did not exceed the classical head on test metrics; **Stage A enhanced is the recommended deployment model**.
"""
    (out / "PUBLICATION_RESULTS.md").write_text(md)


def main():
    if not METRICS_PATH.exists():
        raise SystemExit(f"Missing {METRICS_PATH}")

    data = load_metrics()
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(data, FIG_DIR)
    plot_confusion_panels(data, FIG_DIR)
    plot_dataset_splits(data, FIG_DIR)
    plot_pilot_vs_imaging(data, FIG_DIR)
    plot_training_pipeline(FIG_DIR)
    plot_stage_layer_diagrams(FIG_DIR)
    write_latex_tables(data, TABLE_DIR)
    write_markdown_summary(data, ROOT / "publication")

    print(f"Figures  -> {FIG_DIR}")
    print(f"Tables   -> {TABLE_DIR}")
    print(f"Summary  -> {ROOT / 'publication/PUBLICATION_RESULTS.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
