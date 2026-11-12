#!/usr/bin/env python3
"""Draw the Paper D architecture schematic (Fig. 1). Fig. 2 is exported with metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, fc):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    box(ax, (0.2, 1.4), 1.8, 1.2, "BreakHis\nimage\n40–400×", "#deebf7")
    box(ax, (2.3, 1.4), 2.2, 1.2, "EfficientNet-B0\n+ transformer\n→ 8-D bottleneck", "#c6dbef")
    box(ax, (4.8, 2.35), 2.0, 1.1, "F: freeze\nencoder", "#e5f5e0")
    box(ax, (4.8, 0.55), 2.0, 1.1, "T: last block\nfine-tune", "#c7e9c0")
    box(ax, (7.2, 2.35), 2.1, 1.1, "E2b-matched\nMLP head", "#fee6ce")
    box(ax, (7.2, 0.55), 2.1, 1.1, "VQC head\n8q · 2L · angle-Y", "#fdd0a2")
    box(ax, (9.7, 1.4), 2.1, 1.2, "Case-balanced\nAUPRC\n(held-out cases)", "#f2f0f7")
    for x0, x1, y in ((2.0, 2.3, 2.0), (4.5, 4.8, 2.9), (4.5, 4.8, 1.1), (6.8, 7.2, 2.9), (6.8, 7.2, 1.1), (9.3, 9.7, 2.0)):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(5.8, 3.65, "Regime", ha="center", fontsize=8)
    ax.text(8.25, 3.65, "Head (only difference)", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_architecture.pdf", dpi=300)
    fig.savefig(FIG / "fig01_architecture.png", dpi=300)
    plt.close(fig)
    print("wrote fig01")


if __name__ == "__main__":
    main()
