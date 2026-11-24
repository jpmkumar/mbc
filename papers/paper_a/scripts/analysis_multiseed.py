#!/usr/bin/env python3
"""Aggregate the multi-seed revision runs written by run_multiseed.sh.

For every model it separates initialisation-seed variance from fold variance,
and for each contrast it applies the Nadeau--Bengio corrected resampled t-test
to seed-averaged fold scores. Equivalence (TOST) uses a margin stated before
the run: |delta balanced accuracy| < 0.02.

Writes <root>/multiseed_summary.json and <root>/multiseed_summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_TOTAL, K = 569, 5
MARGIN = 0.02
METRICS = ("balanced_accuracy", "recall_malignant", "auc")


def load(root: Path, sub: str, fname: str, tag: str) -> pd.DataFrame:
    frames = []
    for d in sorted((root / sub).glob("seed_*")):
        f = d / fname
        if f.exists():
            df = pd.read_csv(f)
            df["init_seed"] = int(d.name.split("_")[1])
            df["run"] = tag
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nb(d: np.ndarray) -> dict:
    j = d.size
    ratio = (N_TOTAL / K) / (N_TOTAL - N_TOTAL / K)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + ratio) * var)
    df = j - 1
    if se == 0:
        return {"mean_delta": float(d.mean()), "nb_p": None}
    t = d.mean() / se
    t95, t90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
    lo90, hi90 = d.mean() - t90 * se, d.mean() + t90 * se
    return {
        "mean_delta": float(d.mean()),
        "nb_p": float(2 * stats.t.sf(abs(t), df)),
        "ci95": [float(d.mean() - t95 * se), float(d.mean() + t95 * se)],
        "ci90": [float(lo90), float(hi90)],
        "equivalent_within_margin": bool(lo90 > -MARGIN and hi90 < MARGIN),
        "margin": MARGIN,
    }


def describe(df: pd.DataFrame) -> dict:
    out = {}
    for model, g in df.groupby("model"):
        out[model] = {}
        for m in METRICS:
            if m not in g:
                continue
            per_seed = g.groupby("init_seed")[m].mean()
            per_fold = g.groupby("fold")[m].mean()
            fold_sd_within_seed = g.groupby("init_seed")[m].std(ddof=1).mean()
            out[model][m] = {
                "mean": float(g[m].mean()),
                "n_seeds": int(per_seed.size),
                "sd_between_seeds": float(per_seed.std(ddof=1)) if per_seed.size > 1 else 0.0,
                "sd_between_folds_seedavg": float(per_fold.std(ddof=1)),
                "mean_fold_sd_within_seed": float(fold_sd_within_seed),
                "seed_range": [float(per_seed.min()), float(per_seed.max())],
            }
    return out


def fold_means(df, model, metric="balanced_accuracy"):
    g = df[df.model == model]
    return g.groupby("fold")[metric].mean().sort_index().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("papers/paper_a/results/multiseed"))
    args = ap.parse_args()
    root = args.root

    conf = load(root, "confirmatory", "confirmatory_folds.csv", "confirmatory")
    ce = load(root, "vqc_ce80", "confirmatory_folds.csv", "vqc_ce80")
    boost = load(root, "vqc_boost", "rows.csv", "vqc_boost")

    out = {"margin_bal_acc": MARGIN, "runs": {}}
    for tag, df in (("confirmatory", conf), ("vqc_ce80", ce), ("vqc_boost", boost)):
        if len(df):
            out["runs"][tag] = describe(df)

    contrasts = {}
    if len(conf):
        for a, b in [("VQC_8", "MLP_PCA8"), ("VQC_8", "SVM_RBF"), ("HYBRID_MLP_VQC", "MLP_H6"),
                     ("HYBRID_MLP_VQC", "MLP_30D"), ("QSVM_ZZ4", "SVM_LIN_PCA4"),
                     ("MLP_PCA8", "SVM_RBF"), ("MLP_30D", "SVM_RBF")]:
            for m in ("balanced_accuracy", "recall_malignant"):
                contrasts[f"confirmatory:{a}-{b}:{m}"] = nb(fold_means(conf, a, m) - fold_means(conf, b, m))
    if len(ce):
        for a, b in [("VQC_8", "MLP_PCA8"), ("HYBRID_MLP_VQC", "MLP_H6")]:
            contrasts[f"vqc_ce80:{a}-{b}:balanced_accuracy"] = nb(fold_means(ce, a) - fold_means(ce, b))
    if len(boost):
        svm = fold_means(conf, "SVM_RBF") if len(conf) else None
        for a in sorted(set(boost.model) - {"linear_pca8", "mlp_pca8"}):
            for b in ("linear_pca8", "mlp_pca8"):
                for m in ("balanced_accuracy", "recall_malignant"):
                    contrasts[f"vqc_boost:{a}-{b}:{m}"] = nb(fold_means(boost, a, m) - fold_means(boost, b, m))
            if svm is not None:
                contrasts[f"vqc_boost:{a}-SVM_RBF:balanced_accuracy"] = nb(fold_means(boost, a) - svm)
        if svm is not None:
            contrasts["vqc_boost:linear_pca8-SVM_RBF:balanced_accuracy"] = nb(fold_means(boost, "linear_pca8") - svm)
    out["contrasts"] = contrasts

    (root / "multiseed_summary.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = ["# Multi-seed summary", "", f"Equivalence margin: ±{MARGIN} balanced accuracy (TOST, 90% CI).", ""]
    for tag, d in out["runs"].items():
        lines += [f"## {tag}", "", "| Model | Bal. acc. | SD seeds | SD folds | Rec. mal. | AUC |", "|---|---|---|---|---|---|"]
        for model, mm in sorted(d.items(), key=lambda kv: -kv[1]["balanced_accuracy"]["mean"]):
            ba = mm["balanced_accuracy"]
            lines.append(
                f"| {model} | {ba['mean']:.3f} | {ba['sd_between_seeds']:.3f} | {ba['sd_between_folds_seedavg']:.3f} "
                f"| {mm.get('recall_malignant', {}).get('mean', float('nan')):.3f} | {mm.get('auc', {}).get('mean', float('nan')):.3f} |")
        lines.append("")
    lines += ["## Contrasts (seed-averaged fold scores, Nadeau–Bengio corrected)", "",
              "| Contrast | Δ | 95% CI | p | Equivalent (±margin) |", "|---|---|---|---|---|"]
    for k, v in contrasts.items():
        if v.get("nb_p") is None:
            lines.append(f"| {k} | {v['mean_delta']:+.3f} | — | — | — |")
            continue
        lines.append(f"| {k} | {v['mean_delta']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                     f"| {v['nb_p']:.2g} | {'yes' if v['equivalent_within_margin'] else 'no'} |")
    (root / "multiseed_summary.md").write_text("\n".join(lines) + "\n")
    print("Wrote", root / "multiseed_summary.json")
    print("Wrote", root / "multiseed_summary.md")


if __name__ == "__main__":
    main()
