#!/usr/bin/env python3
"""Revision statistics for Paper A (no retraining; numpy/scipy/sklearn only).

Reads the deposited fold-level CSVs and writes results/revision_stats.json:

1. One SD convention for every table: sample SD (ddof=1) over the five folds.
2. Nadeau--Bengio corrected resampled paired t-tests and 95% CIs for every
   reported contrast (folds share ~75% of their training rows, so the naive
   paired t is anti-conservative).
3. Kernel--target alignment (KTA) recomputed with an exact 4-qubit statevector
   simulator. The diagnosis script's IQP encoder lacked data-dependent ZZ
   phases, so its fidelity kernel collapsed to the angle kernel and duplicated
   five of the twenty KTA points. Here IQP follows the sweep script
   (H, RZ(2x_i), all-pairs CNOT-RZ(2 x_i x_j)-CNOT), which produced the
   reported QSVM-IQP4 scores. Pooled and within-kernel Spearman correlations
   are reported.

Usage:  python papers/paper_a/scripts/analysis_revision_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MALIGNANT = 0
N_TOTAL = 569


# ---------------------------------------------------------------- statistics
def summary(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(v.size)}


def nb_test(a, b, k_folds=5, n_repeats=1):
    """Nadeau & Bengio (2003) corrected resampled paired t-test, a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    j = d.size
    n_test = N_TOTAL / k_folds
    ratio = n_test / (N_TOTAL - n_test)
    var = d.var(ddof=1)
    naive_se = np.sqrt(var / j)
    corr_se = np.sqrt((1.0 / j + ratio) * var)
    df = j - 1
    tcrit = stats.t.ppf(0.975, df)
    out = {
        "n": j,
        "mean_delta": float(d.mean()),
        "naive_t_p": float(2 * stats.t.sf(abs(d.mean() / naive_se), df)) if naive_se > 0 else None,
        "nb_t": float(d.mean() / corr_se) if corr_se > 0 else None,
        "nb_p": float(2 * stats.t.sf(abs(d.mean() / corr_se), df)) if corr_se > 0 else None,
        "nb_ci95": [float(d.mean() - tcrit * corr_se), float(d.mean() + tcrit * corr_se)],
        "folds_a_better": int((d > 0).sum()),
        "folds_b_better": int((d < 0).sum()),
    }
    return out


def pick(df, model, col="balanced_accuracy", sort=("repeat", "fold")):
    g = df[df["model"] == model].copy()
    keys = [c for c in sort if c in g.columns]
    return g.sort_values(keys)[col].to_numpy()


# ------------------------------------------------ 4-qubit statevector kernels
H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)


def apply_1q(psi, u, q, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(u, np.moveaxis(psi, q, 0), axes=1), 0, q)
    return psi.reshape(-1)


def rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], complex)


def zz_phase(psi, i, j, phi, n):
    """CNOT(i,j) RZ_j(phi) CNOT(i,j) == exp(-i phi/2 Z_i Z_j)."""
    idx = np.arange(2 ** n)
    zi = 1 - 2 * ((idx >> (n - 1 - i)) & 1)
    zj = 1 - 2 * ((idx >> (n - 1 - j)) & 1)
    return psi * np.exp(-0.5j * phi * zi * zj)


def state(x, kind):
    n = len(x)
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    if kind == "angle":
        for q in range(n):
            psi = apply_1q(psi, ry(x[q]), q, n)
        return psi
    reps = 2 if kind in ("zz", "z") else 1
    for _ in range(reps):
        for q in range(n):
            psi = apply_1q(psi, H, q, n)
            psi = apply_1q(psi, rz(2 * x[q]), q, n)
        if kind == "zz":
            for q in range(n - 1):
                psi = zz_phase(psi, q, q + 1, 2 * x[q] * x[q + 1], n)
        elif kind == "iqp":
            for a in range(n):
                for b in range(a + 1, n):
                    psi = zz_phase(psi, a, b, 2 * x[a] * x[b], n)
    return psi


def gram(xa, xb, kind):
    sa = np.array([state(v, kind) for v in xa])
    sb = np.array([state(v, kind) for v in xb])
    return np.abs(sa.conj() @ sb.T) ** 2


def kta(k, y):
    ys = np.where(y == MALIGNANT, 1.0, -1.0)
    t = np.outer(ys, ys)
    n = len(y)
    h = np.eye(n) - 1.0 / n
    kc, tc = h @ k @ h, h @ t @ h
    return float((kc * tc).sum() / (np.linalg.norm(kc) * np.linalg.norm(tc)))


def kta_block(seed=42):
    data = load_breast_cancer()
    x, y = data.data, data.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(x, y)):
        sc = StandardScaler().fit(x[tr])
        pca = PCA(n_components=4, random_state=0).fit(sc.transform(x[tr]))
        mm = MinMaxScaler((0, np.pi)).fit(pca.transform(sc.transform(x[tr])))
        xtr = mm.transform(pca.transform(sc.transform(x[tr])))
        xte = mm.transform(pca.transform(sc.transform(x[te])))
        for kind in ("zz", "angle", "iqp", "z"):
            ktr = gram(xtr, xtr, kind)
            kte = gram(xte, xtr, kind)
            svc = SVC(C=10.0, kernel="precomputed").fit(ktr, y[tr])
            rows.append({
                "fold": fold, "kernel": kind, "kta_train": kta(ktr, y[tr]),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], svc.predict(kte))),
            })
    df = pd.DataFrame(rows)
    pooled = stats.spearmanr(df["kta_train"], df["balanced_accuracy"])
    within = {}
    for kind, g in df.groupby("kernel"):
        r = stats.spearmanr(g["kta_train"], g["balanced_accuracy"])
        within[kind] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    # rank-on-kernel-means (between-kernel only, 4 points)
    means = df.groupby("kernel")[["kta_train", "balanced_accuracy"]].mean()
    return df, {
        "pooled_rho": float(pooled.statistic), "pooled_p": float(pooled.pvalue), "n_pairs": len(df),
        "within_kernel": within,
        "kernel_means": {k: {"kta": float(r.kta_train), "bal_acc": float(r.balanced_accuracy)}
                         for k, r in means.iterrows()},
    }


# ------------------------------------------------- QFEAT information check
def cnot(psi, c, t, n):
    idx = np.arange(2 ** n)
    cb = (idx >> (n - 1 - c)) & 1
    flipped = idx ^ (1 << (n - 1 - t))
    out = psi.copy()
    out[cb == 1] = psi[flipped[cb == 1]]
    return out


def qfeat(x):
    """Sidebranch fixed map: RY(x) on 4 wires, CNOT chain + ring, <Z_i>, <Z_i Z_i+1>."""
    n = len(x)
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for q in range(n):
        psi = apply_1q(psi, ry(x[q]), q, n)
    for q in range(n - 1):
        psi = cnot(psi, q, q + 1, n)
    psi = cnot(psi, n - 1, 0, n)
    p = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    z = [1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)]
    feats = [float((p * z[q]).sum()) for q in range(n)]
    feats += [float((p * z[q] * z[q + 1]).sum()) for q in range(n - 1)]
    return feats


def qfeat_block(seed=42):
    data = load_breast_cancer()
    x, y = data.data, data.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(x, y)):
        sc = StandardScaler().fit(x[tr])
        xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
        pca4 = PCA(n_components=4).fit(xtr)
        mm4 = MinMaxScaler((0.0, np.pi)).fit(pca4.transform(xtr))
        qtr = np.array([qfeat(v) for v in mm4.transform(pca4.transform(xtr))])
        qte = np.array([qfeat(v) for v in mm4.transform(pca4.transform(xte))])
        res = {"fold": fold}
        def fit(name, a, b, **kw):
            m = SVC(C=10, gamma="scale", **kw).fit(a, y[tr])
            res[name] = float(balanced_accuracy_score(y[te], m.predict(b)))
            return m
        base = fit("svm_rbf_30", xtr, xte)
        both = fit("qfeat_svm_repro", np.hstack([xtr, qtr]), np.hstack([xte, qte]))
        fit("qfeat_only_rbf", qtr, qte)
        fit("qfeat_only_linear", qtr, qte, kernel="linear")
        qs = StandardScaler().fit(qtr)
        fit("qfeat_std_concat_rbf", np.hstack([xtr, qs.transform(qtr)]), np.hstack([xte, qs.transform(qte)]))
        # share of the mean squared pairwise distance carried by the 7 extras
        xa = np.hstack([xtr, qtr])
        var_tot = xa.var(axis=0).sum()
        res["qfeat_share_sq_distance"] = float(qtr.var(axis=0).sum() / var_tot)
        res["identical_predictions"] = bool(np.array_equal(base.predict(xte), both.predict(np.hstack([xte, qte]))))
        rows.append(res)
    df = pd.DataFrame(rows)
    return {c: summary(df[c]) for c in df.columns if c not in ("fold", "identical_predictions")} | {
        "identical_predictions_all_folds": bool(df["identical_predictions"].all())}


# ------------------------------------------------- boost (summary only, ddof=0 -> ddof=1)
def boost_block():
    f = RES / "vqc_boost" / "summary.json"
    rows = RES / "vqc_boost" / "rows.csv"
    if rows.exists():
        df = pd.read_csv(rows)
        return {"source": "rows.csv", "summaries": {
            m: {c: summary(g[c]) for c in ("balanced_accuracy", "recall_malignant", "auc") if c in g}
            for m, g in df.groupby("model")}}
    d = json.loads(f.read_text())["summary"]
    k = np.sqrt(5 / 4)
    return {"source": "summary.json (population SD rescaled by sqrt(n/(n-1)); fold rows not deposited)",
            "summaries": {m: {c: {"mean": v["mean"], "sd": v["std"] * k, "n": v["n"]}
                              for c, v in mm.items() if c != "n_params"} | {"n_params": mm["n_params"]["mean"]}
                          for m, mm in d.items()}}


def gpu_block():
    df = pd.read_csv(RES / "gpu_investigation" / "rows.csv")
    out = {}
    for (arm, model), g in df.groupby(["arm", "model"]):
        key = f"{arm}|{model}"
        if arm == "entangle_ablation":
            for ent, gg in g.groupby("entangle"):
                out[f"{key}|{ent}"] = summary(gg["balanced_accuracy"])
        elif arm == "sample_efficiency":
            for fr, gg in g.groupby("train_fraction"):
                out[f"{key}|{fr}"] = summary(gg["balanced_accuracy"])
        else:
            out[key] = summary(g["balanced_accuracy"].dropna())
    lr = df[df.arm == "lr_grid_best"]
    if len(lr):
        out["lr_grid_best_contrast"] = nb_test(pick(lr, "VQC-8"), pick(lr, "MLP-PCA8"))
    return out


# ----------------------------------------------------------------------- main
def main():
    out = {"sd_convention": "sample SD (ddof=1) across folds", "summaries": {}, "contrasts": {}}

    conf = pd.read_csv(RES / "confirmatory_folds.csv")
    rep = pd.read_csv(RES / "repeated_2x5" / "confirmatory_folds.csv")
    ce = pd.read_csv(RES / "vqc_ce80" / "confirmatory_folds.csv")
    if "repeat" not in rep.columns:
        rep["repeat"] = rep.groupby("model").cumcount() // 5

    files = {
        "confirmatory": conf,
        "quantum_sweep": pd.read_csv(RES / "quantum_sweep_folds.csv"),
        "hybrid_sweep": pd.read_csv(RES / "hybrid_sweep_folds.csv"),
        "sidebranch": pd.read_csv(RES / "sidebranch_folds.csv"),
        "morph_kta": pd.read_csv(RES / "morph_kta_folds.csv"),
        "vqc_ce80": ce,
    }
    metrics = ["balanced_accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "auc"]
    for name, df in files.items():
        out["summaries"][name] = {
            m: {c: summary(g[c]) for c in metrics if c in g.columns}
            for m, g in df.groupby("model")
        }
    depth = pd.read_csv(RES / "quantum_diagnosis_depth.csv")
    out["summaries"]["diagnosis_depth"] = {
        f"{m}_L{L}": {c: summary(g[c]) for c in ("balanced_accuracy", "recall_malignant")}
        for (m, L), g in depth.groupby(["model", "n_layers"])
    }

    c = out["contrasts"]
    pairs = [
        ("VQC_8", "MLP_PCA8"), ("VQC_8", "SVM_RBF"), ("HYBRID_MLP_VQC", "MLP_H6"),
        ("HYBRID_MLP_VQC", "MLP_30D"), ("QSVM_ZZ4", "SVM_RBF"), ("QSVM_ZZ4", "SVM_RBF_PCA4"),
        ("QSVM_ZZ4", "SVM_LIN_PCA4"), ("QSVM_ZZ4", "MLP_PCA4"), ("QSVM_ZZ4", "SVM_LIN"),
    ]
    for a, b in pairs:
        for col in ("balanced_accuracy", "recall_malignant"):
            c[f"confirmatory:{a}-{b}:{col}"] = nb_test(pick(conf, a, col), pick(conf, b, col))
    for a, b in [("VQC_8", "MLP_PCA8"), ("HYBRID_MLP_VQC", "MLP_H6")]:
        c[f"repeated_2x5:{a}-{b}:balanced_accuracy"] = nb_test(pick(rep, a), pick(rep, b), 5, 2)
        c[f"vqc_ce80:{a}-{b}:balanced_accuracy"] = nb_test(pick(ce, a), pick(ce, b))

    sw = files["quantum_sweep"]
    for a in ("QSVM_ANGLE4", "QSVM_IQP4", "QSVM_Z4", "QSVM_ZZ8", "VQC_FULL8", "VQC_REUP8"):
        if a in set(sw.model):
            for b in ("SVM_RBF", "SVM_LIN_PCA4"):
                if b in set(sw.model):
                    c[f"quantum_sweep:{a}-{b}:balanced_accuracy"] = nb_test(pick(sw, a), pick(sw, b))
    sb = files["sidebranch"]
    for a in ("QFEAT_SVM", "CLASS_NL_SVM", "SIDEBRANCH"):
        if a in set(sb.model):
            c[f"sidebranch:{a}-SVM_RBF:balanced_accuracy"] = nb_test(pick(sb, a), pick(sb, "SVM_RBF"))

    out["qfeat_check"] = qfeat_block()
    out["boost"] = boost_block()
    out["gpu_investigation"] = gpu_block()

    kdf, kres = kta_block()
    kdf.to_csv(RES / "revision_kta_recomputed.csv", index=False)
    out["kta"] = kres

    (RES / "revision_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    print("Wrote", RES / "revision_stats.json")


if __name__ == "__main__":
    main()
