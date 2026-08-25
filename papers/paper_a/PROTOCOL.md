# Paper A confirmatory protocol (Q1/Q2)

This is the **submission experiment** for the tabular WBCD paper.
The earlier 80/20 holdout in the separate local pilot-study checkout is the
**reference study** (cite it; do not treat its default-sklearn recall as
malignant sensitivity).

## Scientific claim (must match the data)

On native 30-D clinical features, a small VQC can be **competitive** with
classical models. A two-stage Hybrid MLP+VQC is **not** assumed to win.
No “quantum advantage” and no “zero missed cancers” language unless
malignant-class recall (`pos_label=0`) supports it.

## Dataset

- `sklearn.datasets.load_breast_cancer` (Wisconsin Diagnostic).
- *n* = 569, 30 numeric features.
- Labels: **0 = malignant**, **1 = benign**.
- Mild imbalance (~212 / 357).

## Validation (no leakage)

- Stratified *k*-fold (*k* = 5, seed 42). Optional `--n-repeats` for
  repeated CV.
- `StandardScaler` and `PCA` are **fit on the training fold only**.
- Quantum inputs: train-fold `MinMaxScaler` to \([0, \pi]\) after PCA
  (standalone VQC) or after the MLP embedding (hybrid).

## Models (same folds)

| ID | Input | Role |
|----|--------|------|
| `SVM_RBF` | scaled 30-D | Strong classical baseline |
| `RF` | scaled 30-D | Tree baseline |
| `MLP_30D` | scaled 30-D | Full-feature classical net |
| `MLP_PCA8` | same 8-D as VQC | **Capacity-matched** control |
| `VQC_8` | PCA-8, angle-Y, 2× SEL | Standalone VQC (48 params) |
| `MLP_H6` | same 6-D as hybrid VQC | **Capacity-matched** hybrid control |
| `HYBRID_MLP_VQC` | MLP(30→32→6) then VQC-6 | Two-stage hybrid |
| `SVM_LIN` | scaled 30-D | Linear SVM (the weak control many QSVM papers beat) |
| `SVM_RBF_PCA4` / `SVM_LIN_PCA4` / `MLP_PCA4` | same 4-D as QSVM | Matched controls for the ZZ kernel |
| `QSVM_ZZ4` | PCA-4, ZZ fidelity kernel | Literature QSVM (Jose / Yadav / Vashisth) |

`MLP_PCA8` / `MLP_H6` use LayerNorm + GELU MLP with **equal or greater**
parameter count than the paired VQC (same spirit as Paper B E2b). If the
VQC still wins, the gap is not “extra non-linearity.”

## Metrics (malignant = positive)

Report **mean ± std** over folds:

- Accuracy, balanced accuracy
- Malignant precision / recall / F1 (`pos_label=0`)
- ROC-AUC with malignant as the positive class

Do **not** use default `recall_score` (that is benign recall).

## Statistics

Paired tests on the **same fold scores**:

- `VQC_8` vs `MLP_PCA8` (quantum vs matched classical on 8-D)
- `VQC_8` vs `SVM_RBF`
- `HYBRID_MLP_VQC` vs `MLP_H6`
- `HYBRID_MLP_VQC` vs `MLP_30D`

Wilcoxon signed-rank and paired *t*-test. *n* = 5 is small; treat *p*
as descriptive unless `--n-repeats` ≥ 2.

## How to run

```bash
# Full confirmatory run (default)
python papers/paper_a/scripts/train_wbcd_cv.py

# QSVM arm only (ZZ kernel + matched PCA-4 classical)
.venv/bin/python papers/paper_a/scripts/train_wbcd_cv.py --qsvm-only

# Smoke test
python papers/paper_a/scripts/train_wbcd_cv.py --quick --qsvm-only
```

Do **not** drop `SVM_RBF` to make QSVM look like a winner. If QSVM beats
linear SVM on 4-D but not RBF-SVM on 30-D, report that exactly.

## Quantum algorithm sweep

```bash
.venv/bin/python papers/paper_a/scripts/train_wbcd_quantum_sweep.py
```

Output: `papers/paper_a/results/quantum_sweep.json`. Same folds and
malignant-positive metrics. As of the 5-fold run, no quantum arm beats
`SVM_RBF` (bal. acc. 0.975). Closest: `QSVM_ANGLE4` and `VQC_FULL8`
(~0.958), matching `SVM_LIN_PCA4`, not exceeding `SVM_RBF`.

## Hybrid algorithm sweep

```bash
.venv/bin/python papers/paper_a/scripts/train_wbcd_hybrid_sweep.py
```

Output: `papers/paper_a/results/hybrid_sweep.json`. End-to-end, fusion,
two-stage full-readout, and MLP+QSVM were tested. None beat `SVM_RBF`.
The best hybrid (`HYBRID_E2E`, bal. acc. ~0.938) is below standalone
`VQC_FULL8` (~0.956). Fusion weight settled near α≈0.60 (leans classical).

## Side-branch / early-fusion hybrid (not histopath)

The histopath recipe and the old two-stage MLP→VQC both **discard** native
features and let the VQC classify a compressed embedding. This arm does the
opposite: keep 30-D clinical features and add quantum as extras or a side
branch.

```bash
.venv/bin/python papers/paper_a/scripts/train_wbcd_sidebranch.py
```

| ID | What is new vs histopath / Mari |
|----|----------------------------------|
| `QFEAT_SVM` | Fixed (untrained) 4-qubit Z/ZZ expectations concatenated to 30-D → SVM-RBF |
| `CLASS_NL_SVM` | Width-matched classical nonlinear extras (not quantum) |
| `SIDEBRANCH` | MLP on 30-D concatenated with VQC(PCA-8); VQC is not the only head |
| `SVM_RBF` | Same 30-D baseline — do not drop |

Output: `papers/paper_a/results/sidebranch.json`.

Outputs (gitignored): `papers/paper_a/results/confirmatory_cv.json`,
`confirmatory_folds.csv`, `confirmatory_stats.json`.
