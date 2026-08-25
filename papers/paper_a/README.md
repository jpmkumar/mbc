# Paper A — Tabular WBCD hybrid QML

**This is the Paper A research paper** (tabular Wisconsin Breast Cancer).
Do not describe it as a pilot study.

**Data:** Wisconsin Breast Cancer Diagnostic — **569 samples**, **30 numeric
features** (tabular, not images).

**Claim (must match confirmatory data):** A small VQC can be competitive on
native clinical features. A two-stage Hybrid MLP+VQC is not assumed to win.
Malignant is the positive class (sklearn label 0).

**Contrast with Paper B:** Histopath patches route images through EfficientNet
first; quantum only sees a compressed embedding.

## Confirmatory experiment (submission)

Protocol: [`PROTOCOL.md`](PROTOCOL.md)

```bash
python papers/paper_a/scripts/train_wbcd_cv.py          # 5-fold, leakage-free
python papers/paper_a/scripts/train_wbcd_cv.py --quick  # smoke test
python papers/paper_a/scripts/export_curated_metrics.py
```

| Model | Role |
|-------|------|
| SVM_RBF, RF, MLP_30D | Classical baselines on scaled 30-D |
| MLP_PCA8 | Capacity-matched control on the same 8-D as VQC |
| VQC_8 | Standalone VQC (PCA-8, angle-Y, 2 SEL layers) |
| MLP_H6 | Capacity-matched control on the hybrid 6-D embedding |
| HYBRID_MLP_VQC | Two-stage MLP(30→32→6) + VQC-6 |

Metrics: accuracy, **balanced accuracy**, malignant precision/recall/F1, AUC.
Outputs (gitignored): `results/confirmatory_cv.json`.

**5-fold confirmatory (seed 42, malignant = positive):**

| Model | Bal. acc. | Rec. mal. | AUC |
|-------|-----------|-----------|-----|
| SVM_RBF | 0.975±0.018 | 0.962±0.040 | 0.994±0.006 |
| RF | 0.952±0.027 | 0.934±0.062 | 0.992±0.006 |
| MLP_30D | 0.942±0.028 | 0.915±0.036 | 0.982±0.007 |
| MLP_PCA8 | 0.957±0.041 | 0.934±0.083 | 0.992±0.009 |
| VQC_8 | 0.840±0.034 | 0.694±0.078 | 0.972±0.007 |
| MLP_extractor | 0.927±0.049 | 0.859±0.095 | 0.990±0.009 |
| MLP_H6 | 0.959±0.030 | 0.943±0.049 | 0.981±0.019 |
| HYBRID_MLP_VQC | 0.894±0.050 | 0.821±0.116 | 0.970±0.010 |
| QSVM_ZZ4 (literature ZZ kernel) | 0.859±0.030 | 0.797±0.086 | 0.940±0.016 |
| QSVM_ANGLE4 (best kernel in sweep) | 0.958±0.025 | 0.939±0.051 | 0.994±0.006 |
| VQC_FULL8 (all-qubit readout) | 0.958±0.022 | 0.944±0.049 | 0.991±0.009 |
| SVM_LIN (30-D) | 0.959±0.031 | 0.944±0.046 | 0.991±0.011 |
| SVM_LIN_PCA4 | 0.959±0.036 | 0.934±0.072 | 0.994±0.006 |

## Reference holdout (do not submit these numbers as confirmatory)

Source: the separate local pilot-study checkout (80/20 split). Default sklearn
recall is **benign-class** recall — not malignant sensitivity.

| Model | Accuracy | Precision | Recall (default) | F1 | AUC |
|-------|----------|-----------|------------------|-----|-----|
| Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 99.32% |
| SVM (RBF) | 97.37% | 98.59% | 97.22% | 97.90% | 99.57% |
| XGBoost | 94.74% | 94.59% | 97.22% | 95.89% | 99.34% |
| VQC standalone (8q) | 93.86% | 91.14% | 100% | 95.36% | 97.78% |
| Hybrid MLP+VQC (6q) | 87.72% | 86.25% | 95.83% | 90.79% | 93.78% |

## Status

| Item | Status |
|------|--------|
| Reference holdout + XAI | Separate local pilot-study checkout |
| Confirmatory 5-fold CV (malignant metrics, matched MLP) | **Done** — `results/confirmatory_cv.json` |

## Directory layout

| Path | Git | Purpose |
|------|-----|---------|
| [`PROTOCOL.md`](PROTOCOL.md) | yes | Q1/Q2 confirmatory protocol |
| [`scripts/train_wbcd_cv.py`](scripts/train_wbcd_cv.py) | yes | Confirmatory CV |
| [`scripts/export_curated_metrics.py`](scripts/export_curated_metrics.py) | yes | Snapshot metrics |
| [`results/`](results/) | **no** | CV outputs |
| [`assets/`](assets/) | **no** | Local notes, reference PDFs |
