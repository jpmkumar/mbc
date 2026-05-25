# Colab Training Results Analysis

**Archive:** `results-20260525T102408Z-3-001.zip`  
**Experiment:** `E3_hybrid_seed42`  
**Train time:** ~39 min (2349 s)

## Reported metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 0.40 | Poor |
| Precision | 0.40 | Many false positives |
| Recall | 1.00 | All malignant cases caught |
| F1 | 0.57 | Misleadingly OK due to collapse |
| AUC | 0.44 | Worse than random (0.5) |

## Root cause: class collapse

The saved checkpoint predicts **100% malignant (class 1)** on validation:

- Val labels (mammo): 229 benign, 216 malignant (445 total)
- Predictions: **445 malignant, 0 benign**
- Confusion matrix: `[[0, 229], [0, 216]]`

This is classic **predict-all-positive collapse**. F1 looked acceptable (~0.57) so this checkpoint was saved as "best".

## Fixes applied (in `mbc/` repo)

1. Checkpoint selection uses **balanced accuracy** (not raw F1)
2. **Freeze backbone** during Stage B — train VQC only (pilot-study approach)
3. Lower **lr_quantum** from 1e-3 → 1e-4
4. Final metrics on **test** split (not val)
5. **`--modality mammo`** for real-data-only training
6. **`scripts/analyze_results.py`** for checkpoint diagnostics

## Re-run on Colab

Update `mbc_code.zip` with fixed code, then:

```bash
python experiments/run_training.py --experiment hybrid --modality mammo
```

Analyze checkpoint:

```bash
python scripts/analyze_results.py results/checkpoints/E3_hybrid_seed42.pt --modality mammo --split test
```

Expected: `class_collapse: false`, balanced_accuracy > 0.5 (target 0.75+ after full training).
