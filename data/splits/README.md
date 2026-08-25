# Train/val/test splits

Case-level fold membership is **committed** here and is the authoritative record
of which case identifier belongs to which fold. Patch-level manifests are
generated locally: they are large and can be rebuilt losslessly from the
case-level lists plus the archive.

## Mammography / ultrasound / thermography

After placing images under `data/processed/`, run:

```bash
python data/download/setup_datasets.py
```

This creates `train.csv`, `val.csv`, `test.csv`, and `split_stats.json`.

## Histopathology (IDC archive)

Public-case-ID-grouped **5-fold CV** using `StratifiedGroupKFold` with the
historically named `patient_id` column as the group and the IDC-ratio quartile
as the stratification label. The public derivative does not establish that its
279 identifiers are independent patients.

### Do not regenerate the folds — load the shipped manifest

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` does **not**
return a stable assignment across scikit-learn versions. We verified this on the
IDC cohort: two environments produced byte-identical per-case inputs (same
`n0`, `n1`, `total` and `ratio_bin` for all 279 identifiers, 277,524 patches in
total) yet disagreed on the fold of 231 of those identifiers. Regenerating the
split is therefore not a valid way to reproduce any published number.

Case-ID fold membership is committed for exactly this reason. Patch-level
manifests stay untracked because they are large and can be rebuilt losslessly
from the case-ID lists plus the archive.

### The two partitions, and which results used which

| Directory | Partition used by |
|-----------|-------------------|
| `histopath_kaggle/` | All reported five-fold results: E2, E2b, E3, E4, the Stage-B frozen-head analysis, and the Kaggle width cells |
| `histopath/` | The twelve-cell RTX A4000 width matrix only |

Both hold 279 public case identifiers and 277,524 patches with seed 42 and four
ratio bins; they differ only in the case-ID-to-fold assignment. Absolute metrics from the two
partitions are **not** comparable, so never place a number from one beside a
number from the other. `configs/histopath.yaml` points at `histopath/`, so set
the splits root explicitly when reproducing the five-fold results.

Each directory contains:

```
patient_stats.csv                     # n0, n1, total, idc_ratio, ratio_bin, test_fold
split_stats.json                      # per-fold patient/patch counts and IDC ratios
folds/fold_k/{train,test}_patients.csv # committed patient-level membership
folds/fold_k/{train,test}.csv          # patch-level, untracked, rebuilt locally
```

Rebuild the patch-level manifests for a partition from its patient lists:

```bash
python data/download/split_histopath_archive.py \
  --archive-path ~/Downloads/Histopathology-dataset \
  --mode cv --folds 5
```

For quick debugging only, use `--mode holdout` (single 80/20 split).

Load folds in Python:

```python
from src.data.histopath_splits import load_histopath_folds
folds = load_histopath_folds("data/splits/histopath_kaggle")
```

### Train / test (5-fold CV)

Regenerate patch manifests locally if needed, then train:

```bash
# Smoke test (fold 0, 256 train patches)
python scripts/train_histopath_cv.py --fold 0 --quick --max-samples 256

# Full fold 0 classical baseline
python scripts/train_histopath_cv.py --fold 0 --experiment E2

# All 5 folds + Friedman comparison (E2 vs E3)
python scripts/train_histopath_cv.py --compare-classical
```

Results: `results/histopath/cv_summary.json` (mean ± std per fold; Friedman when ≥2 folds/models).

**Colab:** [`COLAB_HISTOPATH.md`](../../COLAB_HISTOPATH.md)  
**Kaggle (full staged commands + backup):** [`KAGGLE_HISTOPATH.md`](../../KAGGLE_HISTOPATH.md)
