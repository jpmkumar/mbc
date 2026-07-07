# Train/val/test splits

CSV manifests are **generated locally** — not stored on GitHub.

## Mammography / ultrasound / thermography

After placing images under `data/processed/`, run:

```bash
python data/download/setup_datasets.py
```

This creates `train.csv`, `val.csv`, `test.csv`, and `split_stats.json`.

## Histopathology (IDC archive)

Patient-level **5-fold CV** (for mean ± std and Friedman/Nemenyi tests):

```bash
python data/download/split_histopath_archive.py \
  --archive-path ~/Downloads/Histopathology-dataset \
  --mode cv --folds 5
```

Uses `StratifiedGroupKFold` with `patient_id` as group and IDC-ratio quartile as
stratification label. Outputs:

```
data/splits/histopath/
  patient_stats.csv      # includes ratio_bin, test_fold
  split_stats.json
  folds/fold_0/{train,test}.csv
  ...
  folds/fold_4/{train,test}.csv
```

For quick debugging only, use `--mode holdout` (single 80/20 split).

Load folds in Python:

```python
from src.data.histopath_splits import load_histopath_folds
folds = load_histopath_folds("data/splits/histopath")
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

**Colab:** mount Drive → unzip `mbc_mammo.zip` → run `setup_datasets.py` → train.
