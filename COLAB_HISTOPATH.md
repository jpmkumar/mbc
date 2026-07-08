# Histopathology (IDC) Training on Google Colab

Train the **Breast Histopathology (IDC patch)** model with patient-level **5-fold CV** using code from [github.com/jpmkumar/mbc](https://github.com/jpmkumar/mbc).

**GitHub:** code + small split metadata (`patient_stats.csv`, `split_stats.json`)  
**Google Drive:** dataset zip (~2 GB) + generated fold CSVs + checkpoints + results

---

## Before you start (one-time on Mac)

1. Download the Kaggle **Breast Histopathology Images** dataset locally.
2. Zip the folder that contains patient IDs (`10253/`, `10254/`, … each with `0/` and `1/` subfolders):

```bash
cd ~/Downloads
zip -r Histopathology-dataset.zip Histopathology-dataset
```

3. Upload `Histopathology-dataset.zip` to Google Drive, e.g.:

```
MyDrive/mbc_colab/Histopathology-dataset.zip
```

4. Push latest code from Mac (if you changed anything):

```bash
cd "/Users/muthu/ResTest/paper1/mbc"
git pull
git push   # if needed
```

---

## Colab notebook cells

**Runtime → Change runtime type → GPU** (recommended for Stage A; VQC Stage B still uses CPU in config).

### Cell 1 — Clone repo

```python
GITHUB_USER = "jpmkumar"
REPO = "mbc"
TOKEN = ""  # paste classic PAT with `repo` scope, or leave empty if repo is public

import os
if TOKEN:
    url = f"https://{TOKEN}@github.com/{GITHUB_USER}/{REPO}.git"
else:
    url = f"https://github.com/{GITHUB_USER}/{REPO}.git"

if os.path.isdir("/content/mbc"):
    !rm -rf /content/mbc

!git clone {url} /content/mbc
%cd /content/mbc
!git log -1 --oneline
```

### Cell 2 — Install dependencies

```python
!pip install -q -r requirements.txt
```

### Cell 3 — Mount Drive & unpack dataset

```python
from google.colab import drive
import os

drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/mbc_colab"
ZIP_PATH = f"{DRIVE_ROOT}/Histopathology-dataset.zip"
ARCHIVE_PATH = "/content/Histopathology-dataset"

if not os.path.isdir(ARCHIVE_PATH):
    assert os.path.isfile(ZIP_PATH), f"Upload zip to: {ZIP_PATH}"
    !unzip -q -o "{ZIP_PATH}" -d /content/
    print("Unzipped to", ARCHIVE_PATH)
else:
    print("Using existing", ARCHIVE_PATH)

# sanity check
patient_dirs = [d for d in os.listdir(ARCHIVE_PATH)
                if os.path.isdir(os.path.join(ARCHIVE_PATH, d))
                and d != "IDC_regular_ps50_idx5"]
print("Patient folders:", len(patient_dirs))
```

### Cell 4 — Generate 5-fold split manifests

Patch CSVs are **not on GitHub** (too large). Generate once per Colab VM (or copy from Drive if you already built them).

```python
import os

FOLDS_DIR = "data/splits/histopath/folds"
DRIVE_FOLDS = f"{DRIVE_ROOT}/histopath_folds"

if os.path.isdir(DRIVE_FOLDS):
    !mkdir -p data/splits/histopath
    !cp -r "{DRIVE_FOLDS}" data/splits/histopath/folds
    print("Restored folds from Drive")
else:
    !python data/download/split_histopath_archive.py \
        --archive-path /content/Histopathology-dataset \
        --mode cv --folds 5
    !mkdir -p "{DRIVE_FOLDS}"
    !cp -r data/splits/histopath/folds "{DRIVE_FOLDS}"
    print("Saved folds to Drive for next session")
```

### Cell 5 — Use GPU on Colab (optional but recommended)

The default `configs/histopath.yaml` sets `classical_device: cpu` (safe on Mac). On Colab, switch to GPU:

```python
from pathlib import Path

cfg_path = Path("configs/histopath.yaml")
text = cfg_path.read_text()
text = text.replace("classical_device: cpu", "classical_device: auto")
cfg_path.write_text(text)
print("Updated configs/histopath.yaml → classical_device: auto")
```

### Cell 6 — Smoke test (fast, ~5–10 min on GPU)

```python
!python scripts/train_histopath_cv.py \
  --fold 0 --quick --max-samples 256 --experiment E2
```

Expected: test metrics printed + `results/histopath/cv_summary.json`.

### Cell 7 — Train one fold (full)

Run **one fold per Colab session** to avoid disconnects (~220k train patches per fold).

```python
FOLD = 0
EXPERIMENT = "E2"   # classical baseline
# EXPERIMENT = "E3"   # hybrid + VQC (slower)

!python scripts/train_histopath_cv.py \
  --fold {FOLD} --experiment {EXPERIMENT}
```

### Cell 8 — Train all 5 folds + Friedman (E2 vs E3)

Long run — use Colab Pro or split across sessions with `--fold N`.

```python
!python scripts/train_histopath_cv.py --compare-classical
```

### Cell 9 — Save results to Drive

```python
!mkdir -p "{DRIVE_ROOT}/histopath_results" "{DRIVE_ROOT}/histopath_checkpoints"
!cp -r results/histopath/* "{DRIVE_ROOT}/histopath_results/" 2>/dev/null
!cp -r results/histopath/checkpoints/* "{DRIVE_ROOT}/histopath_checkpoints/" 2>/dev/null
!ls -lh "{DRIVE_ROOT}/histopath_results"
```

---

## Resume after disconnect

```python
%cd /content/mbc
!git pull

# restore dataset + folds from Drive (Cells 3–4)
# restore checkpoints if resuming same fold/experiment:
!mkdir -p results/histopath/checkpoints
!cp {DRIVE_ROOT}/histopath_checkpoints/* results/histopath/checkpoints/

# continue same fold
!python scripts/train_histopath_cv.py --fold 0 --experiment E2
```

Checkpoints are named like `E2_histopath_fold0_histopath_seed42.pt`.

---

## Outputs

| File | Description |
|------|-------------|
| `results/histopath/cv_summary.json` | Per-fold test metrics, mean ± std, Friedman (if `--compare-classical`) |
| `results/histopath/checkpoints/` | Best + latest weights per fold/experiment |
| `data/splits/histopath/patient_stats.csv` | 279 patients, IDC ratio bins, test fold assignment |
| `data/splits/histopath/folds/fold_k/{train,test}.csv` | Patch manifests (generated on Colab or copied from Drive) |

---

## Command reference

```bash
# Single fold, classical
python scripts/train_histopath_cv.py --fold 0 --experiment E2

# Single fold, hybrid quantum
python scripts/train_histopath_cv.py --fold 0 --experiment E3

# All folds + Friedman E2 vs E3
python scripts/train_histopath_cv.py --compare-classical

# Debug subset
python scripts/train_histopath_cv.py --fold 0 --quick --max-samples 256 --experiment E2
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA error: no kernel image is available` on **P100** | Kaggle's PyTorch does **not** support P100 (sm_60). Switch to **GPU T4 x2** |
| Archive path not found | Re-run Cell 3; check zip path on Drive |
| `No k-fold manifests found` | Run Cell 4 (`split_histopath_archive.py --mode cv`) |
| Colab disconnects | Train one `--fold` at a time; copy checkpoints to Drive after each run |
| Out of memory | Lower `batch_size` in `configs/histopath.yaml` (e.g. 16) |
| Training very slow on CPU | Run Cell 5 to enable GPU (`classical_device: auto`) |
| `Mismatched Tensor types` on Mac MPS | Keep `classical_device: cpu` locally; use GPU on Colab |

---

## Citation

```
Sawyer-Lee, R., et al. (2016). Curated Breast Imaging Subset of DDSM [Data set]. TCIA.
https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
```

IDC patch dataset: Kaggle **Breast Histopathology Images** (patch-level labels in `0/` vs `1/` folders).
