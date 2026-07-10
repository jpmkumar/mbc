# Histopathology (IDC) — Complete Kaggle Setup

Full staged commands for **patient-level 5-fold CV** on Kaggle, with **backup after every stage**.

**Repo:** https://github.com/jpmkumar/mbc  
**Dataset:** [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (add as notebook Input)

---

## Experiment roadmap

| Phase | What | Folds | Model |
|-------|------|-------|-------|
| **0** | One-time setup + splits | — | — |
| **1** | Classical baseline | 0–4 | **E2** |
| **2** | Hybrid quantum | 0–4 | **E3** |
| **3** | Friedman E2 vs E3 | all | `--compare-classical` |
| **4** | Copy results to Mac | — | — |

**Expected time (GPU T4, per fold):** ~5–6 h (E2, less with early stopping), ~8–12 h (E3).

---

## IMPORTANT — how to save results reliably

An interactive **draft** session **wipes `/kaggle/working/` when you power off**. If you only create the
backup zip and power off without downloading it, the results (checkpoints) are lost even though the
metrics printed in the log.

Choose **one** of these to keep results safe:

### Option A — Save Version → Save & Run All (recommended for one fold)

1. Put the setup + one training fold + backup cell in the notebook.
2. Click **Save Version → Save & Run All**.
3. Kaggle runs it as a **background job** (survives browser close), **persists `/kaggle/working/`
   output**, and **emails you** when done (like fold 0).
4. After it finishes, download the zip from the **Output** tab.

Requirement: the whole notebook must finish within **12 h**. One E2 fold fits easily.

### Option B — interactive, but download before power off

1. Run the training + backup cells.
2. When the backup prints `... 130M ...mbc_histopath_*.zip`, **download it**.
3. **Verify the file is ~130 MB in `~/Downloads`** (a few-kB zip means the backup ran before
   training finished — rerun the backup cell).
4. **Only then** `Run → Power off`.

Do **not** power off until the ~130 MB zip is confirmed on your Mac.

---

## Kaggle settings (every session)

1. **Settings → Accelerator → GPU T4 ×2** (do **not** use P100 — unsupported by current PyTorch)
2. **Settings → Internet → On**
3. Add Input: **Breast Histopathology Images**

---

## Stage 0 — One-time setup (new notebook session)

Run these cells **once** at the start of each Kaggle session.

### Cell 0.1 — Clone repo

```python
import os

if os.path.isdir("/kaggle/working/mbc"):
    !rm -rf /kaggle/working/mbc

!git clone https://github.com/jpmkumar/mbc.git /kaggle/working/mbc
%cd /kaggle/working/mbc
!git pull
!git log -1 --oneline
```

### Cell 0.2 — Install dependencies

```python
!pip install -q -r requirements.txt
```

### Cell 0.3 — Find dataset path

```python
import os

ARCHIVE = None
for root, dirs, _ in os.walk("/kaggle/input"):
    if "IDC_regular_ps50_idx5" in dirs or "10253" in dirs:
        ARCHIVE = root
        break

assert ARCHIVE, "Add dataset: Breast Histopathology Images"
print("ARCHIVE =", repr(ARCHIVE))

patient_dirs = [
    d for d in os.listdir(ARCHIVE)
    if os.path.isdir(os.path.join(ARCHIVE, d)) and d != "IDC_regular_ps50_idx5"
]
print("Patient folders:", len(patient_dirs))
```

### Cell 0.4 — Enable GPU in config

```python
from pathlib import Path

p = Path("configs/histopath.yaml")
text = p.read_text()
if "classical_device: cpu" in text:
    text = text.replace("classical_device: cpu", "classical_device: auto")
    p.write_text(text)
    print("Set classical_device: auto")
else:
    print("GPU config already set:", "classical_device: auto" in text)
```

Verify CUDA:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

You should **not** see `Classical device: cpu` during training.

### Cell 0.5 — Generate 5-fold splits (once per session, or skip if restored)

```python
import os

FOLDS_OK = os.path.isfile("data/splits/histopath/folds/fold_0/train.csv")
if FOLDS_OK:
    print("Folds already exist — skip generation")
else:
    !python data/download/split_histopath_archive.py \
        --archive-path "{ARCHIVE}" \
        --mode cv --folds 5
    !ls -lh data/splits/histopath/folds/fold_0/
```

**Do not interrupt** while indexing 277,524 patches (~5–10 min).

### Cell 0.6 — Backup helper (reuse after every stage)

```python
import shutil
from pathlib import Path
from datetime import datetime

def backup_mbc(label: str = "manual"):
    """Backup results + splits to /kaggle/working/ and create a zip."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup = Path(f"/kaggle/working/mbc_backup_{stamp}_{label}")
    backup.mkdir(parents=True, exist_ok=True)

    src = Path("/kaggle/working/mbc/results/histopath")
    if src.exists():
        shutil.copytree(src, backup / "histopath", dirs_exist_ok=True)
        print("Saved:", backup / "histopath")

    splits = Path("/kaggle/working/mbc/data/splits/histopath")
    if splits.exists():
        shutil.copytree(splits, backup / "splits", dirs_exist_ok=True)
        print("Saved:", backup / "splits")

    zip_path = f"/kaggle/working/mbc_histopath_{stamp}_{label}.zip"
    !cd /kaggle/working && zip -rq {Path(zip_path).name} {backup.name}
    print("Zip:", zip_path)
    !ls -lh {zip_path}
    return zip_path

print("backup_mbc('fold0_e2')  # call after each training stage")
```

---

## Stage 1 — Classical baseline (E2), one fold at a time

Train **one fold per Kaggle session** to reduce disconnect risk.

### Cell 1.0 — Smoke test (optional, ~10 min)

```python
!python scripts/train_histopath_cv.py \
  --fold 0 --quick --max-samples 256 --experiment E2 \
  --archive-path "{ARCHIVE}"
```

### Cell 1.1 — Train fold 0 (E2)

```python
!python scripts/train_histopath_cv.py \
  --fold 0 --experiment E2 \
  --archive-path "{ARCHIVE}"
```

**Expected output:** `Classical device: cuda`, then `stage_a epoch X/Y`, finally test metrics + `cv_summary.json`.

### Cell 1.2 — Backup after fold 0

```python
backup_mbc("fold0_e2")
```

**Download:** File browser → `/kaggle/working/` → `mbc_histopath_*_fold0_e2.zip` → Download.

---

### Cell 1.3 — Train fold 1 (E2)

```python
%cd /kaggle/working/mbc
!git pull   # get latest fixes if any

!python scripts/train_histopath_cv.py \
  --fold 1 --experiment E2 \
  --archive-path "{ARCHIVE}"
```

### Cell 1.4 — Backup after fold 1

```python
backup_mbc("fold1_e2")
```

Download the new zip.

---

### Cell 1.5 — Train fold 2 (E2)

```python
!python scripts/train_histopath_cv.py \
  --fold 2 --experiment E2 \
  --archive-path "{ARCHIVE}"
```

### Cell 1.6 — Backup after fold 2

```python
backup_mbc("fold2_e2")
```

---

### Cell 1.7 — Train fold 3 (E2)

```python
!python scripts/train_histopath_cv.py \
  --fold 3 --experiment E2 \
  --archive-path "{ARCHIVE}"
```

### Cell 1.8 — Backup after fold 3

```python
backup_mbc("fold3_e2")
```

---

### Cell 1.9 — Train fold 4 (E2)

```python
!python scripts/train_histopath_cv.py \
  --fold 4 --experiment E2 \
  --archive-path "{ARCHIVE}"
```

### Cell 1.10 — Backup after fold 4 (full E2 complete)

```python
backup_mbc("fold4_e2_all_classical")
```

---

## Stage 2 — Hybrid quantum (E3), one fold at a time

Same pattern as Stage 1. E3 runs Stage A → B (VQC) → C; **Stage B uses CPU** for quantum.

### Cell 2.1 — Train fold 0 (E3)

```python
!python scripts/train_histopath_cv.py \
  --fold 0 --experiment E3 \
  --archive-path "{ARCHIVE}"
```

### Cell 2.2 — Backup after fold 0 E3

```python
backup_mbc("fold0_e3")
```

---

### Cell 2.3 — Train fold 1 (E3)

```python
!python scripts/train_histopath_cv.py \
  --fold 1 --experiment E3 \
  --archive-path "{ARCHIVE}"
```

### Cell 2.4 — Backup after fold 1 E3

```python
backup_mbc("fold1_e3")
```

---

### Cell 2.5 — Train fold 2 (E3)

```python
!python scripts/train_histopath_cv.py \
  --fold 2 --experiment E3 \
  --archive-path "{ARCHIVE}"
```

### Cell 2.6 — Backup after fold 2 E3

```python
backup_mbc("fold2_e3")
```

---

### Cell 2.7 — Train fold 3 (E3)

```python
!python scripts/train_histopath_cv.py \
  --fold 3 --experiment E3 \
  --archive-path "{ARCHIVE}"
```

### Cell 2.8 — Backup after fold 3 E3

```python
backup_mbc("fold3_e3")
```

---

### Cell 2.9 — Train fold 4 (E3)

```python
!python scripts/train_histopath_cv.py \
  --fold 4 --experiment E3 \
  --archive-path "{ARCHIVE}"
```

### Cell 2.10 — Backup after fold 4 E3

```python
backup_mbc("fold4_e3_all_hybrid")
```

---

## Stage 3 — Friedman comparison (E2 vs E3)

Run **after all folds** are trained for both models, or run per-fold and aggregate manually.

### Cell 3.1 — Compare all folds (long run)

```python
!python scripts/train_histopath_cv.py \
  --compare-classical \
  --archive-path "{ARCHIVE}"
```

This retrains E2 and E3 on **all 5 folds** — use only if you have not already trained each fold individually.

### Cell 3.2 — View aggregated summary

```python
import json
from pathlib import Path

summary = json.loads(Path("results/histopath/cv_summary.json").read_text())
print(json.dumps(summary, indent=2))
```

Look for: `E2_f1_mean`, `E2_f1_std`, `E3_f1_mean`, `E3_f1_std`, `friedman_f1`.

### Cell 3.3 — Backup final summary

```python
backup_mbc("final_cv_summary")
```

---

## Stage 4 — Copy results to Mac

After downloading zips from Kaggle:

```bash
# Unzip on Mac (example)
cd ~/Downloads
unzip mbc_histopath_*_fold0_e2.zip -d mbc_backup_fold0

# Copy into repo (local only, gitignored)
cp -r ~/Downloads/mbc_backup_*/histopath/* \
  /Users/muthu/ResTest/paper1/mbc/results/histopath_kaggle/

cp -r ~/Downloads/mbc_backup_*/splits/* \
  /Users/muthu/ResTest/paper1/mbc/data/splits/histopath_kaggle/
```

Per-fold layout (optional):

```bash
REPO="/Users/muthu/ResTest/paper1/mbc"
cp -r ~/Downloads/mbc_backup_*_fold0_e2/histopath "$REPO/results/histopath_kaggle_fold0/"
cp -r ~/Downloads/mbc_backup_*_fold1_e2/histopath "$REPO/results/histopath_kaggle_fold1/"
# ... fold2, fold3, fold4
```

---

## What gets saved in each backup

```
mbc_backup_YYYYMMDD_HHMM_<label>/
  histopath/
    checkpoints/
      E2_histopath_fold{N}_histopath_seed42.pt       # best weights
      E2_histopath_fold{N}_histopath_seed42_latest.pt
      E3_histopath_fold{N}_...                        # after E3 runs
    cv_summary.json
    E2_histopath_fold{N}_*_metrics.json
    E2_histopath_fold{N}_*_progress.json
    E2_histopath_fold{N}_*_history.json
  splits/
    patient_stats.csv
    split_stats.json
    folds/fold_0..4/{train,test}.csv
```

Each backup includes **all folds completed so far** under `histopath/checkpoints/`.

---

## Session checklist (print this)

```
[ ] GPU T4 ×2 selected (NOT P100)
[ ] Dataset attached
[ ] Cell 0.1–0.4 run (clone, pip, ARCHIVE, GPU config)
[ ] Folds generated (Cell 0.5) OR restored from backup
[ ] backup_mbc() defined (Cell 0.6)
[ ] Train one fold
[ ] backup_mbc("foldN_e2") + download zip
[ ] Copy to Mac when convenient
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Classical device: cpu` + very slow | Settings → **GPU T4 ×2**; run Cell 0.4 |
| P100 / `no kernel image` | Switch to **T4**, not P100 |
| `No module named 'pennylane'` | `!pip install -q -r requirements.txt` |
| `No k-fold manifests found` | Run Cell 0.5 (split generation) |
| `Could not locate histopath archive` | Run Cell 0.3; pass `--archive-path "{ARCHIVE}"` |
| `loss=nan` mid-training | Best checkpoint already saved; use `*_seed42.pt` not `*_latest.pt` |
| `FloatTensor` / `HalfTensor` | `!git pull` (needs commit `b7d1e45+`) |
| Session died | Use downloaded zip; re-run only missing folds |
| Powered off before download | Draft `/kaggle/working/` is wiped — use **Save & Run All** (Option A) or download the ~130 MB zip first (Option B) |
| Backup zip only a few kB | Backup cell ran before training finished; rerun it after training completes |
| **Save & Run All** mid interactive run | Don't click it *during* a live draft run (it starts a separate clean run); use it deliberately as Option A instead |

---

## Training defaults (`configs/histopath.yaml`)

Histopath CV enables real-world-safe evaluation helpers:

| Setting | Default | Purpose |
|---------|---------|---------|
| `early_stopping_patience` | 5 | Stop when val balanced accuracy stalls |
| `grad_clip_norm` | 1.0 | Prevent NaN loss under AMP |
| `tune_threshold` | true | Pick cutoff on **val** only, then apply to test |
| `threshold_metric` | f1 | Metric optimized when tuning threshold |

After `git pull`, folds 2+ use these automatically. Re-run fold 0–1 only if you want comparable threshold-tuned numbers.

---

## E2 reference results (completed folds)

| Fold | Balanced accuracy | F1 | AUC | Threshold | Notes |
|------|-------------------|-----|-----|-----------|-------|
| 0 | 0.880 | 0.807 | 0.949 | 0.50 | 25 epochs, ~5.3 h |
| 1 | 0.873 | 0.800 | 0.946 | 0.50 | 25 epochs |
| 2 | 0.884 | 0.827 | 0.954 | 0.55 | early stop @ ep 12, val-tuned threshold |

Folds 0–1 used the fixed 0.5 threshold; fold 2+ use early stopping + val-tuned threshold.

---

## Citation

```
Sawyer-Lee, R., et al. (2016). CBIS-DDSM [Data set]. TCIA.
https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
```

IDC patches: Kaggle **Breast Histopathology Images** (Paul Mooney).
