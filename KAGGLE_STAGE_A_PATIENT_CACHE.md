# Kaggle: patient-aware Stage-A cache for one fold

This run creates an independent Stage-A representation for a single patient
fold while preserving patient IDs for clustered uncertainty analysis. It does
not run Stage B or C. All five patient folds were completed on 2026-08-13;
retain these cells as the reproducibility runbook.

Before running:

1. Attach the **Breast Histopathology Images** dataset.
2. Set **Accelerator: GPU T4 ×2** and **Internet: On**.
3. Ensure at least **6 GPU-hours** remain; do not start with only 4 h 10 min.
4. Set `FOLD` in Cell 1 to the fold you are generating.
5. Add the eight cells below in order and choose **Save Version → Save & Run All**.

## Cell 1 — Clone the repository

```python
import os
import shutil
import subprocess
from pathlib import Path

FOLD = 4  # replace only when intentionally reproducing another fold
REPO = Path("/kaggle/working/mbc")
BRANCH = "docs/histopath-writing-q1-guidelines"

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    [
        "git",
        "clone",
        "--branch",
        BRANCH,
        "--single-branch",
        "https://github.com/jpmkumar/mbc.git",
        str(REPO),
    ],
    check=True,
)
os.chdir(REPO)
COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
print("Repository:", REPO)
print("Commit:", COMMIT)
print("Fold:", FOLD)
```

## Cell 2 — Install dependencies

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)
print("Dependencies installed")
```

## Cell 3 — Locate the images and verify GPU

```python
import os
from pathlib import Path

import torch
import yaml

ARCHIVE = None
for root, directories, _ in os.walk("/kaggle/input"):
    if "IDC_regular_ps50_idx5" in directories or "10253" in directories:
        ARCHIVE = Path(root)
        break

if ARCHIVE is None:
    raise FileNotFoundError("Attach the Breast Histopathology Images dataset.")

config = yaml.safe_load(Path("configs/histopath.yaml").read_text())
assert config["training"]["classical_device"] == "auto"
assert torch.cuda.is_available(), "Select GPU T4 ×2 before running."

print("Archive:", ARCHIVE)
print("GPU:", torch.cuda.get_device_name(0))
print("Classical device:", config["training"]["classical_device"])
```

## Cell 4 — Generate patient-level folds

```python
import subprocess
import sys
from pathlib import Path

fold_manifest = Path(f"data/splits/histopath/folds/fold_{FOLD}/train.csv")
if not fold_manifest.exists():
    subprocess.run(
        [
            sys.executable,
            "data/download/split_histopath_archive.py",
            "--archive-path",
            str(ARCHIVE),
            "--mode",
            "cv",
            "--folds",
            "5",
        ],
        check=True,
    )

assert fold_manifest.exists()
print(f"Fold-{FOLD} manifest:", fold_manifest)
```

## Cell 5 — Train Stage A and export the patient-aware cache

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "scripts/train_histopath_cv.py",
        "--fold",
        str(FOLD),
        "--experiment",
        "E3",
        "--seed",
        "42",
        "--archive-path",
        str(ARCHIVE),
        "--stage-a-only-cache",
    ],
    check=True,
)
```

## Cell 6 — Verify provenance and patient alignment

```python
import json
from pathlib import Path

import torch

cache_candidates = list(
    Path("results/histopath/feature_cache").glob("*_features.pt")
)
if len(cache_candidates) != 1:
    raise RuntimeError(f"Expected one feature cache, found {cache_candidates}")

FEATURE_CACHE = cache_candidates[0]
cache = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
metadata = cache["metadata"]

assert metadata["source_stage"] == "stage_a"
assert metadata["feature_cache_format_version"] == 3
assert metadata["contains_patient_ids"] is True

report = {
    "feature_cache": str(FEATURE_CACHE),
    "commit": COMMIT,
    "fold": FOLD,
    "source_stage": metadata["source_stage"],
    "source_checkpoint": metadata["source_checkpoint"],
    "splits": {},
}
patients_by_split = {}

for split in ("train", "val", "test"):
    payload = cache[split]
    patient_ids = payload["patient_ids"]
    labels = payload["labels"]
    assert len(patient_ids) == len(labels) == len(payload["features"])
    assert all(not value.startswith("sample_") for value in patient_ids)
    patients_by_split[split] = set(patient_ids)
    report["splits"][split] = {
        "samples": len(labels),
        "class_counts": torch.bincount(labels, minlength=2).tolist(),
        "unique_patients": len(set(patient_ids)),
    }

for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = patients_by_split[left] & patients_by_split[right]
    assert not overlap, f"Patient leakage between {left} and {right}: {overlap}"
report["patient_overlap"] = "none"

REPORT_PATH = Path(f"results/histopath/fold{FOLD}_patient_cache_report.json")
REPORT_PATH.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
```

## Cell 7 — Build the output bundle

```python
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
BUNDLE = Path(f"/kaggle/working/stage_a_patient_cache_fold{FOLD}_{STAMP}")
BUNDLE.mkdir(parents=True, exist_ok=False)

shutil.copy2(FEATURE_CACHE, BUNDLE / FEATURE_CACHE.name)
shutil.copy2(REPORT_PATH, BUNDLE / REPORT_PATH.name)

summary = Path("results/histopath/cv_summary.json")
if summary.exists():
    shutil.copy2(summary, BUNDLE / summary.name)

checkpoint = Path(cache["metadata"]["source_checkpoint"])
if checkpoint.exists():
    shutil.copy2(checkpoint, BUNDLE / checkpoint.name)

runtime_splits = Path(f"data/splits/histopath/runtime/fold_{FOLD}")
if runtime_splits.exists():
    shutil.copytree(runtime_splits, BUNDLE / "runtime_splits")

(BUNDLE / "git_commit.txt").write_text(COMMIT + "\n")
(BUNDLE / "run_metadata.json").write_text(
    json.dumps(
        {
            "commit": COMMIT,
            "fold": FOLD,
            "experiment": "E3_stage_a_only",
            "archive": str(ARCHIVE),
            "accelerator": "GPU_T4_x2",
            "patient_ids_preserved": True,
        },
        indent=2,
    )
)
print("Bundle:", BUNDLE)
```

## Cell 8 — Create the downloadable ZIP

```python
import shutil
from pathlib import Path

ZIP_PATH = Path(
    shutil.make_archive(str(BUNDLE), "zip", BUNDLE.parent, BUNDLE.name)
)
print("Download:", ZIP_PATH)
print("Size: %.2f MB" % (ZIP_PATH.stat().st_size / (1024 * 1024)))
```

Expected runtime: **2.5–5.5 hours**, dominated by Stage-A epochs before early
stopping plus roughly 80 minutes to extract features for all 277,524 patches.
Fold 1 stopped after 7 epochs and finished in about 2 h 10 min. A **GPU T4 ×2
session** is required; the quantum simulator is not used in this run.

## After the download

The matched MLP/VQC comparison on the resulting cache runs locally on CPU in
about eight minutes. See `STAGE_B_LOCKED_PROTOCOL.md`.
