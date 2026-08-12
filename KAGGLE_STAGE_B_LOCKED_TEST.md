# Kaggle: validation-locked Stage-B test evaluation

This notebook performs the one-time Fold-0 test evaluation after the Stage-B
pilot has locked its settings. It does not tune thresholds or hyperparameters on
the test split.

Before running:

1. Upload `vqc_stage_b_pilot_20260812_0311.zip` as a private Kaggle Dataset.
2. Attach that dataset and `stage-a-full-cache-20260810-2140` to the notebook.
3. Set **Accelerator: None** and leave **Internet: On**.
4. Add the eight cells below in order and choose **Save Version → Save & Run All**.

## Cell 1 — Clone the locked evaluator

```python
import os
import shutil
import subprocess
from pathlib import Path

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
```

## Cell 2 — Install dependencies

```python
import sys
import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)
print("Dependencies installed")
```

## Cell 3 — Locate both inputs

```python
import zipfile
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
EXTRACT_ROOT = Path("/kaggle/working/stage_b_pilot_input")
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

locked_candidates = list(INPUT_ROOT.rglob("locked_pilot_selection.json"))
if not locked_candidates:
    pilot_archives = [
        path
        for path in INPUT_ROOT.rglob("*.zip")
        if "vqc_stage_b_pilot" in path.name
    ]
    if not pilot_archives:
        raise FileNotFoundError(
            "Attach the vqc_stage_b_pilot_20260812_0311 dataset."
        )
    for archive in pilot_archives:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(EXTRACT_ROOT)
    locked_candidates = list(
        EXTRACT_ROOT.rglob("locked_pilot_selection.json")
    )

feature_candidates = list(INPUT_ROOT.rglob("*_features.pt"))
if len(feature_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one Stage-A feature cache, found {feature_candidates}"
    )
if len(locked_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one locked selection, found {locked_candidates}"
    )

FEATURE_CACHE = feature_candidates[0]
LOCKED_SELECTION = locked_candidates[0]
PILOT_DIR = LOCKED_SELECTION.parent

print("Feature cache:", FEATURE_CACHE)
print("Pilot directory:", PILOT_DIR)
print("Locked selection:", LOCKED_SELECTION)
```

## Cell 4 — Verify the locked protocol

```python
import json
import torch

cache = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
locked = json.loads(LOCKED_SELECTION.read_text())

assert cache["metadata"]["source_stage"] == "stage_a"
assert locked["selection_endpoint"] == "validation_auprc"
assert locked["held_out_test_evaluated"] is False
assert set(locked["best_by_model"]) == {"mlp", "vqc"}

print("Source stage:", cache["metadata"]["source_stage"])
print("Test samples:", len(cache["test"]["labels"]))
print("Locked MLP:", locked["best_by_model"]["mlp"])
print("Locked VQC:", locked["best_by_model"]["vqc"])
```

## Cell 5 — Run the one-time held-out evaluation

```python
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/mbc/results/histopath/vqc_stage_b_locked_test")

subprocess.run(
    [
        sys.executable,
        "scripts/evaluate_vqc_stage_b_locked.py",
        "--feature-cache",
        str(FEATURE_CACHE),
        "--pilot-dir",
        str(PILOT_DIR),
        "--locked-selection",
        str(LOCKED_SELECTION),
        "--output-dir",
        str(OUTPUT_DIR),
        "--seeds",
        "42",
        "43",
        "44",
        "--batch-size",
        "256",
    ],
    check=True,
)
```

## Cell 6 — Display the result

```python
import json

SUMMARY_PATH = OUTPUT_DIR / "locked_test_summary.json"
summary = json.loads(SUMMARY_PATH.read_text())

print(json.dumps(summary["by_model"], indent=2))
print(
    "Mean VQC - MLP test AUPRC:",
    f"{summary['mean_vqc_minus_mlp_test_auprc']:+.6f}",
)
print("Verdict:", summary["verdict"])
print("Uncertainty:", summary["uncertainty_note"])
```

## Cell 7 — Build the result bundle

```python
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
BUNDLE = Path(f"/kaggle/working/vqc_stage_b_locked_test_{STAMP}")
BUNDLE.mkdir(parents=True, exist_ok=False)

shutil.copytree(OUTPUT_DIR, BUNDLE / "locked_test")
shutil.copy2(LOCKED_SELECTION, BUNDLE / "locked_pilot_selection.json")
(BUNDLE / "git_commit.txt").write_text(COMMIT + "\n")
(BUNDLE / "run_metadata.json").write_text(
    json.dumps(
        {
            "commit": COMMIT,
            "feature_cache": str(FEATURE_CACHE),
            "pilot_dir": str(PILOT_DIR),
            "accelerator": "none_cpu",
            "test_time_tuning": False,
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

ZIP_PATH = Path(shutil.make_archive(str(BUNDLE), "zip", BUNDLE.parent, BUNDLE.name))
print("Download:", ZIP_PATH)
print("Size: %.2f MB" % (ZIP_PATH.stat().st_size / (1024 * 1024)))
```

Expected runtime: approximately **5–10 minutes on CPU**. A GPU is neither
required nor used by this evaluator.
