# Kaggle: end-to-end VQC width ablation

This runbook executes one immutable `(fold, width)` cell from
`preregistration/histopath_vqc_width_protocol.md`. Fold 0 is already
exploratory. The eight outstanding replications are folds 1–4 at 4 and
12 qubits. Run one pair per Kaggle version.

Before running:

1. Attach the **Breast Histopathology Images** dataset.
2. Set **Accelerator: GPU T4 ×2** and **Internet: On**.
3. Set `FOLD` and `N_QUBITS` in Cell 1.
4. Choose **Save Version → Save & Run All**.
5. Do not change learning rates, epochs or any other setting by width.

## Cell 1 — Clone the declared code

```python
import os
import shutil
import subprocess
from pathlib import Path

FOLD = 1       # allowed: 1, 2, 3, 4
N_QUBITS = 4   # allowed: 4, 12

assert FOLD in {1, 2, 3, 4}
assert N_QUBITS in {4, 12}

REPO = Path("/kaggle/working/mbc")
BRANCH = "docs/histopath-writing-q1-guidelines"

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    [
        "git", "clone", "--branch", BRANCH, "--single-branch",
        "https://github.com/jpmkumar/mbc.git", str(REPO),
    ],
    check=True,
)
os.chdir(REPO)
COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
assert Path("preregistration/histopath_vqc_width_protocol.md").exists()
print("Repository:", REPO)
print("Commit:", COMMIT)
print("Fold:", FOLD, "Qubits:", N_QUBITS)
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

## Cell 3 — Locate images and verify the fixed configuration

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
assert config["model"]["quantum"]["n_layers"] == 2
assert config["model"]["quantum"]["entanglement"] == "linear"
assert config["model"]["quantum"]["encoding"] == "angle_y"
assert config["model"]["quantum"]["data_reuploading"] is False
assert config["training"]["loss"] == "focal"
assert config["training"]["tta"] is True
assert torch.cuda.is_available(), "Select GPU T4 ×2 before running."

print("Archive:", ARCHIVE)
print("GPU:", torch.cuda.get_device_name(0))
print("Declared width:", N_QUBITS)
```

## Cell 4 — Generate and verify the outer patient fold

```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

fold_dir = Path(f"data/splits/histopath/folds/fold_{FOLD}")
if not (fold_dir / "train.csv").exists():
    subprocess.run(
        [
            sys.executable,
            "data/download/split_histopath_archive.py",
            "--archive-path", str(ARCHIVE),
            "--mode", "cv",
            "--folds", "5",
        ],
        check=True,
    )

outer_patient_sets = {}
for split in ("train", "test"):
    frame = pd.read_csv(fold_dir / f"{split}.csv")
    outer_patient_sets[split] = set(frame["patient_id"].astype(str))
    print(
        split,
        "patches:", len(frame),
        "patients:", len(outer_patient_sets[split]),
    )

assert not outer_patient_sets["train"] & outer_patient_sets["test"]
assert len(set.union(*outer_patient_sets.values())) == 279
print("Outer train/test patient overlap: none")
print("Validation will be carved from outer train at runtime.")
```

## Cell 5 — Train all three stages

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "scripts/train_histopath_cv.py",
        "--fold", str(FOLD),
        "--experiment", "E3",
        "--seed", "42",
        "--archive-path", str(ARCHIVE),
        "--n-qubits", str(N_QUBITS),
    ],
    check=True,
)
```

## Cell 6 — Verify width, provenance and stage attribution

```python
import json
from pathlib import Path

import pandas as pd

summary_path = Path("results/histopath/cv_summary.json")
summary = json.loads(summary_path.read_text())
record = summary["results"]["E3"][0]

assert record["fold"] == FOLD
assert record["seed"] == 42
assert record["n_qubits"] == N_QUBITS
assert record["n_layers"] == 2
assert record["entanglement"] == "linear"
assert record["encoding"] == "angle_y"
assert record["data_reuploading"] is False

stage_path = Path(record["train_metrics"]["stage_comparison_path"])
assert stage_path.exists(), f"Missing stage attribution: {stage_path}"
stage = json.loads(stage_path.read_text())
assert set(stage["stages"]) == {"stage_a", "stage_b", "stage_c"}
assert stage["global_best_stage"] in stage["stages"]

runtime_dir = Path(f"data/splits/histopath/runtime/fold_{FOLD}")
patient_sets = {}
for split in ("train", "val", "test"):
    frame = pd.read_csv(runtime_dir / f"{split}.csv")
    patient_sets[split] = set(frame["patient_id"].astype(str))
    print(split, "patches:", len(frame), "patients:", len(patient_sets[split]))

assert not patient_sets["train"] & patient_sets["val"]
assert not patient_sets["train"] & patient_sets["test"]
assert not patient_sets["val"] & patient_sets["test"]
assert len(set.union(*patient_sets.values())) == 279

REPORT = {
    "commit": COMMIT,
    "fold": FOLD,
    "n_qubits": N_QUBITS,
    "seed": 42,
    "effective_seed": 42 + FOLD,
    "patient_counts": {key: len(value) for key, value in patient_sets.items()},
    "patient_overlap": "none",
    "test_metrics": record["test_metrics"],
    "selected_stage": stage["global_best_stage"],
    "stage_comparison": stage,
}
REPORT_PATH = Path(
    f"results/histopath/width_q{N_QUBITS}_fold{FOLD}_report.json"
)
REPORT_PATH.write_text(json.dumps(REPORT, indent=2))
print(json.dumps(REPORT, indent=2))
```

## Cell 7 — Build the complete output bundle

```python
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
BUNDLE = Path(
    f"/kaggle/working/histopath_width_q{N_QUBITS}_fold{FOLD}_{STAMP}"
)
BUNDLE.mkdir(parents=True, exist_ok=False)

shutil.copytree("results/histopath", BUNDLE / "histopath")
shutil.copytree(
    f"data/splits/histopath/runtime/fold_{FOLD}",
    BUNDLE / "runtime_splits",
)
shutil.copy2(
    "preregistration/histopath_vqc_width_protocol.md",
    BUNDLE / "histopath_vqc_width_protocol.md",
)
(BUNDLE / "git_commit.txt").write_text(COMMIT + "\n")
(BUNDLE / "run_metadata.json").write_text(
    json.dumps(
        {
            "commit": COMMIT,
            "fold": FOLD,
            "n_qubits": N_QUBITS,
            "seed": 42,
            "experiment": "E3_end_to_end_width_ablation",
            "archive": str(ARCHIVE),
            "accelerator": "GPU_T4_x2",
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

Expected runtime per immutable Kaggle version:

- **q4:** approximately 4–5.5 hours.
- **q12:** approximately 5.5–7.5 hours.

Both require GPU T4 ×2 and Save & Run All. Across the eight outstanding
fold-width pairs, budget approximately 42–55 GPU-session hours.
