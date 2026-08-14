#!/usr/bin/env python3
"""Run one declared end-to-end histopathology width cell on Kaggle.

This is the committed implementation of
``preregistration/histopath_vqc_width_protocol.md``. It locates the attached
dataset, verifies the fixed configuration and patient split, trains one E3
fold-width pair, verifies stage attribution, and creates a downloadable ZIP.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FOLDS = {1, 2, 3, 4}
ALLOWED_WIDTHS = {4, 12}


def validate_pair(fold: int, n_qubits: int) -> None:
    if fold not in ALLOWED_FOLDS:
        raise ValueError(f"Fold must be one of {sorted(ALLOWED_FOLDS)}.")
    if n_qubits not in ALLOWED_WIDTHS:
        raise ValueError(f"Width must be one of {sorted(ALLOWED_WIDTHS)}.")


def find_archive(input_root: Path) -> Path:
    for root, directories, _ in os.walk(input_root):
        if "IDC_regular_ps50_idx5" in directories or "10253" in directories:
            return Path(root)
    raise FileNotFoundError(
        "Attach the Breast Histopathology Images dataset to this Kaggle notebook."
    )


def build_train_command(
    fold: int,
    n_qubits: int,
    archive: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_histopath_cv.py",
        "--fold",
        str(fold),
        "--experiment",
        "E3",
        "--seed",
        "42",
        "--archive-path",
        str(archive),
        "--n-qubits",
        str(n_qubits),
    ]


def patient_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    return set(frame["patient_id"].astype(str))


def verify_outer_fold(fold: int) -> None:
    fold_dir = ROOT / f"data/splits/histopath/folds/fold_{fold}"
    train = patient_ids(fold_dir / "train.csv")
    test = patient_ids(fold_dir / "test.csv")
    if train & test:
        raise RuntimeError("Patient leakage between outer train and test.")
    if len(train | test) != 279:
        raise RuntimeError(f"Expected 279 patients, found {len(train | test)}.")
    print(f"Outer split: train={len(train)} patients, test={len(test)}, overlap=0")


def verify_runtime_fold(fold: int) -> dict[str, int]:
    runtime_dir = ROOT / f"data/splits/histopath/runtime/fold_{fold}"
    groups = {
        split: patient_ids(runtime_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    if (
        groups["train"] & groups["val"]
        or groups["train"] & groups["test"]
        or groups["val"] & groups["test"]
    ):
        raise RuntimeError("Patient leakage in runtime train/val/test split.")
    if len(set.union(*groups.values())) != 279:
        raise RuntimeError("Runtime split does not contain exactly 279 patients.")
    return {split: len(values) for split, values in groups.items()}


def verify_config() -> None:
    config = yaml.safe_load((ROOT / "configs/histopath.yaml").read_text())
    quantum = config["model"]["quantum"]
    training = config["training"]
    expected = {
        "n_layers": 2,
        "entanglement": "linear",
        "encoding": "angle_y",
        "data_reuploading": False,
    }
    observed = {key: quantum[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Quantum config changed: {observed} != {expected}")
    if training["classical_device"] != "auto":
        raise RuntimeError("classical_device must remain auto.")
    if training["loss"] != "focal" or training["tta"] is not True:
        raise RuntimeError("The declared focal-loss/TTA bundle has changed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select GPU T4 ×2 before running.")
    print("GPU:", torch.cuda.get_device_name(0))


def select_result(summary: dict, fold: int, n_qubits: int) -> dict:
    matches = [
        row
        for row in summary["results"]["E3"]
        if row["fold"] == fold and row["n_qubits"] == n_qubits
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one q{n_qubits} fold-{fold} result, found {len(matches)}."
        )
    return matches[0]


def checkpoint_finiteness(path: Path) -> dict:
    """Audit a stage checkpoint without treating NaN predictions as zero."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = (
        payload.get("model_state_dict")
        or payload.get("best_state_dict")
        or payload.get("state_dict")
    )
    if not isinstance(state, dict):
        raise RuntimeError(f"No model state found in {path}.")

    total_values = 0
    nonfinite_values = 0
    affected_tensors = []
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        total_values += value.numel()
        count = int((~torch.isfinite(value)).sum().item())
        if count:
            nonfinite_values += count
            affected_tensors.append(
                {"name": name, "nonfinite": count, "values": value.numel()}
            )
    return {
        "checkpoint": str(path),
        "numerically_valid": nonfinite_values == 0,
        "total_values": total_values,
        "nonfinite_values": nonfinite_values,
        "affected_tensor_count": len(affected_tensors),
        "affected_tensors": affected_tensors,
    }


def verify_result(fold: int, n_qubits: int, commit: str) -> tuple[dict, Path]:
    summary_path = ROOT / "results/histopath/cv_summary.json"
    summary = json.loads(summary_path.read_text())
    record = select_result(summary, fold, n_qubits)
    if record["seed"] != 42:
        raise RuntimeError("Unexpected seed in result.")
    expected_circuit = {
        "n_layers": 2,
        "entanglement": "linear",
        "encoding": "angle_y",
        "data_reuploading": False,
    }
    for key, expected in expected_circuit.items():
        if record[key] != expected:
            raise RuntimeError(f"Unexpected {key}: {record[key]!r}")

    stage_path = ROOT / record["train_metrics"]["stage_comparison_path"]
    stage = json.loads(stage_path.read_text())
    if set(stage["stages"]) != {"stage_a", "stage_b", "stage_c"}:
        raise RuntimeError("Stage attribution is incomplete.")
    if stage["global_best_stage"] not in stage["stages"]:
        raise RuntimeError("Selected stage is absent from stage attribution.")

    stage_numerics = {}
    for name, stage_record in stage["stages"].items():
        checkpoint_path = ROOT / stage_record["checkpoint"]
        audit = checkpoint_finiteness(checkpoint_path)
        stage_numerics[name] = audit
        if not audit["numerically_valid"]:
            print(
                f"WARNING: {name} is a numerical convergence failure: "
                f"{audit['nonfinite_values']} non-finite checkpoint values. "
                "Its derived metrics must not be interpreted as performance."
            )

    selected_audit = stage_numerics[stage["global_best_stage"]]
    if not selected_audit["numerically_valid"]:
        raise RuntimeError("Validation selected a numerically invalid stage.")

    report = {
        "commit": commit,
        "fold": fold,
        "n_qubits": n_qubits,
        "seed": 42,
        "effective_seed": 42 + fold,
        "patient_counts": verify_runtime_fold(fold),
        "patient_overlap": "none",
        "test_metrics": record["test_metrics"],
        "selected_stage": stage["global_best_stage"],
        "stage_numerics": stage_numerics,
        "stage_comparison": stage,
    }
    report_path = (
        ROOT
        / f"results/histopath/width_q{n_qubits}_fold{fold}_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report, report_path


def build_bundle(
    fold: int,
    n_qubits: int,
    archive: Path,
    commit: str,
    output_root: Path,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    name = f"histopath_width_q{n_qubits}_fold{fold}_{stamp}"
    bundle = output_root / name
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "results/histopath", bundle / "histopath")
    shutil.copytree(
        ROOT / f"data/splits/histopath/runtime/fold_{fold}",
        bundle / "runtime_splits",
    )
    shutil.copy2(
        ROOT / "preregistration/histopath_vqc_width_protocol.md",
        bundle / "histopath_vqc_width_protocol.md",
    )
    (bundle / "git_commit.txt").write_text(commit + "\n")
    (bundle / "run_metadata.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "fold": fold,
                "n_qubits": n_qubits,
                "seed": 42,
                "experiment": "E3_end_to_end_width_ablation",
                "archive": str(archive),
                "accelerator": "GPU_T4_x2",
            },
            indent=2,
        )
        + "\n"
    )
    zip_path = Path(
        shutil.make_archive(str(bundle), "zip", bundle.parent, bundle.name)
    )
    print("Download:", zip_path)
    print("Size: %.2f MB" % (zip_path.stat().st_size / (1024 * 1024)))
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the train command without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_pair(args.fold, args.n_qubits)
    archive = find_archive(args.input_root)
    command = build_train_command(args.fold, args.n_qubits, archive)
    if args.dry_run:
        print(" ".join(command))
        return

    declaration = ROOT / "preregistration/histopath_vqc_width_protocol.md"
    if not declaration.exists():
        raise RuntimeError("Width-ablation declaration is missing.")
    verify_config()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    fold_manifest = ROOT / f"data/splits/histopath/folds/fold_{args.fold}/train.csv"
    if not fold_manifest.exists():
        subprocess.run(
            [
                sys.executable,
                "data/download/split_histopath_archive.py",
                "--archive-path",
                str(archive),
                "--mode",
                "cv",
                "--folds",
                "5",
            ],
            cwd=ROOT,
            check=True,
        )
    verify_outer_fold(args.fold)

    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    verify_result(args.fold, args.n_qubits, commit)
    build_bundle(
        args.fold,
        args.n_qubits,
        archive,
        commit,
        args.output_root,
    )


if __name__ == "__main__":
    main()
