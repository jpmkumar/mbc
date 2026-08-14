#!/usr/bin/env python3
"""Run one server-primary end-to-end histopathology width cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALLOWED_FOLDS = {1, 2, 3, 4}
ALLOWED_WIDTHS = {4, 8, 12}
PROTOCOL = ROOT / "preregistration/histopath_vqc_width_server_protocol.md"
EXPECTED_LOCK_SHA = (
    "55ae29e55a5e3643fb59be8e3aaa2c1466e63efcd3559bc22b4addab4e7c829a"
)
EXPECTED_SPLIT_SHA = (
    "ac9d06510ca3555e6d481f1f870ab92fc69411ee3b9fa53da9aa7a60ce9bd013"
)
EXPECTED_GPU = "NVIDIA RTX A4000"

from scripts.run_histopath_width_kaggle import (  # noqa: E402
    checkpoint_finiteness,
    select_result,
)


def validate_pair(fold: int, n_qubits: int) -> None:
    if fold not in ALLOWED_FOLDS:
        raise ValueError(f"Fold must be one of {sorted(ALLOWED_FOLDS)}.")
    if n_qubits not in ALLOWED_WIDTHS:
        raise ValueError(f"Width must be one of {sorted(ALLOWED_WIDTHS)}.")


def build_train_command(
    fold: int,
    n_qubits: int,
    archive: Path,
    splits_dir: Path,
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
        "--splits-dir",
        str(splits_dir),
        "--n-qubits",
        str(n_qubits),
    ]


def patient_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    return set(frame["patient_id"].astype(str))


def verify_outer_fold(splits_dir: Path, fold: int) -> dict[str, int]:
    fold_dir = splits_dir / "folds" / f"fold_{fold}"
    train = patient_ids(fold_dir / "train.csv")
    test = patient_ids(fold_dir / "test.csv")
    if train & test:
        raise RuntimeError("Patient leakage between outer train and test.")
    if len(train | test) != 279:
        raise RuntimeError(f"Expected 279 patients, found {len(train | test)}.")
    return {"outer_train": len(train), "outer_test": len(test)}


def verify_runtime_fold(splits_dir: Path, fold: int) -> dict[str, int]:
    runtime_dir = splits_dir / "runtime" / f"fold_{fold}"
    groups = {
        split: patient_ids(runtime_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    if any(
        groups[left] & groups[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("Patient leakage in runtime train/val/test split.")
    if len(set.union(*groups.values())) != 279:
        raise RuntimeError("Runtime split does not contain exactly 279 patients.")
    return {split: len(values) for split, values in groups.items()}


def verify_config() -> None:
    config = yaml.safe_load((ROOT / "configs/histopath.yaml").read_text())
    quantum = config["model"]["quantum"]
    expected = {
        "n_layers": 2,
        "entanglement": "linear",
        "encoding": "angle_y",
        "data_reuploading": False,
    }
    observed = {key: quantum[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Quantum config changed: {observed} != {expected}")
    training = config["training"]
    if training["classical_device"] != "auto":
        raise RuntimeError("classical_device must remain auto.")
    if training["loss"] != "focal" or training["tta"] is not True:
        raise RuntimeError("The declared focal-loss/TTA bundle has changed.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")


def required_environment() -> dict[str, str]:
    names = (
        "MBC_GIT_COMMIT",
        "MBC_IMAGE_ID",
        "MBC_ENV_LOCK_SHA",
        "MBC_DATASET_SHA",
        "MBC_SPLIT_MANIFEST_SHA",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing provenance variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def nvidia_smi_metadata() -> list[dict[str, str]]:
    fields = (
        "name,uuid,driver_version,memory.total,"
        "compute_cap,pci.bus_id"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    keys = fields.split(",")
    return [
        dict(zip(keys, (part.strip() for part in line.split(",")), strict=True))
        for line in output.splitlines()
        if line.strip()
    ]


def capture_provenance(fold: int, n_qubits: int) -> dict:
    environment = required_environment()
    commit = environment["MBC_GIT_COMMIT"]
    if len(commit) != 40:
        raise RuntimeError("MBC_GIT_COMMIT must be a full 40-character commit.")
    if environment["MBC_ENV_LOCK_SHA"] != EXPECTED_LOCK_SHA:
        raise RuntimeError("The dependency lock is not the qualified server lock.")
    if environment["MBC_SPLIT_MANIFEST_SHA"] != EXPECTED_SPLIT_SHA:
        raise RuntimeError("The split manifest differs from the canonical server split.")
    for name in ("MBC_DATASET_SHA", "MBC_ENV_LOCK_SHA", "MBC_SPLIT_MANIFEST_SHA"):
        value = environment[name]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RuntimeError(f"{name} must be a lowercase SHA-256 digest.")
    gpus = nvidia_smi_metadata()
    if len(gpus) != 1 or gpus[0]["name"] != EXPECTED_GPU:
        raise RuntimeError(f"Expected one {EXPECTED_GPU}; observed {gpus}.")
    return {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "n_qubits": n_qubits,
        "seed": 42,
        "effective_seed": 42 + fold,
        "source_commit": commit,
        "image_id": environment["MBC_IMAGE_ID"],
        "environment_lock_sha256": environment["MBC_ENV_LOCK_SHA"],
        "dataset_sha256": environment["MBC_DATASET_SHA"],
        "split_manifest_sha256": environment["MBC_SPLIT_MANIFEST_SHA"],
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
    }


def verify_result(
    fold: int,
    n_qubits: int,
    splits_dir: Path,
    provenance: dict,
) -> tuple[dict, Path]:
    summary_path = ROOT / "results/histopath/cv_summary.json"
    record = select_result(json.loads(summary_path.read_text()), fold, n_qubits)
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
        audit = checkpoint_finiteness(ROOT / stage_record["checkpoint"])
        stage_numerics[name] = audit
        if not audit["numerically_valid"]:
            print(
                f"WARNING: {name} is a numerical convergence failure with "
                f"{audit['nonfinite_values']} non-finite checkpoint values."
            )
    if not stage_numerics[stage["global_best_stage"]]["numerically_valid"]:
        raise RuntimeError("Validation selected a numerically invalid stage.")

    report = {
        "provenance": provenance,
        "patient_counts": verify_runtime_fold(splits_dir, fold),
        "patient_overlap": "none",
        "selected_stage": stage["global_best_stage"],
        "test_metrics": record["test_metrics"],
        "stage_numerics": stage_numerics,
        "stage_comparison": stage,
    }
    path = ROOT / (
        f"results/histopath/width_server_q{n_qubits}_fold{fold}_report.json"
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report, path


def build_bundle(
    fold: int,
    n_qubits: int,
    splits_dir: Path,
    output_root: Path,
    provenance: dict,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    name = f"histopath_server_width_q{n_qubits}_fold{fold}_{stamp}"
    bundle = output_root / name
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "results/histopath", bundle / "histopath")
    shutil.copytree(
        splits_dir / "runtime" / f"fold_{fold}",
        bundle / "runtime_splits",
    )
    shutil.copy2(PROTOCOL, bundle / PROTOCOL.name)
    (bundle / "server_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    zip_path = Path(
        shutil.make_archive(str(bundle), "zip", bundle.parent, bundle.name)
    )
    print(f"Bundle: {zip_path}")
    print("Size: %.2f MB" % (zip_path.stat().st_size / (1024 * 1024)))
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "data/splits/histopath",
    )
    parser.add_argument("--output-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_pair(args.fold, args.n_qubits)
    if not args.archive_path.exists():
        raise FileNotFoundError(args.archive_path)
    if not PROTOCOL.exists():
        raise RuntimeError("Server-primary width declaration is missing.")
    command = build_train_command(
        args.fold,
        args.n_qubits,
        args.archive_path,
        args.splits_dir,
    )
    if args.dry_run:
        print(" ".join(map(str, command)))
        return

    verify_config()
    outer_counts = verify_outer_fold(args.splits_dir, args.fold)
    provenance = capture_provenance(args.fold, args.n_qubits)
    provenance["outer_patient_counts"] = outer_counts
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    report, _ = verify_result(
        args.fold,
        args.n_qubits,
        args.splits_dir,
        provenance,
    )
    print(json.dumps(report, indent=2))
    build_bundle(
        args.fold,
        args.n_qubits,
        args.splits_dir,
        args.output_root,
        provenance,
    )


if __name__ == "__main__":
    main()

