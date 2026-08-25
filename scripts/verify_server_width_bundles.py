#!/usr/bin/env python3
"""Independently verify the twelve server-primary width bundles.

Each bundle ships its own ``width_server_q*_fold*_report.json``, but that report
was written by the same container that produced the result. This script re-derives
every claim the primary analysis depends on from the raw artifacts: patient splits
from the runtime CSVs, checkpoint finiteness from ``torch.isfinite``, and the
environment invariants required by
``preregistration/histopath_vqc_width_server_protocol.md``.

The emitted matrix is the only input the width aggregation is allowed to read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "preregistration/histopath_vqc_width_server_protocol.md"

WIDTHS = (4, 8, 12)
FOLDS = (1, 2, 3, 4)
STAGES = ("stage_a", "stage_b", "stage_c")

TOTAL_PATIENTS = 279
BASE_SEED = 42

# Declared in the server protocol; a cell that disagrees is not part of the
# immutable environment and cannot be pooled with the rest of the matrix.
DECLARED = {
    "environment_lock_sha256": (
        "55ae29e55a5e3643fb59be8e3aaa2c1466e63efcd3559bc22b4addab4e7c829a"
    ),
    "split_manifest_sha256": (
        "ac9d06510ca3555e6d481f1f870ab92fc69411ee3b9fa53da9aa7a60ce9bd013"
    ),
}
DECLARED_GPU = "NVIDIA RTX A4000"
DECLARED_DRIVER = "580.173.02"

METRIC_KEYS = (
    "auprc",
    "auc",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "threshold",
    "n_samples",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(path: Path) -> tuple[set[str], int]:
    """Return (patient ids, patch count) for one runtime split CSV."""
    patients: set[str] = set()
    patches = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            patients.add(row["patient_id"])
            patches += 1
    return patients, patches


def verify_splits(cell_dir: Path) -> dict:
    splits = {}
    for name in ("train", "val", "test"):
        patients, patches = read_split(cell_dir / "runtime_splits" / f"{name}.csv")
        splits[name] = {"patients": patients, "patches": patches}

    train, val, test = (splits[n]["patients"] for n in ("train", "val", "test"))
    union = train | val | test
    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    return {
        "patient_counts": {n: len(splits[n]["patients"]) for n in splits},
        "patch_counts": {n: splits[n]["patches"] for n in splits},
        "total_patients": len(union),
        "total_patches": sum(splits[n]["patches"] for n in splits),
        "overlaps": overlaps,
        "disjoint": not any(overlaps.values()),
        "total_patients_ok": len(union) == TOTAL_PATIENTS,
        "test_patients": sorted(test),
    }


def audit_checkpoint(path: Path) -> dict:
    """Count non-finite floating-point values across every tensor."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    payload = state.get("model_state", state) if isinstance(state, dict) else state
    total = 0
    nonfinite = 0
    affected = []
    stack = [("", payload)]
    while stack:
        prefix, node = stack.pop()
        if torch.is_tensor(node):
            if node.is_floating_point():
                total += node.numel()
                bad = int((~torch.isfinite(node)).sum())
                if bad:
                    nonfinite += bad
                    affected.append(prefix)
        elif isinstance(node, dict):
            stack.extend((f"{prefix}.{k}" if prefix else str(k), v)
                         for k, v in node.items())
        elif isinstance(node, (list, tuple)):
            stack.extend((f"{prefix}[{i}]", v) for i, v in enumerate(node))
    return {
        "total_float_values": total,
        "nonfinite_values": nonfinite,
        "numerically_valid": nonfinite == 0,
        "affected_tensors": affected[:10],
    }


def stage_test_auprc(stage: dict, valid: bool) -> float | None:
    """Test AUPRC, or None when the stage failed to converge numerically.

    The evaluator writes zeros for a diverged checkpoint. Per the protocol
    amendment those zeros are missing values, never AUPRC = 0.
    """
    return float(stage["test_metrics"]["auprc"]) if valid else None


def verify_cell(cell_dir: Path, fold: int, width: int, skip_checkpoints: bool) -> dict:
    report_path = cell_dir / "histopath" / f"width_server_q{width}_fold{fold}_report.json"
    report = json.loads(report_path.read_text())
    provenance = report["provenance"]
    comparison = report["stage_comparison"]

    problems = []

    bundled_protocol = cell_dir / "histopath_vqc_width_server_protocol.md"
    protocol_sha = sha256(bundled_protocol)
    if protocol_sha != sha256(PROTOCOL):
        problems.append("bundled protocol differs from the local preregistration")

    for key, expected in DECLARED.items():
        if provenance[key] != expected:
            problems.append(f"{key} is not the declared value")
    gpu = provenance["gpus"][0]
    if gpu["name"] != DECLARED_GPU:
        problems.append(f"unexpected GPU {gpu['name']}")
    if gpu["driver_version"] != DECLARED_DRIVER:
        problems.append(f"unexpected driver {gpu['driver_version']}")
    if provenance["fold"] != fold or provenance["n_qubits"] != width:
        problems.append("provenance fold/width disagrees with the bundle name")
    if provenance["seed"] != BASE_SEED:
        problems.append("seed is not 42")
    if provenance["effective_seed"] != BASE_SEED + fold:
        problems.append("effective seed is not 42 + fold")

    splits = verify_splits(cell_dir)
    if not splits["disjoint"]:
        problems.append("patients overlap across train/val/test")
    if not splits["total_patients_ok"]:
        problems.append(f"{splits['total_patients']} patients, expected {TOTAL_PATIENTS}")

    checkpoints = cell_dir / "histopath" / "checkpoints"
    prefix = f"E3_histopath_fold{fold}_histopath_q{width}_seed{BASE_SEED}"
    numerics = {}
    for stage in STAGES:
        path = checkpoints / f"{prefix}_best_{stage}.pt"
        if skip_checkpoints:
            bundled = report["stage_numerics"][stage]
            numerics[stage] = {
                "numerically_valid": bundled["numerically_valid"],
                "nonfinite_values": bundled["nonfinite_values"],
                "source": "bundled_report",
            }
            continue
        audited = audit_checkpoint(path)
        bundled = report["stage_numerics"][stage]
        audited["source"] = "independent"
        audited["agrees_with_bundled_report"] = (
            audited["numerically_valid"] == bundled["numerically_valid"]
        )
        if not audited["agrees_with_bundled_report"]:
            problems.append(f"{stage} finiteness disagrees with the bundled report")
        numerics[stage] = audited

    selected = comparison["global_best_stage"]
    if not numerics[selected]["numerically_valid"]:
        problems.append(f"selected {selected} is non-finite; cell is invalid")

    selected_test = comparison["stages"][selected]["test_metrics"]
    if abs(selected_test["auprc"] - report["test_metrics"]["auprc"]) > 1e-12:
        problems.append("selected-stage AUPRC disagrees with the reported test metrics")

    cv_summary = json.loads((cell_dir / "histopath" / "cv_summary.json").read_text())
    cv_result = cv_summary["results"]["E3"][0]

    return {
        "fold": fold,
        "n_qubits": width,
        "directory": str(cell_dir.relative_to(ROOT)),
        "valid": not problems,
        "problems": problems,
        "source_commit": provenance["source_commit"],
        "image_id": provenance["image_id"],
        "captured_utc": provenance["captured_utc"],
        "gpu_uuid": gpu["uuid"],
        "protocol_sha256": protocol_sha,
        "splits": {k: v for k, v in splits.items() if k != "test_patients"},
        "test_patients": splits["test_patients"],
        "selected_stage": selected,
        "selection_metric": comparison["selection_metric"],
        "test_metrics": {k: selected_test[k] for k in METRIC_KEYS},
        "stage_validation_scores": {
            stage: comparison["stages"][stage]["best_val_selection_score"]
            for stage in STAGES
        },
        "stage_test_auprc": {
            stage: stage_test_auprc(
                comparison["stages"][stage], numerics[stage]["numerically_valid"]
            )
            for stage in STAGES
        },
        "stage_numerics": numerics,
        "stage_epochs_done": cv_result["train_metrics"]["stage_epochs_done"],
        "train_time_s": cv_result["train_metrics"]["train_time_s"],
        "circuit": {
            "n_qubits": cv_result["n_qubits"],
            "n_layers": cv_result["n_layers"],
            "entanglement": cv_result["entanglement"],
            "encoding": cv_result["encoding"],
            "data_reuploading": cv_result["data_reuploading"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and summarise the server-primary width matrix."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Directory holding the unpacked histopath_server_width_* bundles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/histopath/server_width_matrix.json",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Trust the bundled finiteness audit instead of reloading checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cells = []
    missing = []
    for fold in FOLDS:
        for width in WIDTHS:
            cell_dir = args.results_root / f"histopath_server_width_q{width}_fold{fold}"
            if not cell_dir.is_dir():
                missing.append(f"q{width} fold{fold}")
                continue
            cells.append(verify_cell(cell_dir, fold, width, args.skip_checkpoints))
            state = "ok " if cells[-1]["valid"] else "BAD"
            print(
                f"{state} fold {fold} q{width:<2} "
                f"selected {cells[-1]['selected_stage']:<7} "
                f"AUPRC {cells[-1]['test_metrics']['auprc']:.5f} "
                f"stageC {'finite' if cells[-1]['stage_numerics']['stage_c']['numerically_valid'] else 'FAILED'}"
            )
            for problem in cells[-1]["problems"]:
                print(f"      ! {problem}")

    commits = sorted({cell["source_commit"] for cell in cells})
    images = sorted({cell["image_id"] for cell in cells})
    gpus = sorted({cell["gpu_uuid"] for cell in cells})

    summary = {
        "protocol": "preregistration/histopath_vqc_width_server_protocol.md",
        "protocol_sha256": sha256(PROTOCOL),
        "environment": "server_primary_rtx_a4000",
        "complete": len(cells) == len(WIDTHS) * len(FOLDS),
        "missing_cells": missing,
        "all_cells_valid": all(cell["valid"] for cell in cells),
        "single_gpu": len(gpus) == 1,
        "single_image": len(images) == 1,
        "source_commits": commits,
        "single_source_commit": len(commits) == 1,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print()
    print(f"Cells verified: {len(cells)}/{len(WIDTHS) * len(FOLDS)}")
    print(f"All valid: {summary['all_cells_valid']}")
    print(f"Distinct GPUs: {gpus}")
    print(f"Distinct images: {images}")
    print(f"Distinct source commits: {commits}")
    print(f"Saved {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
