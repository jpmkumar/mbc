#!/usr/bin/env python3
"""Aggregate the end-to-end VQC width matrix under the pre-declared rules.

Two arms are reported separately and never pooled:

* ``server_primary`` -- the twelve RTX A4000 cells declared in
  ``preregistration/histopath_vqc_width_server_protocol.md``. q8 was rerun in the
  same immutable image, so ``q4 - q8`` and ``q12 - q8`` are paired within fold.
* ``kaggle_secondary`` -- the historical Kaggle cells declared in
  ``preregistration/histopath_vqc_width_protocol.md``, whose q8 reference is the
  already completed five-fold E3 run. Those runs share the Kaggle arm's patient
  partition, which is *not* the partition the server arm uses, so the two arms
  are commensurable only as independent replications of the same contrast.

Both arms use the Nadeau-Bengio corrected t procedure and the +-0.01 AUPRC
practical margin fixed by the width protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_vqc_stage_b_folds import (  # noqa: E402
    corrected_interval,
    equivalence_test,
)
from scripts.evaluate_vqc_stage_b_locked import (  # noqa: E402
    PRACTICAL_AUPRC_MARGIN,
)

REFERENCE_WIDTH = 8
COMPARISON_WIDTHS = (4, 12)
FOLDS = (1, 2, 3, 4)

# Kaggle arm: the width protocol names the completed five-fold E3 runs as the
# q8 reference, so those bundles supply the Kaggle q8 arm rather than a rerun.
KAGGLE_Q8_BUNDLES = {
    1: "results/histopath_kaggle_fold1_e3_v2/mbc_backup_20260719_1135_fold1_e3",
    2: "results/histopath_kaggle_fold2_e3_v2/mbc_backup_20260719_1641_fold2_e3",
    3: "results/histopath_kaggle_fold3_e3_v2/mbc_backup_20260719_2034_fold3_e3",
    4: "results/histopath_kaggle_fold4_e3_v2/mbc_backup_20260720_0414_fold4_e3",
}


def patients(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle)}


def load_server_arm(matrix_path: Path) -> dict:
    matrix = json.loads(matrix_path.read_text())
    if not matrix["complete"] or not matrix["all_cells_valid"]:
        raise ValueError("The server matrix is incomplete or contains invalid cells.")
    cells = {}
    for cell in matrix["cells"]:
        cells[(cell["fold"], cell["n_qubits"])] = {
            "auprc": cell["test_metrics"]["auprc"],
            "balanced_accuracy": cell["test_metrics"]["balanced_accuracy"],
            "f1": cell["test_metrics"]["f1"],
            "auc": cell["test_metrics"]["auc"],
            "recall": cell["test_metrics"]["recall"],
            "precision": cell["test_metrics"]["precision"],
            "selected_stage": cell["selected_stage"],
            "stage_test_auprc": cell["stage_test_auprc"],
            "stage_c_valid": cell["stage_numerics"]["stage_c"]["numerically_valid"],
            "stage_b_valid": cell["stage_numerics"]["stage_b"]["numerically_valid"],
            "train_time_s": cell["train_time_s"],
            "train_patches": cell["splits"]["patch_counts"]["train"],
            "test_patches": cell["splits"]["patch_counts"]["test"],
        }
    return cells


def load_kaggle_cell(directory: Path, fold: int, width: int) -> dict | None:
    report_path = directory / "histopath" / f"width_q{width}_fold{fold}_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    comparison = report["stage_comparison"]
    selected = comparison["global_best_stage"]
    stages = {}
    for stage in ("stage_a", "stage_b", "stage_c"):
        metrics = comparison["stages"][stage]["test_metrics"]
        # Zeroed metrics mark a diverged checkpoint; the protocol treats those as
        # missing, never as AUPRC = 0.
        stages[stage] = None if metrics["auprc"] == 0.0 else metrics["auprc"]
    selected_metrics = comparison["stages"][selected]["test_metrics"]
    train = patients(directory / "runtime_splits" / "train.csv")
    return {
        "auprc": selected_metrics["auprc"],
        "balanced_accuracy": selected_metrics["balanced_accuracy"],
        "f1": selected_metrics["f1"],
        "auc": selected_metrics["auc"],
        "recall": selected_metrics["recall"],
        "precision": selected_metrics["precision"],
        "selected_stage": selected,
        "stage_test_auprc": stages,
        "stage_c_valid": stages["stage_c"] is not None,
        "stage_b_valid": stages["stage_b"] is not None,
        "train_patches": sum(1 for _ in (directory / "runtime_splits" / "train.csv").open()) - 1,
        "test_patches": int(selected_metrics["n_samples"]),
        "test_patients": sorted(patients(directory / "runtime_splits" / "test.csv")),
        "train_patients": len(train),
    }


def load_kaggle_q8(directory: Path, fold: int) -> dict:
    summary = json.loads((directory / "histopath" / "cv_summary.json").read_text())
    result = summary["results"]["E3"][0]
    metrics = result["train_metrics"]
    if result["n_qubits"] != REFERENCE_WIDTH:
        raise ValueError(f"{directory} is not an 8-qubit reference run.")
    return {
        "auprc": metrics["auprc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "selected_stage": metrics["best_stage"],
        "stage_test_auprc": None,
        "stage_c_valid": None,
        "stage_b_valid": None,
        "test_patches": int(metrics["n_samples"]),
        "test_patients": sorted(patients(directory / "splits" / "runtime" / f"fold_{fold}" / "test.csv")),
        "train_patches": None,
    }


def load_kaggle_arm() -> dict:
    cells = {}
    for fold, relative in KAGGLE_Q8_BUNDLES.items():
        cells[(fold, REFERENCE_WIDTH)] = load_kaggle_q8(ROOT / relative, fold)
    for fold in FOLDS:
        for width in COMPARISON_WIDTHS:
            directory = ROOT / f"results/histopath_width_q{width}_fold{fold}"
            if not directory.is_dir():
                continue
            cell = load_kaggle_cell(directory, fold, width)
            if cell is not None:
                cells[(fold, width)] = cell
    return cells


def paired_contrast(cells: dict, width: int, ratio: float | None) -> dict:
    """Per-fold ``width - q8`` AUPRC differences and the corrected aggregate."""
    per_fold = []
    for fold in FOLDS:
        treatment = cells.get((fold, width))
        reference = cells.get((fold, REFERENCE_WIDTH))
        if treatment is None or reference is None:
            continue
        entry = {
            "fold": fold,
            "treatment_auprc": treatment["auprc"],
            "reference_auprc": reference["auprc"],
            "delta_auprc": treatment["auprc"] - reference["auprc"],
            "treatment_selected_stage": treatment["selected_stage"],
            "reference_selected_stage": reference["selected_stage"],
        }
        if "test_patients" in treatment and "test_patients" in reference:
            entry["same_test_patients"] = (
                treatment["test_patients"] == reference["test_patients"]
            )
        else:
            entry["same_test_patients"] = True
        per_fold.append(entry)

    deltas = [entry["delta_auprc"] for entry in per_fold]
    contrast = {
        "width": width,
        "reference_width": REFERENCE_WIDTH,
        "folds": [entry["fold"] for entry in per_fold],
        "per_fold": per_fold,
        "complete": len(per_fold) == len(FOLDS),
        "paired_on_identical_test_patients": all(
            entry["same_test_patients"] for entry in per_fold
        ),
        "mean_delta_auprc": statistics.mean(deltas) if deltas else None,
    }
    if len(deltas) < 2 or ratio is None:
        contrast["aggregate"] = None
        return contrast

    interval_90 = corrected_interval(deltas, ratio, 0.90)
    interval_95 = corrected_interval(deltas, ratio, 0.95)
    tost = equivalence_test(deltas, ratio)

    inside_margin = (
        interval_90["lower"] > -PRACTICAL_AUPRC_MARGIN
        and interval_90["upper"] < PRACTICAL_AUPRC_MARGIN
    )
    excludes_zero = interval_95["lower"] > 0.0 or interval_95["upper"] < 0.0
    exceeds_margin = abs(interval_95["mean"]) > PRACTICAL_AUPRC_MARGIN
    if inside_margin and tost["equivalent"]:
        decision = "practical_equivalence"
    elif excludes_zero and exceeds_margin:
        decision = "difference"
    else:
        decision = "inconclusive"

    # Two-sided difference-from-zero p-value on the corrected standard error,
    # supplied so the protocol's Holm correction across the two widths is shown.
    statistic = interval_95["mean"] / interval_95["corrected_standard_error"]
    p_value = float(2.0 * stats.t.sf(abs(statistic), len(deltas) - 1))

    contrast["aggregate"] = {
        "corrected_interval_90": interval_90,
        "corrected_interval_95": interval_95,
        "equivalence_test": tost,
        "difference_p_value_uncorrected": p_value,
        "decision": decision,
    }
    return contrast


def holm(p_values: dict[int, float]) -> dict[int, float]:
    """Holm step-down adjustment across the two planned width comparisons."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for index, (width, raw) in enumerate(ordered):
        value = (len(ordered) - index) * raw
        running = max(running, min(value, 1.0))
        adjusted[width] = running
    return adjusted


def quantum_stage_evidence(cells: dict) -> dict:
    """The protocol admits a quantum-stage benefit only under these conditions."""
    selected_quantum = []
    increments = []
    for (fold, width), cell in sorted(cells.items()):
        if cell["selected_stage"] in ("stage_b", "stage_c"):
            selected_quantum.append({"fold": fold, "width": width,
                                     "stage": cell["selected_stage"]})
        stages = cell.get("stage_test_auprc")
        if not stages or stages.get("stage_a") is None or stages.get("stage_b") is None:
            continue
        increments.append({
            "fold": fold,
            "width": width,
            "stage_b_minus_stage_a_auprc": stages["stage_b"] - stages["stage_a"],
        })
    positive = [entry for entry in increments
                if entry["stage_b_minus_stage_a_auprc"] > 0.0]
    return {
        "cells_selecting_quantum_stage": selected_quantum,
        "n_cells_selecting_quantum_stage": len(selected_quantum),
        "within_width_stage_b_minus_a": increments,
        "n_with_stage_attribution": len(increments),
        "n_positive_increments": len(positive),
        "mean_stage_b_minus_a": (
            statistics.mean(e["stage_b_minus_stage_a_auprc"] for e in increments)
            if increments else None
        ),
        "max_stage_b_minus_a": (
            max(e["stage_b_minus_stage_a_auprc"] for e in increments)
            if increments else None
        ),
    }


def numerical_failures(cells: dict) -> dict:
    by_width = {}
    for width in (4, 8, 12):
        entries = [cell for (fold, w), cell in cells.items()
                   if w == width and cell["stage_c_valid"] is not None]
        if not entries:
            continue
        by_width[f"q{width}"] = {
            "cells": len(entries),
            "stage_c_failures": sum(1 for cell in entries if not cell["stage_c_valid"]),
            "stage_b_failures": sum(1 for cell in entries if not cell["stage_b_valid"]),
        }
    return by_width


def analyse_arm(name: str, cells: dict, ratio: float | None, notes: str) -> dict:
    contrasts = {width: paired_contrast(cells, width, ratio)
                 for width in COMPARISON_WIDTHS}
    p_values = {
        width: contrast["aggregate"]["difference_p_value_uncorrected"]
        for width, contrast in contrasts.items()
        if contrast["aggregate"] is not None
    }
    adjusted = holm(p_values) if len(p_values) == len(COMPARISON_WIDTHS) else {}
    for width, value in adjusted.items():
        contrasts[width]["aggregate"]["difference_p_value_holm"] = value

    return {
        "arm": name,
        "notes": notes,
        "test_train_ratio": ratio,
        "practical_auprc_margin": PRACTICAL_AUPRC_MARGIN,
        "cells": {f"fold{fold}_q{width}": cell
                  for (fold, width), cell in sorted(cells.items())},
        "selected_auprc_by_width": {
            f"q{width}": {
                f"fold{fold}": cells[(fold, width)]["auprc"]
                for fold in FOLDS if (fold, width) in cells
            }
            for width in (4, 8, 12)
        },
        "contrasts": {f"q{width}_minus_q8": contrast
                      for width, contrast in contrasts.items()},
        "quantum_stage_evidence": quantum_stage_evidence(cells),
        "numerical_failures_by_width": numerical_failures(cells),
    }


def environment_robustness(server: dict, kaggle: dict) -> dict:
    """Compare the two arms as independent replications of the same contrast."""
    comparison = {}
    for width in COMPARISON_WIDTHS:
        key = f"q{width}_minus_q8"
        server_contrast = server["contrasts"][key]
        kaggle_contrast = kaggle["contrasts"][key]
        server_mean = server_contrast["mean_delta_auprc"]
        kaggle_mean = kaggle_contrast["mean_delta_auprc"]
        comparison[key] = {
            "server_mean_delta": server_mean,
            "kaggle_mean_delta": kaggle_mean,
            "server_folds": server_contrast["folds"],
            "kaggle_folds": kaggle_contrast["folds"],
            "same_sign": (server_mean > 0) == (kaggle_mean > 0)
            if None not in (server_mean, kaggle_mean) else None,
            "both_within_margin": (
                abs(server_mean) < PRACTICAL_AUPRC_MARGIN
                and abs(kaggle_mean) < PRACTICAL_AUPRC_MARGIN
            ) if None not in (server_mean, kaggle_mean) else None,
            "absolute_gap_between_arms": abs(server_mean - kaggle_mean)
            if None not in (server_mean, kaggle_mean) else None,
        }

    server_evidence = server["quantum_stage_evidence"]
    kaggle_evidence = kaggle["quantum_stage_evidence"]
    pooled_increments = (
        server_evidence["within_width_stage_b_minus_a"]
        + kaggle_evidence["within_width_stage_b_minus_a"]
    )
    return {
        "statement": (
            "The two arms use different hardware and different patient-to-fold "
            "assignments, so only the direction and magnitude of the within-arm "
            "width contrast is comparable; absolute AUPRC levels are not."
        ),
        "width_contrasts": comparison,
        "pooled_stage_b_minus_a": {
            "runs": len(pooled_increments),
            "positive": sum(1 for e in pooled_increments
                            if e["stage_b_minus_stage_a_auprc"] > 0.0),
            "mean": statistics.mean(
                e["stage_b_minus_stage_a_auprc"] for e in pooled_increments
            ),
            "max": max(e["stage_b_minus_stage_a_auprc"] for e in pooled_increments),
            "min": min(e["stage_b_minus_stage_a_auprc"] for e in pooled_increments),
        },
        "quantum_stage_selected": {
            "server": server_evidence["n_cells_selecting_quantum_stage"],
            "kaggle": kaggle_evidence["n_cells_selecting_quantum_stage"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate the server-primary and Kaggle-secondary width matrices."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "results/histopath/server_width_matrix.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/histopath/width_matrix_analysis.json",
    )
    return parser.parse_args()


def describe(arm: dict) -> None:
    print(f"--- {arm['arm']} ---")
    for key, contrast in arm["contrasts"].items():
        deltas = ", ".join(
            f"F{entry['fold']} {entry['delta_auprc']:+.5f}"
            for entry in contrast["per_fold"]
        )
        print(f"{key}: {deltas}")
        if contrast["mean_delta_auprc"] is None:
            print("   no folds available")
            continue
        print(f"   mean {contrast['mean_delta_auprc']:+.5f}", end="")
        aggregate = contrast["aggregate"]
        if aggregate is None:
            print("  (too few folds for a corrected interval)")
            continue
        interval = aggregate["corrected_interval_90"]
        print(
            f"  corrected 90% [{interval['lower']:+.5f}, {interval['upper']:+.5f}]"
            f"  TOST p={aggregate['equivalence_test']['p_value']:.4f}"
            f"  -> {aggregate['decision']}"
        )
    evidence = arm["quantum_stage_evidence"]
    print(
        f"quantum stage selected in {evidence['n_cells_selecting_quantum_stage']} cell(s); "
        f"Stage B - Stage A positive in {evidence['n_positive_increments']}"
        f"/{evidence['n_with_stage_attribution']} runs"
    )
    print(f"numerical failures: {arm['numerical_failures_by_width']}")
    print()


def main():
    args = parse_args()

    server_cells = load_server_arm(args.matrix)
    server_ratio = statistics.mean(
        cell["test_patches"] / cell["train_patches"] for cell in server_cells.values()
    )
    server = analyse_arm(
        "server_primary",
        server_cells,
        server_ratio,
        "Twelve cells on one RTX A4000, one image, one source commit, and the "
        "committed split manifest. q8 was rerun inside the same image, so both "
        "contrasts are paired within fold on identical test patients.",
    )

    kaggle_cells = load_kaggle_arm()
    kaggle_ratio = statistics.mean(
        cell["test_patches"] / cell["train_patches"]
        for cell in kaggle_cells.values()
        if cell.get("train_patches")
    )
    kaggle = analyse_arm(
        "kaggle_secondary",
        kaggle_cells,
        kaggle_ratio,
        "Historical Kaggle T4 x2 cells. The q8 reference is the completed "
        "five-fold E3 run named by the original width protocol; it shares the "
        "Kaggle patient partition, which differs from the committed manifest "
        "used by the server arm.",
    )

    summary = {
        "primary_metric": "held_out_test_auprc_of_validation_selected_checkpoint",
        "unit_of_analysis": "fold",
        "practical_auprc_margin": PRACTICAL_AUPRC_MARGIN,
        "arms_are_never_pooled": True,
        "server_primary": server,
        "kaggle_secondary": kaggle,
        "environment_robustness": environment_robustness(server, kaggle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    describe(server)
    describe(kaggle)
    robustness = summary["environment_robustness"]
    print("--- environment robustness ---")
    for key, entry in robustness["width_contrasts"].items():
        print(
            f"{key}: server {entry['server_mean_delta']:+.5f} "
            f"(folds {entry['server_folds']}) vs Kaggle "
            f"{entry['kaggle_mean_delta']:+.5f} (folds {entry['kaggle_folds']}), "
            f"same sign {entry['same_sign']}"
        )
    pooled = robustness["pooled_stage_b_minus_a"]
    print(
        f"Stage B - Stage A across both arms: {pooled['positive']}/{pooled['runs']} "
        f"positive, mean {pooled['mean']:+.5f}, "
        f"range [{pooled['min']:+.5f}, {pooled['max']:+.5f}]"
    )
    print(f"Saved {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
