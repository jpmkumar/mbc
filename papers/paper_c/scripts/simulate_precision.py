#!/usr/bin/env python3
"""Pre-outcome simulation using only IDC case sizes and label prevalences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-stats",
        type=Path,
        default=Path("data/splits/histopath_kaggle/patient_stats.csv"),
    )
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=72026)
    parser.add_argument("--max-per-class-per-case", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.case_stats.open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    if len(cases) != 279:
        raise SystemExit(f"Expected 279 case rows, found {len(cases)}")
    labels = []
    case_ids = []
    for row in cases:
        for label, column in ((0, "n0"), (1, "n1")):
            count = min(int(row[column]), args.max_per_class_per_case)
            labels.extend([label] * count)
            case_ids.extend([row["patient_id"]] * count)
    y = np.asarray(labels)
    case_ids_array = np.asarray(case_ids)
    unique, counts = np.unique(case_ids_array, return_counts=True)
    count_by_case = dict(zip(unique, counts))
    weights = np.asarray([1 / count_by_case[case] for case in case_ids_array])

    rng = np.random.default_rng(args.seed)
    results = []
    for signal_shift in (0.0, 0.05, 0.10):
        for seed_logit_sd in (0.02, 0.05, 0.10):
            for n_seeds in (1, 3, 5):
                differences = []
                for _ in range(args.trials):
                    case_effect = {
                        case: rng.normal(0, 0.5) for case in unique
                    }
                    shared = np.asarray([case_effect[case] for case in case_ids_array])
                    patch_noise = rng.normal(0, 1, len(y))
                    base_logit = (2 * y - 1) * 1.0 + shared + patch_noise
                    optimization_noise = rng.normal(
                        0, seed_logit_sd / np.sqrt(n_seeds), len(y)
                    )
                    alternate_logit = (
                        (2 * y - 1) * (1.0 + signal_shift)
                        + shared
                        + patch_noise
                        + optimization_noise
                    )
                    base_ap = average_precision_score(
                        y, base_logit, sample_weight=weights
                    )
                    alternate_ap = average_precision_score(
                        y, alternate_logit, sample_weight=weights
                    )
                    differences.append(alternate_ap - base_ap)
                values = np.asarray(differences)
                results.append({
                    "signal_shift": signal_shift,
                    "seed_logit_sd": seed_logit_sd,
                    "n_seeds": n_seeds,
                    "mean_ap_difference": float(values.mean()),
                    "simulation_sd": float(values.std(ddof=1)),
                    "central_95_interval": [
                        float(np.quantile(values, 0.025)),
                        float(np.quantile(values, 0.975)),
                    ],
                })
    report = {
        "analysis": "pre-outcome simulation; no model predictions used",
        "case_identifiers": len(cases),
        "simulated_rows": len(y),
        "cap_per_class_per_case": args.max_per_class_per_case,
        "trials_per_cell": args.trials,
        "seed": args.seed,
        "assumptions": {
            "case_random_effect_sd": 0.5,
            "patch_noise_sd": 1.0,
            "baseline_label_signal": 1.0,
            "paired_shared_case_and_patch_noise": True,
        },
        "cells": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
