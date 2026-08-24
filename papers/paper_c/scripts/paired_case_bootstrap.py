#!/usr/bin/env python3
"""Paired whole-case bootstrap for Paper C OOF prediction bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_weight_vector(cases: np.ndarray, multiplicities: dict[str, int]) -> np.ndarray:
    unique, counts = np.unique(cases, return_counts=True)
    rows_per_case = {case: count for case, count in zip(unique, counts)}
    return np.asarray(
        [multiplicities.get(case, 0) / rows_per_case[case] for case in cases],
        dtype=np.float64,
    )


def metric(name: str, y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    if name == "case_balanced_auprc":
        return float(average_precision_score(y, probability, sample_weight=weight))
    if name == "case_balanced_auroc":
        return float(roc_auc_score(y, probability, sample_weight=weight))
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument(
        "--metric",
        choices=("case_balanced_auprc", "case_balanced_auroc"),
        default="case_balanced_auprc",
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=52026)
    parser.add_argument(
        "--strata-column",
        help="Optional case-level bootstrap stratum, e.g. site for BCSS.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)
    required = {"filepath", "case_id", "label", "probability"}
    if args.strata_column:
        required.add(args.strata_column)
    for name, frame in (("left", left), ("right", right)):
        if not required <= set(frame):
            raise SystemExit(f"{name} bundle lacks {sorted(required - set(frame))}")
        if frame["filepath"].duplicated().any():
            raise SystemExit(f"{name} bundle has duplicate filepaths.")
    paired = left[list(required)].merge(
        right[list(required)],
        on="filepath",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise SystemExit("Prediction bundles do not have identical filepath populations.")
    if not (
        (paired["case_id_left"].astype(str) == paired["case_id_right"].astype(str)).all()
        and (paired["label_left"] == paired["label_right"]).all()
    ):
        raise SystemExit("Paired prediction metadata disagree.")

    cases = paired["case_id_left"].astype(str).to_numpy()
    y = paired["label_left"].to_numpy()
    left_probability = paired["probability_left"].to_numpy()
    right_probability = paired["probability_right"].to_numpy()
    unique_cases = np.unique(cases)
    strata_cases: list[np.ndarray]
    if args.strata_column:
        strata = paired[f"{args.strata_column}_left"].astype(str).to_numpy()
        if not (
            paired[f"{args.strata_column}_left"].astype(str)
            == paired[f"{args.strata_column}_right"].astype(str)
        ).all():
            raise SystemExit("Paired prediction bootstrap strata disagree.")
        case_to_strata: dict[str, str] = {}
        for case, stratum in zip(cases, strata):
            previous = case_to_strata.setdefault(case, stratum)
            if previous != stratum:
                raise SystemExit(f"Case {case} appears in multiple bootstrap strata.")
        strata_cases = [
            np.asarray(sorted(
                case for case, value in case_to_strata.items() if value == stratum
            ))
            for stratum in sorted(set(case_to_strata.values()))
        ]
    else:
        strata_cases = [unique_cases]
    base_multiplicity = {case: 1 for case in unique_cases}
    base_weight = case_weight_vector(cases, base_multiplicity)
    left_estimate = metric(args.metric, y, left_probability, base_weight)
    right_estimate = metric(args.metric, y, right_probability, base_weight)
    estimate = left_estimate - right_estimate

    rng = np.random.default_rng(args.seed)
    differences = np.empty(args.replicates, dtype=np.float64)
    for replicate in range(args.replicates):
        sampled = np.concatenate([
            rng.choice(stratum_case_ids, size=len(stratum_case_ids), replace=True)
            for stratum_case_ids in strata_cases
        ])
        selected, counts = np.unique(sampled, return_counts=True)
        multiplicity = dict(zip(selected, counts))
        weight = case_weight_vector(cases, multiplicity)
        differences[replicate] = (
            metric(args.metric, y, left_probability, weight)
            - metric(args.metric, y, right_probability, weight)
        )

    null_deviations = differences - estimate
    extreme = int(np.sum(np.abs(null_deviations) >= abs(estimate)))
    centred_bootstrap_p = (extreme + 1) / (args.replicates + 1)
    report = {
        "metric": args.metric,
        "contrast": f"{args.left_label} - {args.right_label}",
        "left_estimate": left_estimate,
        "right_estimate": right_estimate,
        "difference": estimate,
        "ci_95_percentile": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "two_sided_bootstrap_p": centred_bootstrap_p,
        "p_value_method": (
            "two-sided centred case bootstrap with +1 finite-replicate correction"
        ),
        "replicates": args.replicates,
        "seed": args.seed,
        "case_identifiers": len(unique_cases),
        "strata_column": args.strata_column,
        "strata_count": len(strata_cases),
        "rows": len(paired),
        "left_sha256": sha256(args.left),
        "right_sha256": sha256(args.right),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
