#!/usr/bin/env python3
"""Combine per-fold locked Stage-B results into one cross-fold decision.

Implements the decision rules fixed in
``preregistration/stage_b_protocol_v2.md``: the fold is the unit of analysis,
the interval uses the Nadeau-Bengio correction for overlapping cross-validation
training sets, and equivalence is claimed only when the corrected 90% interval
falls inside the practical margin.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_vqc_stage_b_locked import (  # noqa: E402
    PRACTICAL_AUPRC_MARGIN,
)

EQUIVALENCE_ALPHA = 0.05


def corrected_interval(
    deltas: list[float],
    test_train_ratio: float,
    confidence: float,
) -> dict:
    """Nadeau-Bengio corrected t interval for the mean fold-level gap.

    Ordinary k-fold t intervals are anticonservative because the folds share
    training data. The correction inflates the variance by
    ``1/k + n_test/n_train``.
    """
    folds = len(deltas)
    if folds < 2:
        raise ValueError("At least two folds are required.")
    mean = statistics.mean(deltas)
    variance = statistics.variance(deltas)
    naive_standard_error = (variance / folds) ** 0.5
    correction = 1.0 / folds + test_train_ratio
    corrected_standard_error = (correction * variance) ** 0.5
    quantile = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, folds - 1)
    return {
        "confidence": confidence,
        "folds": folds,
        "mean": mean,
        "sample_standard_deviation": variance**0.5,
        "test_train_ratio": test_train_ratio,
        "variance_inflation": correction * folds,
        "naive_standard_error": naive_standard_error,
        "corrected_standard_error": corrected_standard_error,
        "naive_lower": mean - quantile * naive_standard_error,
        "naive_upper": mean + quantile * naive_standard_error,
        "lower": mean - quantile * corrected_standard_error,
        "upper": mean + quantile * corrected_standard_error,
    }


def equivalence_test(deltas: list[float], test_train_ratio: float) -> dict:
    """Two one-sided tests against the practical margin, corrected variance."""
    folds = len(deltas)
    mean = statistics.mean(deltas)
    variance = statistics.variance(deltas)
    standard_error = ((1.0 / folds + test_train_ratio) * variance) ** 0.5
    degrees = folds - 1
    upper_statistic = (mean - PRACTICAL_AUPRC_MARGIN) / standard_error
    lower_statistic = (mean + PRACTICAL_AUPRC_MARGIN) / standard_error
    p_below_margin = float(stats.t.cdf(upper_statistic, degrees))
    p_above_negative_margin = float(stats.t.sf(lower_statistic, degrees))
    p_value = max(p_below_margin, p_above_negative_margin)
    return {
        "margin": PRACTICAL_AUPRC_MARGIN,
        "alpha": EQUIVALENCE_ALPHA,
        "p_gap_below_positive_margin": p_below_margin,
        "p_gap_above_negative_margin": p_above_negative_margin,
        "p_value": p_value,
        "equivalent": p_value < EQUIVALENCE_ALPHA,
    }


def decide(interval_90: dict, interval_95: dict, tost: dict) -> tuple[str, str]:
    """Apply the pre-declared decision rules and return (decision, wording)."""
    inside_margin = (
        interval_90["lower"] > -PRACTICAL_AUPRC_MARGIN
        and interval_90["upper"] < PRACTICAL_AUPRC_MARGIN
    )
    excludes_zero = interval_95["lower"] > 0.0 or interval_95["upper"] < 0.0
    exceeds_margin = abs(interval_95["mean"]) > PRACTICAL_AUPRC_MARGIN

    if inside_margin and tost["equivalent"]:
        return (
            "practical_equivalence",
            "The corrected 90% interval lies entirely inside the "
            f"±{PRACTICAL_AUPRC_MARGIN} AUPRC margin, so the matched VQC and "
            "MLP heads are practically equivalent under the pre-declared "
            "rule. This is an equivalence claim within the stated margin, not "
            "a claim that the two heads are identical.",
        )
    if excludes_zero and exceeds_margin:
        favored = "VQC" if interval_95["mean"] > 0 else "MLP"
        return (
            "difference",
            f"The corrected 95% interval excludes zero and the mean gap "
            f"exceeds the margin, so the {favored} head is better by a "
            "clinically relevant amount.",
        )
    return (
        "inconclusive",
        "The corrected interval neither clears the equivalence margin nor "
        "establishes a relevant difference. Report this as inconclusive at "
        "the observed interval width, not as a null result.",
    )


def pooled_instability(folds: list[dict]) -> dict:
    """Count how often each head produced a badly converged seed."""
    report = {}
    for model in ("mlp", "vqc"):
        flagged = [fold["fold"] for fold in folds if model in fold["unstable_models"]]
        spreads = [fold["seed_stability"][model]["spread"] for fold in folds]
        report[model] = {
            "folds_flagged_unstable": flagged,
            "max_seed_spread": max(spreads),
            "mean_seed_spread": statistics.mean(spreads),
        }
    return report


def load_folds(paths: list[Path]) -> list[dict]:
    folds = []
    for path in paths:
        summary = json.loads(path.read_text())
        if not summary.get("held_out_test_evaluated"):
            raise ValueError(f"{path} is not a held-out test summary.")
        folds.append(
            {
                "fold": summary["fold"],
                "path": str(path),
                "seeds": summary["by_model"]["mlp"]["seeds"],
                "mean_delta": summary["mean_vqc_minus_mlp_test_auprc"],
                "median_delta": summary["median_vqc_minus_mlp_test_auprc"],
                "unstable_models": summary["unstable_models"],
                "seed_stability": summary["seed_stability"],
                "train_samples": summary["source"].get("train_samples"),
                "test_samples": summary["source"]["test_samples"],
            }
        )
    seen = [fold["fold"] for fold in folds]
    if len(set(seen)) != len(seen):
        raise ValueError(f"Duplicate folds in input: {seen}")
    return sorted(folds, key=lambda fold: fold["fold"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-fold locked Stage-B summaries into the "
            "pre-declared cross-fold decision."
        )
    )
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-train-ratio",
        type=float,
        default=None,
        help="Override the ratio used by the Nadeau-Bengio correction.",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=5,
        help="Warn when fewer folds than the protocol specifies are supplied.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folds = load_folds(args.summaries)
    deltas = [fold["mean_delta"] for fold in folds]
    medians = [fold["median_delta"] for fold in folds]

    if args.test_train_ratio is not None:
        ratio = args.test_train_ratio
    else:
        ratios = [
            fold["test_samples"] / fold["train_samples"]
            for fold in folds
            if fold["train_samples"]
        ]
        if not ratios:
            raise ValueError(
                "No train sample counts found; pass --test-train-ratio."
            )
        ratio = statistics.mean(ratios)

    interval_90 = corrected_interval(deltas, ratio, 0.90)
    interval_95 = corrected_interval(deltas, ratio, 0.95)
    tost = equivalence_test(deltas, ratio)
    decision, wording = decide(interval_90, interval_95, tost)

    summary = {
        "protocol": "stage_b_protocol_v2_cross_fold",
        "primary_statistic": "mean_over_seeds_vqc_minus_mlp_test_auprc",
        "unit_of_analysis": "fold",
        "folds": folds,
        "complete": len(folds) >= args.expected_folds,
        "corrected_interval_90": interval_90,
        "corrected_interval_95": interval_95,
        "equivalence_test": tost,
        "median_based_interval_90": corrected_interval(medians, ratio, 0.90),
        "instability": pooled_instability(folds),
        "decision": decision,
        "statement": wording,
    }
    if not summary["complete"]:
        summary["statement"] = (
            f"PROVISIONAL, {len(folds)} of {args.expected_folds} folds. "
            + wording
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"Folds: {[fold['fold'] for fold in folds]}")
    print(f"Mean cross-fold gap: {interval_90['mean']:+.6f}")
    print(
        "Corrected 90% interval: "
        f"[{interval_90['lower']:+.6f}, {interval_90['upper']:+.6f}] "
        f"(naive [{interval_90['naive_lower']:+.6f}, "
        f"{interval_90['naive_upper']:+.6f}])"
    )
    print(f"TOST p-value: {tost['p_value']:.5f}")
    print(f"Decision: {decision}")
    print(summary["statement"])
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
