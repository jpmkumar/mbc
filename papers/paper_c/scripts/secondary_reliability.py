#!/usr/bin/env python3
"""Case-balanced calibration and empirical risk–coverage summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss

from run_idc_probe_cv import case_weights, sha256


def calibration_model(
    y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    logits = logit(np.clip(probabilities, 1e-7, 1 - 1e-7))

    def objective(parameters: np.ndarray) -> float:
        fitted = expit(parameters[0] + parameters[1] * logits)
        return float(log_loss(y, fitted, sample_weight=weights, labels=[0, 1]))

    result = minimize(objective, x0=np.asarray([0.0, 1.0]), method="BFGS")
    if not result.success:
        raise RuntimeError(f"Calibration fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, dtype={"case_id": str})
    required = {"filepath", "case_id", "label", "probability", "threshold"}
    if not required <= set(frame):
        raise SystemExit(f"Missing columns: {sorted(required - set(frame))}")
    if frame["filepath"].duplicated().any():
        raise SystemExit("Prediction filepath is not unique.")
    y = frame["label"].to_numpy()
    probabilities = frame["probability"].to_numpy()
    weights = case_weights(frame["case_id"].to_numpy())
    intercept, slope = calibration_model(y, probabilities, weights)

    ranked = frame.assign(_weight=weights).sort_values("probability")
    ranked["_bin"] = pd.qcut(
        ranked["probability"],
        q=args.bins,
        labels=False,
        duplicates="drop",
    )
    reliability_rows = []
    for bin_id, group in ranked.groupby("_bin", observed=True):
        w = group["_weight"].to_numpy()
        reliability_rows.append({
            "bin": int(bin_id),
            "rows": len(group),
            "weighted_mean_probability": float(np.average(group["probability"], weights=w)),
            "weighted_observed_fraction": float(np.average(group["label"], weights=w)),
        })

    confidence = np.abs(probabilities - 0.5)
    risk_rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        retained = np.argsort(-confidence)[: max(1, round(coverage * len(frame)))]
        errors = (probabilities[retained] >= frame["threshold"].to_numpy()[retained]) != y[retained]
        risk_rows.append({
            "coverage": float(coverage),
            "rows": len(retained),
            "case_balanced_error_risk": float(np.average(
                errors,
                weights=case_weights(frame["case_id"].to_numpy()[retained]),
            )),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = args.output_dir / "reliability.csv"
    risk_path = args.output_dir / "risk_coverage.csv"
    summary_path = args.output_dir / "reliability_summary.json"
    if any(path.exists() for path in (reliability_path, risk_path, summary_path)):
        raise SystemExit(f"Reliability output already exists: {args.output_dir}")
    pd.DataFrame(reliability_rows).to_csv(reliability_path, index=False)
    pd.DataFrame(risk_rows).to_csv(risk_path, index=False)
    summary = {
        "rows": len(frame),
        "case_identifiers": int(frame["case_id"].nunique()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "case_balanced_brier": float(
            brier_score_loss(y, probabilities, sample_weight=weights)
        ),
        "case_balanced_log_loss": float(
            log_loss(y, probabilities, sample_weight=weights)
        ),
        "predictions_sha256": sha256(args.predictions),
        "reliability_sha256": sha256(reliability_path),
        "risk_coverage_sha256": sha256(risk_path),
        "coverage_claim": (
            "descriptive empirical selective risk only; no conformal guarantee"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
