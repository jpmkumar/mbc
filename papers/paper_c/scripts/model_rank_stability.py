#!/usr/bin/env python3
"""Summarize model-rank changes between paired Paper C conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import kendalltau


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("ENCODER", "LEFT_SUMMARY", "RIGHT_SUMMARY"),
        required=True,
    )
    parser.add_argument("--metric", default="case_balanced_auprc")
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for encoder, left_path, right_path in args.pair:
        left = json.loads(Path(left_path).read_text())["metrics"][args.metric]
        right = json.loads(Path(right_path).read_text())["metrics"][args.metric]
        rows.append({
            "encoder": encoder,
            "left": float(left),
            "right": float(right),
            "difference": float(left - right),
        })
    if len({row["encoder"] for row in rows}) != len(rows):
        raise SystemExit("Encoder names must be unique.")
    left_order = sorted(rows, key=lambda row: (-row["left"], row["encoder"]))
    right_order = sorted(rows, key=lambda row: (-row["right"], row["encoder"]))
    left_rank = {row["encoder"]: rank + 1 for rank, row in enumerate(left_order)}
    right_rank = {row["encoder"]: rank + 1 for rank, row in enumerate(right_order)}
    for row in rows:
        row["left_rank"] = left_rank[row["encoder"]]
        row["right_rank"] = right_rank[row["encoder"]]
        row["rank_change_right_minus_left"] = (
            row["right_rank"] - row["left_rank"]
        )
    encoders = sorted(left_rank)
    tau = kendalltau(
        [left_rank[encoder] for encoder in encoders],
        [right_rank[encoder] for encoder in encoders],
    )
    report = {
        "metric": args.metric,
        "left_label": args.left_label,
        "right_label": args.right_label,
        "kendall_tau": float(tau.statistic),
        "kendall_p_descriptive": float(tau.pvalue),
        "models": sorted(rows, key=lambda row: row["left_rank"]),
        "inference_note": (
            "Descriptive rank analysis; case-bootstrap rank uncertainty must be "
            "computed from paired predictions for manuscript reporting."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
