#!/usr/bin/env python3
"""Apply the preregistered Holm correction to Paper C's two co-primary tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-report", required=True, type=Path)
    parser.add_argument("--context-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = {
        "protocol_optimism": json.loads(args.protocol_report.read_text()),
        "context_gain": json.loads(args.context_report.read_text()),
    }
    ordered = sorted(
        reports,
        key=lambda name: reports[name]["two_sided_bootstrap_p"],
    )
    raw = {name: float(reports[name]["two_sided_bootstrap_p"]) for name in reports}
    adjusted = {
        ordered[0]: min(1.0, 2 * raw[ordered[0]]),
        ordered[1]: max(raw[ordered[1]], min(1.0, 2 * raw[ordered[0]])),
    }
    output = {
        "method": "Holm step-down, family size 2",
        "alpha": 0.05,
        "ordered_hypotheses": ordered,
        "results": {
            name: {
                "raw_p": raw[name],
                "holm_adjusted_p": adjusted[name],
                "reject_at_0.05": adjusted[name] <= 0.05,
                "difference": reports[name]["difference"],
                "ci_95_percentile": reports[name]["ci_95_percentile"],
            }
            for name in reports
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
