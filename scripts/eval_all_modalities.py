#!/usr/bin/env python3
"""Evaluate a checkpoint on mammo, ultrasound, and thermography."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("mammo", "ultrasound", "thermo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", default="configs/mammo_enhanced.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--eval-stage",
        default="a",
        choices=["a", "b"],
        help="a=classical head, b=VQC head",
    )
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    results = {}
    for modality in MODALITIES:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/analyze_results.py"),
            args.checkpoint,
            "--config",
            args.config,
            "--split",
            args.split,
            "--modality",
            modality,
            "--eval-stage",
            args.eval_stage,
        ]
        if args.threshold_sweep:
            cmd.append("--threshold-sweep")
        if args.threshold is not None:
            cmd.extend(["--threshold", str(args.threshold)])

        print(f"\n{'=' * 60}\nModality: {modality}\n{'=' * 60}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            results[modality] = {"error": proc.stderr.strip() or "failed"}
            continue
        try:
            results[modality] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            results[modality] = {"raw_output": proc.stdout}

    out = ROOT / "results" / "modality_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
