#!/usr/bin/env python3
"""End-to-end pipeline: data -> train -> experiments -> XAI -> figures."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast validation run")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    quick_flag = ["--quick"] if args.quick else []

    # Phase 1: data
    run([py, "data/download/generate_synthetic.py", "--samples", "40"])
    run([py, "data/download/setup_datasets.py"])

    if not args.skip_train:
        # Phase 2-4: experiments
        run([py, "experiments/run_experiments.py"] + quick_flag)

    # Phase 5: XAI + figures
    run([py, "experiments/run_xai.py"])
    run([py, "experiments/generate_figures.py"])
    run([py, "experiments/generate_vqc_diagram.py"])

    print("\nPipeline complete. See results/ and figures/")


if __name__ == "__main__":
    main()
