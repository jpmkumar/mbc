#!/usr/bin/env python3
"""Run full experiment matrix E1–E8."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.experiments import run_experiment_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="Reduced epochs and single seed")
    args = parser.parse_args()

    config_path = str(ROOT / args.config)
    results = run_experiment_matrix(config_path, quick=args.quick)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
