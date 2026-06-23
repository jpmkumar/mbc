#!/usr/bin/env python3
"""Generate publication figures from experiment results."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment_matrix_summary.json")
    args = parser.parse_args()

    pub_script = ROOT / "scripts/generate_publication.py"
    if (ROOT / "publication/publication_metrics.json").exists():
        subprocess.run([sys.executable, str(pub_script)], check=True)
    else:
        print("Run with results/publication_metrics.json for mammography figures.")

    # Legacy experiment matrix figures (E4/E6/E7) when summary exists
    results_path = ROOT / args.results
    if results_path.exists():
        import json
        import matplotlib.pyplot as plt
        import numpy as np

        output_dir = ROOT / "figures"
        with open(results_path) as f:
            summary = json.load(f)
        # ... keep optional LOMO/ablation if matrix exists
        print(f"Optional matrix summary loaded from {results_path}")

    print(f"See publication/PUBLICATION_RESULTS.md for figure list.")


if __name__ == "__main__":
    main()
