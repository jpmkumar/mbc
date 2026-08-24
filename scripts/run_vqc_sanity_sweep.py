#!/usr/bin/env python3
"""Run the optimization-controlled tiny-subset VQC sanity sweep."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def learning_rate_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_summary(
    runs: list[dict],
    sizes: list[int],
    learning_rates: list[float],
    success_threshold: float,
) -> dict:
    rows = []
    for size in sizes:
        for learning_rate in learning_rates:
            setting_runs = [
                run
                for run in runs
                if run["samples_per_class"] == size
                and run["learning_rate"] == learning_rate
            ]
            for model in ("mlp", "vqc"):
                model_runs = [
                    run for run in setting_runs if run["model"] == model
                ]
                if not model_runs:
                    continue
                scores = [
                    run["best_train_tuned_balanced_accuracy"]
                    for run in model_runs
                ]
                passing = sum(score >= success_threshold for score in scores)
                rows.append(
                    {
                        "samples_per_class": size,
                        "learning_rate": learning_rate,
                        "model": model,
                        "mean_best_train_tuned_balanced_accuracy": sum(scores)
                        / len(scores),
                        "min_best_train_tuned_balanced_accuracy": min(scores),
                        "max_best_train_tuned_balanced_accuracy": max(scores),
                        "passing_seeds": passing,
                        "total_seeds": len(scores),
                        "passes_majority_gate": passing >= (len(scores) // 2 + 1),
                    }
                )

    best_by_size_model = []
    for size in sizes:
        for model in ("mlp", "vqc"):
            candidates = [
                row
                for row in rows
                if row["samples_per_class"] == size
                and row["model"] == model
            ]
            if candidates:
                best_by_size_model.append(
                    max(
                        candidates,
                        key=lambda row: (
                            row["passing_seeds"],
                            row["mean_best_train_tuned_balanced_accuracy"],
                        ),
                    )
                )

    mlp_passes = any(
        row["model"] == "mlp" and row["passes_majority_gate"]
        for row in rows
    )
    vqc_passes = any(
        row["model"] == "vqc" and row["passes_majority_gate"]
        for row in rows
    )
    if mlp_passes and vqc_passes:
        verdict = (
            "Both MLP and VQC pass the optimization-controlled sanity gate. "
            "Treat VQC mechanics as validated and diagnose staged training "
            "and representation quality next."
        )
    elif mlp_passes:
        verdict = (
            "The MLP passes but the VQC does not. This supports a VQC-specific "
            "optimization or capacity limitation under the tested budget."
        )
    elif vqc_passes:
        verdict = (
            "The VQC passes while the MLP does not. Recheck comparator "
            "optimization before interpreting a quantum-specific effect."
        )
    else:
        verdict = (
            "Neither head passes the controlled sanity gate. Do not attribute "
            "failure specifically to the VQC; inspect the rank-deficient "
            "features and comparator capacity."
        )

    return {
        "success_threshold": success_threshold,
        "rows": rows,
        "best_by_size_model": best_by_size_model,
        "mlp_passes_any_setting": mlp_passes,
        "vqc_passes_any_setting": vqc_passes,
        "verdict": verdict,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run matched MLP/VQC sanity checks with fixed optimizer steps "
            "over tiny balanced feature subsets."
        )
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/histopath/vqc_controlled_sanity"),
    )
    parser.add_argument("--sizes-per-class", nargs="+", type=int, default=[16, 32])
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[1e-3, 3e-3, 1e-2],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--val-per-class", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    parser.add_argument("--stop-after-success", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collected_runs = []
    for size in args.sizes_per_class:
        for learning_rate in args.learning_rates:
            setting_dir = (
                args.output_dir
                / f"n{size}_lr{learning_rate_tag(learning_rate)}"
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("diagnose_histopath_vqc.py")),
                "--feature-cache",
                str(args.feature_cache),
                "--output-dir",
                str(setting_dir),
                "--models",
                "mlp",
                "vqc",
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--sample-seed",
                str(args.sample_seed),
                "--train-per-class",
                str(size),
                "--val-per-class",
                str(args.val_per_class),
                "--max-steps",
                str(args.max_steps),
                "--eval-every-steps",
                str(args.eval_every_steps),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(learning_rate),
                "--weight-decay",
                "0",
                "--grad-clip",
                "1.0",
                "--n-qubits",
                "8",
                "--n-layers",
                "2",
                "--encoding",
                "angle_y",
                "--entanglement",
                "linear",
                "--backend",
                "default.qubit",
                "--diff-method",
                "backprop",
                "--success-train-balanced-accuracy",
                str(args.success_threshold),
                "--stop-after-success",
                str(args.stop_after_success),
            ]
            print(
                f"\n=== N/class={size} learning_rate={learning_rate:g} ===",
                flush=True,
            )
            subprocess.run(command, check=True)

            setting_summary = json.loads(
                (setting_dir / "summary.json").read_text()
            )
            for run in setting_summary["runs"]:
                collected_runs.append(
                    {
                        "samples_per_class": size,
                        "learning_rate": learning_rate,
                        **run,
                    }
                )

    summary = build_summary(
        collected_runs,
        sizes=args.sizes_per_class,
        learning_rates=args.learning_rates,
        success_threshold=args.success_threshold,
    )
    summary["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    summary_path = args.output_dir / "controlled_sanity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    csv_path = args.output_dir / "controlled_sanity_summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary["rows"][0]),
        )
        writer.writeheader()
        writer.writerows(summary["rows"])

    print("\nControlled sanity verdict:")
    print(summary["verdict"])
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
