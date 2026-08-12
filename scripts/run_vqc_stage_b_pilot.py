#!/usr/bin/env python3
"""Run matched frozen-feature MLP/VQC Stage-B pilots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def learning_rate_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_summary(runs: list[dict]) -> dict:
    settings = sorted(
        {
            (run["feature_transform"], run["learning_rate"])
            for run in runs
        }
    )
    rows = []
    for transform, learning_rate in settings:
        for model in ("mlp", "vqc"):
            model_runs = [
                run
                for run in runs
                if run["feature_transform"] == transform
                and run["learning_rate"] == learning_rate
                and run["model"] == model
            ]
            if not model_runs:
                continue

            def values(key: str) -> list[float]:
                return [float(run[key]) for run in model_runs]

            auprc = values("best_val_auprc")
            auc = values("best_val_auc")
            balanced = values("best_val_tuned_balanced_accuracy")
            train = values("best_train_tuned_balanced_accuracy")
            rows.append(
                {
                    "feature_transform": transform,
                    "learning_rate": learning_rate,
                    "model": model,
                    "seeds": len(model_runs),
                    "mean_best_val_auprc": statistics.mean(auprc),
                    "std_best_val_auprc": statistics.pstdev(auprc),
                    "mean_best_val_auc": statistics.mean(auc),
                    "std_best_val_auc": statistics.pstdev(auc),
                    "mean_best_val_tuned_balanced_accuracy": statistics.mean(
                        balanced
                    ),
                    "std_best_val_tuned_balanced_accuracy": statistics.pstdev(
                        balanced
                    ),
                    "mean_best_train_tuned_balanced_accuracy": statistics.mean(
                        train
                    ),
                    "mean_runtime_s": statistics.mean(
                        values("runtime_s")
                    ),
                }
            )

    best_by_transform_model = []
    for transform in sorted({row["feature_transform"] for row in rows}):
        for model in ("mlp", "vqc"):
            candidates = [
                row
                for row in rows
                if row["feature_transform"] == transform
                and row["model"] == model
            ]
            if candidates:
                best_by_transform_model.append(
                    max(candidates, key=lambda row: row["mean_best_val_auprc"])
                )

    best_by_model = {}
    for model in ("mlp", "vqc"):
        candidates = [row for row in rows if row["model"] == model]
        if candidates:
            best_by_model[model] = max(
                candidates,
                key=lambda row: row["mean_best_val_auprc"],
            )

    transform_deltas = []
    for transform, learning_rate in settings:
        mlp = next(
            (
                row
                for row in rows
                if row["feature_transform"] == transform
                and row["learning_rate"] == learning_rate
                and row["model"] == "mlp"
            ),
            None,
        )
        vqc = next(
            (
                row
                for row in rows
                if row["feature_transform"] == transform
                and row["learning_rate"] == learning_rate
                and row["model"] == "vqc"
            ),
            None,
        )
        if mlp and vqc:
            transform_deltas.append(
                {
                    "feature_transform": transform,
                    "learning_rate": learning_rate,
                    "vqc_minus_mlp_val_auprc": (
                        vqc["mean_best_val_auprc"]
                        - mlp["mean_best_val_auprc"]
                    ),
                    "vqc_minus_mlp_val_auc": (
                        vqc["mean_best_val_auc"]
                        - mlp["mean_best_val_auc"]
                    ),
                }
            )

    mlp_best = best_by_model.get("mlp")
    vqc_best = best_by_model.get("vqc")
    if mlp_best and vqc_best:
        delta = (
            vqc_best["mean_best_val_auprc"]
            - mlp_best["mean_best_val_auprc"]
        )
        if delta > 0.01:
            verdict = (
                "The validation-tuned VQC exceeds the matched MLP by more than "
                "0.01 AUPRC in this exploratory pilot. Freeze the selected "
                "settings and confirm on untouched folds before any advantage claim."
            )
        elif delta < -0.01:
            verdict = (
                "The validation-tuned VQC trails the matched MLP by more than "
                "0.01 AUPRC. This supports a frozen-head limitation under the "
                "tested representation and optimization budget."
            )
        else:
            verdict = (
                "The best VQC and matched MLP are within 0.01 validation AUPRC. "
                "Treat the frozen-head pilot as practically tied and proceed "
                "to representation and Stage-C stability analysis."
            )
    else:
        delta = None
        verdict = "Incomplete matched-head results; do not interpret the pilot."

    return {
        "selection_endpoint": "validation_auprc",
        "exploratory_only": True,
        "rows": rows,
        "best_by_transform_model": best_by_transform_model,
        "best_by_model": best_by_model,
        "best_vqc_minus_mlp_val_auprc": delta,
        "matched_setting_deltas": transform_deltas,
        "verdict": verdict,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched MLP and VQC heads on frozen Stage-A features "
            "without touching the held-out test split."
        )
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/histopath/vqc_stage_b_pilot"),
    )
    parser.add_argument(
        "--feature-transforms",
        nargs="+",
        choices=["raw", "standardize"],
        default=["raw", "standardize"],
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[1e-3, 3e-3, 1e-2],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mlp", "vqc"],
        default=["mlp", "vqc"],
        help=(
            "Restrict training to these heads. Use a single head to add seeds "
            "for one pre-registered cell without retraining the other."
        ),
    )
    parser.add_argument(
        "--summary-name",
        default="stage_b_pilot_summary",
        help="Base name for the summary files, so partial runs do not clobber.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--train-per-class", type=int, default=4096)
    parser.add_argument("--val-per-class", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collected_runs = []
    for transform in args.feature_transforms:
        for learning_rate in args.learning_rates:
            setting_dir = (
                args.output_dir
                / f"{transform}_lr{learning_rate_tag(learning_rate)}"
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("diagnose_histopath_vqc.py")),
                "--feature-cache",
                str(args.feature_cache),
                "--output-dir",
                str(setting_dir),
                "--models",
                *args.models,
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--sample-seed",
                str(args.sample_seed),
                "--feature-transform",
                transform,
                "--train-per-class",
                str(args.train_per_class),
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
                "1.0",
                "--stop-after-success",
                "0",
            ]
            print(
                f"\n=== transform={transform} lr={learning_rate:g} ===",
                flush=True,
            )
            subprocess.run(command, check=True)
            setting_summary = json.loads(
                (setting_dir / "summary.json").read_text()
            )
            for run in setting_summary["runs"]:
                collected_runs.append(
                    {
                        "feature_transform": transform,
                        "learning_rate": learning_rate,
                        **run,
                    }
                )

    summary = build_summary(collected_runs)
    summary["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary_path = args.output_dir / f"{args.summary_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    csv_path = args.output_dir / f"{args.summary_name}.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary["rows"][0]),
        )
        writer.writeheader()
        writer.writerows(summary["rows"])

    print("\nStage-B pilot verdict:")
    print(summary["verdict"])
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
