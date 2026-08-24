#!/usr/bin/env python3
"""Select each Stage-B head's learning rate inside a single fold.

Implements the nested selection rule of
``preregistration/stage_b_protocol_v3_nested_secondary.md``: for each head
independently, take the candidate learning rate with the higher mean best
validation AUPRC across the declared seeds. Nothing here reads the held-out
test split, and the resulting file is what the locked evaluator consumes, so
the choice is reproducible from the committed run directories.

Protocol v2 locks both rates to the values chosen on fold 0. That lock fails to
transfer to fold 3, where the MLP is unstable at 1e-3 but stable at 1e-2. This
script produces the secondary, fairly tuned comparison; it does not replace the
primary one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("mlp", "vqc")
# Two rates that differ by less than this in mean validation AUPRC are treated
# as indistinguishable, and the smaller rate wins for reproducibility.
SELECTION_TIE_TOLERANCE = 1e-6


def setting_tag(feature_transform: str, learning_rate: float) -> str:
    return f"{feature_transform}_lr{f'{learning_rate:g}'.replace('.', 'p')}"


def collect_validation_auprc(
    run_dir: Path,
    model: str,
    seeds: tuple[int, ...],
) -> list[float]:
    """Read each seed's best validation AUPRC from its checkpoint summary."""
    summaries = sorted(run_dir.glob("*.json"))
    if not summaries:
        raise FileNotFoundError(f"No run summary in {run_dir}")

    by_seed: dict[int, float] = {}
    for summary_path in summaries:
        payload = json.loads(summary_path.read_text())
        for run in payload.get("runs", []):
            if run.get("model") != model:
                continue
            value = run.get("best_val_auprc")
            if value is None:
                continue
            by_seed[int(run["seed"])] = float(value)

    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(
            f"{run_dir.name} is missing {model} seeds {missing}; "
            "train the full declared seed set before selecting."
        )
    return [by_seed[seed] for seed in seeds]


def select_learning_rate(
    pilot_dir: Path,
    model: str,
    feature_transform: str,
    candidates: tuple[float, ...],
    seeds: tuple[int, ...],
) -> dict:
    scored = []
    for learning_rate in candidates:
        run_dir = pilot_dir / setting_tag(feature_transform, learning_rate)
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        values = collect_validation_auprc(run_dir, model, seeds)
        scored.append(
            {
                "learning_rate": learning_rate,
                "mean_best_val_auprc": sum(values) / len(values),
                "min_best_val_auprc": min(values),
                "max_best_val_auprc": max(values),
                "seeds": len(values),
            }
        )

    ranked = sorted(
        scored,
        key=lambda row: (-row["mean_best_val_auprc"], row["learning_rate"]),
    )
    best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None
    tied = runner_up is not None and (
        best["mean_best_val_auprc"] - runner_up["mean_best_val_auprc"]
        <= SELECTION_TIE_TOLERANCE
    )
    if tied:
        best = min(ranked[:2], key=lambda row: row["learning_rate"])

    return {
        "model": model,
        "feature_transform": feature_transform,
        "learning_rate": best["learning_rate"],
        "seeds": best["seeds"],
        "mean_best_val_auprc": best["mean_best_val_auprc"],
        "selected_by": "mean_best_validation_auprc",
        "resolved_by_tie_rule": tied,
        "candidates": scored,
        "validation_auprc_sensitivity": (
            max(row["mean_best_val_auprc"] for row in scored)
            - min(row["mean_best_val_auprc"] for row in scored)
        ),
    }


def summarize(selections: dict[str, dict], fold: int) -> dict:
    matches_locked = {
        "mlp": abs(selections["mlp"]["learning_rate"] - 1e-3) < 1e-12,
        "vqc": abs(selections["vqc"]["learning_rate"] - 1e-2) < 1e-12,
    }
    return {
        "protocol": "stage_b_protocol_v3_nested_secondary",
        "declaration": (
            "preregistration/stage_b_protocol_v3_nested_secondary.md"
        ),
        "analysis_role": "secondary",
        "fold": fold,
        "selection_endpoint": "validation_auprc",
        "held_out_test_evaluated": False,
        "best_by_model": selections,
        "reproduces_locked_v2_choice": matches_locked,
        "identical_to_locked_v2": all(matches_locked.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--feature-transform", default="raw")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = tuple(args.seeds)
    selections = {
        model: select_learning_rate(
            args.pilot_dir,
            model,
            args.feature_transform,
            tuple(args.learning_rates),
            seeds,
        )
        for model in MODELS
    }
    summary = summarize(selections, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    for model in MODELS:
        record = selections[model]
        print(
            f"Fold {args.fold} {model}: lr={record['learning_rate']:g} "
            f"mean val AUPRC {record['mean_best_val_auprc']:.6f} "
            f"(sensitivity {record['validation_auprc_sensitivity']:.6f})"
        )
    if summary["identical_to_locked_v2"]:
        print(
            "Nested selection reproduces the locked v2 rates, so the v3 test "
            "result on this fold must equal the v2 result."
        )
    else:
        changed = [m for m, ok in summary["reproduces_locked_v2_choice"].items() if not ok]
        print(
            "Nested selection differs from locked v2 for: "
            f"{', '.join(changed)}. The v2 rate did not transfer to this fold."
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
