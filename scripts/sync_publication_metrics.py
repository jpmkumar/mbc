#!/usr/bin/env python3
"""Merge training metrics JSON into publication/publication_metrics.json."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication/publication_metrics.json"

STAGE_MAP = {
    "stage_a": "enhanced_stage_a",
    "stage_b": "enhanced_stage_b",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_metrics(pub: dict, metrics: dict, exp_id: str):
    for exp in pub["mammography_experiments"]:
        if exp["id"] == exp_id:
            for key in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "n_samples",
                "pred_positive_rate",
            ):
                if key in metrics:
                    exp["test"][key] = metrics[key]
            if "confusion_matrix" in metrics:
                exp["test"]["confusion_matrix"] = metrics["confusion_matrix"]
            if "train_time_s" in metrics:
                exp["train_time_s"] = metrics["train_time_s"]
            if "best_stage" in metrics:
                exp["best_stage"] = metrics["best_stage"]
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", help="e.g. results/E3_hybrid_enhanced_seed42_metrics.json")
    parser.add_argument(
        "--exp-id",
        default=None,
        help="publication_metrics experiment id (default: infer from best_stage)",
    )
    args = parser.parse_args()

    pub = load_json(PUB)
    metrics = load_json(ROOT / args.metrics_json)

    exp_id = args.exp_id
    if not exp_id:
        stage = metrics.get("best_stage", "stage_a")
        exp_id = STAGE_MAP.get(stage, "enhanced_stage_a")

    if not apply_metrics(pub, metrics, exp_id):
        raise SystemExit(f"Unknown exp-id: {exp_id}")

    with open(PUB, "w") as f:
        json.dump(pub, f, indent=2)
    print(f"Updated {exp_id} in {PUB}")
    print("Run: python scripts/generate_publication.py")


if __name__ == "__main__":
    main()
