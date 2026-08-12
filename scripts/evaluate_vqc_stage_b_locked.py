#!/usr/bin/env python3
"""Evaluate validation-locked Stage-B heads once on the held-out test cache."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_histopath_vqc import (  # noqa: E402
    apply_feature_transform,
    build_head,
)
from src.utils.metrics import compute_metrics_at_threshold  # noqa: E402


MODELS = ("mlp", "vqc")
PRACTICAL_AUPRC_MARGIN = 0.01


def load_locked_settings(path: Path) -> dict[str, dict]:
    """Load settings selected without access to the held-out test split."""
    payload = json.loads(path.read_text())
    if payload.get("held_out_test_evaluated") is not False:
        raise ValueError(
            "Locked selection must explicitly state "
            "held_out_test_evaluated=false."
        )
    if payload.get("selection_endpoint") != "validation_auprc":
        raise ValueError("Locked settings must be selected by validation AUPRC.")

    settings = payload.get("best_by_model", {})
    if set(settings) != set(MODELS):
        raise ValueError("Locked selection must contain exactly MLP and VQC.")
    for model in MODELS:
        if settings[model].get("model") != model:
            raise ValueError(f"Locked {model} record has inconsistent model.")
    return settings


def setting_tag(setting: dict) -> str:
    learning_rate = f"{float(setting['learning_rate']):g}".replace(".", "p")
    return f"{setting['feature_transform']}_lr{learning_rate}"


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities, targets = [], []
    loss_sum = 0.0
    model.eval()
    for batch_features, batch_labels in loader:
        logits = model(batch_features).float()
        probs = torch.softmax(logits, dim=1)[:, 1]
        loss_sum += float(
            F.cross_entropy(logits, batch_labels, reduction="sum")
        )
        probabilities.append(probs.cpu())
        targets.append(batch_labels.cpu())
    all_probs = torch.cat(probabilities).numpy()
    all_targets = torch.cat(targets).numpy()
    return all_targets, all_probs, loss_sum / len(all_targets)


def evaluate_checkpoint(
    checkpoint_path: Path,
    expected_model: str,
    expected_seed: int,
    expected_setting: dict,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("selection") != "best_val_auprc":
        raise ValueError(f"Wrong checkpoint selection: {checkpoint_path}")
    if checkpoint.get("model") != expected_model:
        raise ValueError(f"Wrong model in checkpoint: {checkpoint_path}")
    if int(checkpoint.get("seed")) != expected_seed:
        raise ValueError(f"Wrong seed in checkpoint: {checkpoint_path}")

    config = checkpoint["config"]
    expected_transform = expected_setting["feature_transform"]
    expected_lr = float(expected_setting["learning_rate"])
    if config.get("feature_transform") != expected_transform:
        raise ValueError(f"Wrong feature transform: {checkpoint_path}")
    if not np.isclose(float(config.get("learning_rate")), expected_lr):
        raise ValueError(f"Wrong learning rate: {checkpoint_path}")

    validation_metrics = checkpoint.get("best_val_auprc_metrics")
    if not validation_metrics or "tuned_threshold" not in validation_metrics:
        raise ValueError(
            f"Checkpoint lacks a validation-derived threshold: {checkpoint_path}"
        )
    threshold = float(validation_metrics["tuned_threshold"])

    args = SimpleNamespace(**config)
    model = build_head(expected_model, args, test_features.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    labels, probs, cross_entropy = predict(
        model,
        test_features,
        test_labels,
        batch_size,
    )
    metrics = compute_metrics_at_threshold(labels, probs, threshold)
    metrics["cross_entropy"] = cross_entropy
    return {
        "model": expected_model,
        "seed": expected_seed,
        "feature_transform": expected_transform,
        "learning_rate": expected_lr,
        "checkpoint": str(checkpoint_path),
        "selection_endpoint": "validation_auprc",
        "selection_validation_auprc": float(
            checkpoint["best_val_auprc"]
        ),
        "threshold_source": "best_val_auprc_checkpoint_validation_subset",
        "test_metrics": metrics,
    }, probs


def patient_row_groups(patient_ids: Sequence[str]) -> list[np.ndarray]:
    """Group row indices by patient so resampling keeps patients intact."""
    _, membership = np.unique(np.asarray(patient_ids), return_inverse=True)
    order = np.argsort(membership, kind="stable")
    boundaries = np.searchsorted(
        membership[order], np.arange(membership.max() + 2)
    )
    return [
        order[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
    ]


def patient_cluster_bootstrap(
    labels: np.ndarray,
    patient_ids: Sequence[str],
    probabilities: dict[str, dict[int, np.ndarray]],
    replicates: int,
    seed: int,
) -> dict:
    """Bootstrap the paired AUPRC gap by resampling whole patients.

    Patches from one patient are correlated, so sample-level intervals are
    anticonservative. Resampling patient clusters keeps that dependence.
    """
    groups = patient_row_groups(patient_ids)
    seeds = sorted(set(probabilities["mlp"]) & set(probabilities["vqc"]))
    generator = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        chosen = generator.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in chosen])
        replicate_labels = labels[rows]
        if replicate_labels.min() == replicate_labels.max():
            continue
        deltas.append(
            float(
                np.mean(
                    [
                        average_precision_score(
                            replicate_labels, probabilities["vqc"][s][rows]
                        )
                        - average_precision_score(
                            replicate_labels, probabilities["mlp"][s][rows]
                        )
                        for s in seeds
                    ]
                )
            )
        )
    if not deltas:
        raise ValueError("Every bootstrap replicate was single-class.")

    lower = float(np.percentile(deltas, 2.5))
    upper = float(np.percentile(deltas, 97.5))
    return {
        "method": "patient_cluster_percentile_bootstrap",
        "statistic": "mean_over_seeds_vqc_minus_mlp_test_auprc",
        "unique_test_patients": len(groups),
        "requested_replicates": replicates,
        "usable_replicates": len(deltas),
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_mean": float(np.mean(deltas)),
        "excludes_zero": lower > 0.0 or upper < 0.0,
        "within_practical_margin": (
            lower > -PRACTICAL_AUPRC_MARGIN and upper < PRACTICAL_AUPRC_MARGIN
        ),
    }


def summarize_results(
    results: list[dict],
    fold: int = 0,
    bootstrap: dict | None = None,
) -> dict:
    by_model = {}
    for model in MODELS:
        model_results = [row for row in results if row["model"] == model]
        if not model_results:
            raise ValueError(f"Missing locked test results for {model}.")
        metric_summary = {}
        for metric in (
            "auprc",
            "auc",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
        ):
            values = [
                float(row["test_metrics"][metric]) for row in model_results
            ]
            metric_summary[metric] = {
                "mean": statistics.mean(values),
                "std_across_initialization_seeds": statistics.pstdev(values),
            }
        by_model[model] = {
            "feature_transform": model_results[0]["feature_transform"],
            "learning_rate": model_results[0]["learning_rate"],
            "seeds": len(model_results),
            "metrics": metric_summary,
        }

    indexed = {
        (row["model"], int(row["seed"])): row
        for row in results
    }
    common_seeds = sorted(
        {
            int(row["seed"])
            for row in results
            if ("mlp", int(row["seed"])) in indexed
            and ("vqc", int(row["seed"])) in indexed
        }
    )
    paired_auprc_deltas = [
        (
            indexed[("vqc", seed)]["test_metrics"]["auprc"]
            - indexed[("mlp", seed)]["test_metrics"]["auprc"]
        )
        for seed in common_seeds
    ]
    mean_delta = statistics.mean(paired_auprc_deltas)
    fold_label = f"Fold {fold}"
    if abs(mean_delta) <= PRACTICAL_AUPRC_MARGIN:
        verdict = (
            "Locked MLP and VQC heads are within the pre-specified 0.01 "
            f"practical AUPRC margin on {fold_label}. This is no detected "
            "quantum advantage, not a formal equivalence claim."
        )
    elif mean_delta > 0:
        verdict = (
            "The locked VQC exceeds the MLP by more than 0.01 test AUPRC on "
            f"{fold_label}; replication on untouched patient folds is required."
        )
    else:
        verdict = (
            "The locked VQC trails the MLP by more than 0.01 test AUPRC on "
            f"{fold_label} under the matched frozen-feature protocol."
        )

    if bootstrap is None:
        uncertainty_note = (
            "Seed variation measures initialization sensitivity only. This "
            "cache has no patient IDs, so patient-cluster confidence "
            "intervals cannot be computed."
        )
    else:
        uncertainty_note = (
            "The 95% interval resamples whole test patients, so it reflects "
            "between-patient variation rather than initialization noise. It "
            "covers this fold alone and is not a cross-fold interval."
        )
        if bootstrap["within_practical_margin"]:
            verdict += (
                " The patient-cluster interval also lies entirely inside the "
                "practical margin on this fold"
            )
            verdict += (
                ", while excluding zero, so the residual gap is measurable "
                "but too small to matter clinically."
                if bootstrap["excludes_zero"]
                else "."
            )
        elif bootstrap["excludes_zero"]:
            verdict += (
                " The patient-cluster interval excludes zero, so the gap is "
                "resolvable above between-patient noise on this fold."
            )
        else:
            verdict += (
                " The patient-cluster interval spans zero and extends beyond "
                "the practical margin, so this fold is underpowered to "
                "separate the heads."
            )

    return {
        "protocol": f"one_time_validation_locked_fold{fold}_test",
        "fold": fold,
        "held_out_test_evaluated": True,
        "primary_metric": "auprc",
        "practical_auprc_margin": PRACTICAL_AUPRC_MARGIN,
        "by_model": by_model,
        "paired_seed_auprc_deltas": paired_auprc_deltas,
        "mean_vqc_minus_mlp_test_auprc": mean_delta,
        "patient_cluster_bootstrap": bootstrap,
        "uncertainty_note": uncertainty_note,
        "verdict": verdict,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply validation-locked MLP and VQC checkpoints to the held-out "
            "Fold-0 test feature cache without test-time tuning."
        )
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument(
        "--locked-selection",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/histopath/vqc_stage_b_locked_test"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    locked_path = (
        args.locked_selection
        if args.locked_selection is not None
        else args.pilot_dir / "locked_pilot_selection.json"
    )
    for path in (args.feature_cache, args.pilot_dir, locked_path):
        if not path.exists():
            raise FileNotFoundError(path)

    result_path = args.output_dir / "locked_test_summary.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to repeat held-out evaluation: {result_path}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locked_settings = load_locked_settings(locked_path)
    cache = torch.load(
        args.feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    if "test" not in cache:
        raise ValueError("Feature cache has no held-out test split.")
    fit_features = cache["train"]["features"].float()
    test_features_raw = cache["test"]["features"].float()
    test_labels = cache["test"]["labels"].long()
    test_patient_ids = cache["test"].get("patient_ids")
    if test_patient_ids is not None and len(test_patient_ids) != len(test_labels):
        raise ValueError("Cached patient IDs are misaligned with test labels.")

    results = []
    probabilities: dict[str, dict[int, np.ndarray]] = {
        model: {} for model in MODELS
    }
    for model in MODELS:
        setting = locked_settings[model]
        transform = setting["feature_transform"]
        _, test_features, transform_metadata = apply_feature_transform(
            fit_features,
            fit_features[:1],
            test_features_raw,
            transform,
        )
        directory = args.pilot_dir / setting_tag(setting)
        for seed in args.seeds:
            checkpoint_path = (
                directory / f"{model}_seed{seed}_best_val_auprc.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            print(
                f"Evaluating locked {model} seed={seed} "
                f"transform={transform} lr={setting['learning_rate']}"
            )
            result, test_probabilities = evaluate_checkpoint(
                checkpoint_path,
                model,
                seed,
                setting,
                test_features,
                test_labels,
                args.batch_size,
            )
            result["transform_metadata"] = transform_metadata
            results.append(result)
            probabilities[model][seed] = test_probabilities

    bootstrap = None
    if test_patient_ids is not None and args.bootstrap_replicates > 0:
        print(
            f"Bootstrapping {args.bootstrap_replicates} patient-cluster "
            "resamples of the paired AUPRC gap"
        )
        bootstrap = patient_cluster_bootstrap(
            test_labels.numpy(),
            test_patient_ids,
            probabilities,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )

    summary = summarize_results(results, fold=args.fold, bootstrap=bootstrap)
    summary["source"] = {
        "feature_cache": str(args.feature_cache),
        "pilot_dir": str(args.pilot_dir),
        "locked_selection": str(locked_path),
        "test_samples": len(test_labels),
        "test_class_counts": torch.bincount(
            test_labels, minlength=2
        ).tolist(),
        "test_patient_ids_available": test_patient_ids is not None,
    }
    summary["runs"] = results
    result_path.write_text(json.dumps(summary, indent=2))

    csv_path = args.output_dir / "locked_test_per_seed.csv"
    with open(csv_path, "w", newline="") as handle:
        fieldnames = [
            "model",
            "seed",
            "feature_transform",
            "learning_rate",
            "selection_validation_auprc",
            "test_auprc",
            "test_auc",
            "test_balanced_accuracy",
            "test_f1",
            "test_precision",
            "test_recall",
            "validation_derived_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            metrics = row["test_metrics"]
            writer.writerow(
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "feature_transform": row["feature_transform"],
                    "learning_rate": row["learning_rate"],
                    "selection_validation_auprc": row[
                        "selection_validation_auprc"
                    ],
                    "test_auprc": metrics["auprc"],
                    "test_auc": metrics["auc"],
                    "test_balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "test_f1": metrics["f1"],
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "validation_derived_threshold": metrics["threshold"],
                }
            )

    print(json.dumps(summary["by_model"], indent=2))
    print(
        "Mean VQC - MLP test AUPRC: "
        f"{summary['mean_vqc_minus_mlp_test_auprc']:+.6f}"
    )
    if bootstrap is not None:
        print(
            "Patient-cluster 95% CI over "
            f"{bootstrap['unique_test_patients']} patients: "
            f"[{bootstrap['ci_lower']:+.6f}, {bootstrap['ci_upper']:+.6f}]"
        )
    print(summary["verdict"])
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
