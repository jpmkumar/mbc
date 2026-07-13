#!/usr/bin/env python3
"""Train and evaluate histopathology models with patient-level stratified k-fold CV."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.constants import HISTOPATH_MODALITY
from src.data.dataloaders import create_dataloaders
from src.data.histopath_splits import (
    load_histopath_folds,
    resolve_archive_path,
    split_train_val_patients,
    write_fold_split_manifests,
)
from src.eval.metrics import evaluate_model
from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel
from src.train.accelerator import configure_runtime, maybe_compile_model
from src.train.seed import set_seed
from src.train.trainer import HybridTrainer


from src.utils.metrics import threshold_for_target_recall, threshold_sweep


def _metrics_summary(metrics: dict) -> dict:
    return {
        k: float(v)
        for k, v in metrics.items()
        if k not in ("labels", "preds", "probs", "roc") and isinstance(v, (int, float))
    }


def _resolve_eval_threshold(
    model,
    val_loader,
    device,
    train_cfg: dict,
) -> tuple[float, dict]:
    """Pick a decision threshold on validation only (real-world safe)."""
    default_threshold = float(train_cfg.get("eval_threshold", 0.5))
    meta = {
        "tuned": False,
        "threshold": default_threshold,
        "threshold_metric": train_cfg.get("threshold_metric", "f1"),
        "val_score_at_threshold": None,
    }
    if not train_cfg.get("tune_threshold", False):
        return default_threshold, meta

    val_full = evaluate_model(model, val_loader, device, threshold=default_threshold)
    metric_name = str(train_cfg.get("threshold_metric", "f1"))
    if metric_name == "recall_target":
        target_recall = float(train_cfg.get("target_recall", 0.90))
        chosen_threshold, best = threshold_for_target_recall(
            val_full["labels"], val_full["probs"], target_recall=target_recall
        )
        score_value = float(best["recall"])
    else:
        _, best = threshold_sweep(
            val_full["labels"], val_full["probs"], metric=metric_name
        )
        chosen_threshold = float(best["threshold"])
        score_value = float(best.get(metric_name, best["f1"]))
    meta.update(
        {
            "tuned": True,
            "threshold": chosen_threshold,
            "threshold_metric": metric_name,
            "val_score_at_threshold": score_value,
            "val_metrics_at_threshold": _metrics_summary(best),
        }
    )
    if metric_name == "recall_target":
        print(
            f"Threshold tuning on val: recall={score_value:.3f} "
            f"(target {train_cfg.get('target_recall', 0.90)}) "
            f"at threshold={meta['threshold']:.2f}"
        )
    else:
        print(
            f"Threshold tuning on val: {metric_name}={score_value:.3f} "
            f"at threshold={meta['threshold']:.2f}"
        )
    return meta["threshold"], meta


def _build_model(config: dict, experiment: str):
    model_cfg = config["model"]
    quantum_cfg = model_cfg.get("quantum", {})
    kwargs = dict(
        num_modalities=model_cfg["num_modality_tokens"],
        num_classes=model_cfg["num_classes"],
        encoder_dim=model_cfg["encoder_dim"],
        projection_dim=model_cfg["projection_dim"],
        compression_dims=model_cfg["compression_dims"],
        transformer_layers=model_cfg["transformer"]["num_layers"],
        transformer_heads=model_cfg["transformer"]["num_heads"],
    )
    if experiment in ("E2", "classical"):
        return ClassicalBreastCancerModel(**kwargs)
    return HybridBreastCancerModel(
        **kwargs,
        n_qubits=quantum_cfg.get("n_qubits", 8),
        n_vqc_layers=quantum_cfg.get("n_layers", 2),
        entanglement=quantum_cfg.get("entanglement", "linear"),
        quantum_feature_norm=quantum_cfg.get("feature_norm", True),
        quantum_full_readout=quantum_cfg.get("full_readout", True),
    )


def _prepare_fold_manifests(
    splits_dir: Path,
    archive_path: Path,
    fold: int,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    patient_df = pd.read_csv(splits_dir / "patient_stats.csv")
    folds = load_histopath_folds(splits_dir)
    fold_info = next(item for item in folds if item["fold"] == fold)
    train_manifest = pd.read_csv(fold_info["train"])
    test_manifest = pd.read_csv(fold_info["test"])

    train_patient_ids = sorted(str(pid) for pid in train_manifest["patient_id"].unique().tolist())
    test_patient_ids = sorted(str(pid) for pid in test_manifest["patient_id"].unique().tolist())
    train_ids, val_ids = split_train_val_patients(
        patient_df,
        train_patient_ids,
        val_ratio=val_ratio,
        seed=seed + fold,
    )

    fold_dir = splits_dir / "runtime" / f"fold_{fold}"
    paths = write_fold_split_manifests(
        archive_path,
        patient_df,
        fold_dir,
        train_ids,
        val_ids,
        test_patient_ids,
    )
    return {name: str(path) for name, path in paths.items()}


def _run_fold(
    config: dict,
    splits: dict[str, str],
    archive_path: Path,
    experiment: str,
    fold: int,
    seed: int,
    quick: bool,
    max_samples: int | None,
    max_eval_samples: int | None = None,
) -> dict:
    set_seed(seed + fold)
    runtime = configure_runtime(config)
    train_cfg = config["training"]
    data_cfg = config["data"]

    loaders = create_dataloaders(
        splits,
        batch_size=train_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=runtime["num_workers"],
        modality_filter=[HISTOPATH_MODALITY],
        preprocess_config=data_cfg.get("preprocessing"),
        prefetch_factor=runtime["prefetch_factor"],
        data_root=str(archive_path),
        max_samples=max_samples,
        max_eval_samples=max_eval_samples,
    )

    use_hybrid = experiment in ("E3", "hybrid")
    model = _build_model(config, experiment)
    suffix = config.get("project", {}).get("experiment_suffix", "")
    exp_name = (
        f"{experiment}_histopath_fold{fold}{suffix}_seed{seed}"
        if experiment in ("E2", "E3")
        else f"{experiment}_fold{fold}{suffix}_seed{seed}"
    )

    if train_cfg.get("compile_model", False):
        model = maybe_compile_model(model, enabled=True)

    train_eval_loader = None
    if use_hybrid and train_cfg.get("cache_frozen_backbone_features", True):
        train_eval_loader = create_dataloaders(
            {"train": splits["train"]},
            batch_size=train_cfg["batch_size"],
            image_size=data_cfg["image_size"],
            num_workers=runtime["num_workers"],
            modality_filter=[HISTOPATH_MODALITY],
            eval_train_transforms=True,
            preprocess_config=data_cfg.get("preprocessing"),
            prefetch_factor=runtime["prefetch_factor"],
            data_root=str(archive_path),
            max_samples=max_samples,
            max_eval_samples=max_eval_samples,
        )["train"]

    trainer = HybridTrainer(
        model,
        loaders["train"],
        loaders["val"],
        config,
        test_loader=loaders["test"],
        experiment_name=exp_name,
        train_eval_loader=train_eval_loader,
    )
    if quick:
        trainer.stage_a_epochs = 1
        trainer.stage_b_epochs = 1 if use_hybrid else 0
        trainer.stage_c_epochs = 1 if use_hybrid else 0
        if not use_hybrid:
            trainer.stage_a_epochs = 2

    train_metrics = trainer.train()
    threshold, threshold_meta = _resolve_eval_threshold(
        model,
        loaders["val"],
        str(trainer.device),
        train_cfg,
    )
    test_metrics = evaluate_model(
        model, loaders["test"], trainer.device, threshold=threshold
    )
    summary = _metrics_summary(test_metrics)
    summary["threshold"] = threshold
    return {
        "fold": fold,
        "experiment": experiment,
        "seed": seed,
        "train_metrics": train_metrics,
        "test_metrics": summary,
        "threshold_tuning": threshold_meta,
    }


def _friedman_summary(results_by_model: dict[str, list[float]]) -> dict:
    models = list(results_by_model.keys())
    if len(models) < 2:
        return {"error": "Need at least two models for Friedman test."}

    samples = [results_by_model[m] for m in models]
    if not all(len(samples[0]) == len(row) for row in samples[1:]):
        return {"error": "All models must have the same number of fold scores."}
    if len(samples[0]) < 2:
        return {"error": "Need at least two folds for Friedman test."}

    stat, p_value = stats.friedmanchisquare(*samples)
    summary = {
        "models": models,
        "metric_per_fold": results_by_model,
        "friedman_statistic": float(stat),
        "p_value": float(p_value),
    }
    for model, scores in results_by_model.items():
        summary[f"{model}_mean"] = float(np.mean(scores))
        summary[f"{model}_std"] = float(np.std(scores, ddof=0))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Histopathology k-fold train/test")
    parser.add_argument("--config", default="configs/histopath.yaml")
    parser.add_argument(
        "--experiment",
        default="E2",
        choices=["E2", "E3", "classical", "hybrid"],
    )
    parser.add_argument("--compare-classical", action="store_true")
    parser.add_argument("--fold", type=int, default=None, help="Run one fold only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit train patches per fold (debug/smoke test)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit val/test patches per fold (defaults to --max-samples when --quick)",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Override archive path from split_stats.json",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits/histopath",
        help="Directory containing patient_stats.csv and folds/",
    )
    args = parser.parse_args()

    splits_dir = ROOT / args.splits_dir
    if not (splits_dir / "patient_stats.csv").exists():
        raise SystemExit(
            "Missing histopath splits. Run:\n"
            "  python data/download/split_histopath_archive.py --mode cv --folds 5"
        )

    archive_path = resolve_archive_path(args.archive_path, splits_dir)
    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    folds = load_histopath_folds(splits_dir)
    fold_ids = [args.fold] if args.fold is not None else [item["fold"] for item in folds]
    val_ratio = float(config.get("training", {}).get("val_ratio", 0.1))

    experiments = ["E2", "E3"] if args.compare_classical else [args.experiment]
    all_results = {exp: [] for exp in experiments}
    friedman_input = {exp: [] for exp in experiments}
    max_eval_samples = args.max_eval_samples
    if args.quick and max_eval_samples is None and args.max_samples is not None:
        max_eval_samples = args.max_samples

    for fold in fold_ids:
        split_paths = _prepare_fold_manifests(
            splits_dir, archive_path, fold, val_ratio=val_ratio, seed=args.seed
        )
        for experiment in experiments:
            print(f"\n=== Fold {fold} | {experiment} ===")
            result = _run_fold(
                config,
                split_paths,
                archive_path,
                experiment,
                fold,
                args.seed,
                args.quick,
                args.max_samples,
                max_eval_samples,
            )
            all_results[experiment].append(result)
            friedman_input[experiment].append(result["test_metrics"]["f1"])
            print(json.dumps(result["test_metrics"], indent=2))

    summary = {
        "archive_path": str(archive_path),
        "folds": fold_ids,
        "results": all_results,
    }
    if len(fold_ids) >= 2 and len(experiments) >= 2:
        summary["friedman_f1"] = _friedman_summary(friedman_input)

    for experiment, fold_results in all_results.items():
        scores = [item["test_metrics"]["f1"] for item in fold_results]
        summary[f"{experiment}_f1_mean"] = float(np.mean(scores))
        summary[f"{experiment}_f1_std"] = float(np.std(scores, ddof=0))

    out_dir = ROOT / config["paths"]["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
