#!/usr/bin/env python3
"""Export fold test labels + P(IDC+) for ROC/PR figures (Fig.7).

Runs inference with the same histopath eval path as training (TTA optional).
Writes compressed NPZ files that keep arrays (unlike *_metrics.json, which strips them).

Example (Kaggle, fold 0, three arms):

  python scripts/export_histopath_fold_probs.py \\
    --config configs/histopath.yaml \\
    --fold 0 \\
    --archive-path /kaggle/input/breast-histopathology-images \\
    --splits-dir data/splits/histopath \\
    --out-dir results/fold0_probs \\
    --arm E2=/path/to/E2_...seed42.pt \\
    --arm E2b=/path/to/E2b_...seed42.pt \\
    --arm E3=/path/to/E3_...seed42.pt \\
    --tta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.constants import HISTOPATH_MODALITY
from src.data.dataloaders import create_dataloaders
from src.data.histopath_splits import resolve_archive_path
from src.eval.metrics import evaluate_model
from src.train.accelerator import configure_runtime
from src.train.seed import set_seed
from src.train.trainer import filter_compatible_state_dict

# Reuse model builder + fold manifests from the CV trainer.
from scripts.train_histopath_cv import _build_model, _prepare_fold_manifests


def _load_checkpoint(model, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {"best_stage": "stage_a", "threshold": None}
    if isinstance(ckpt, dict) and ckpt.get("best_state_dict") is not None:
        state = ckpt["best_state_dict"]
        meta["best_stage"] = str(ckpt.get("best_stage", "stage_a"))
        if ckpt.get("threshold") is not None:
            meta["threshold"] = float(ckpt["threshold"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta["best_stage"] = str(ckpt.get("best_stage", "stage_a"))
    else:
        state = ckpt
    filtered, skipped = filter_compatible_state_dict(model, state)
    model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"  note: skipped {len(skipped)} incompatible keys")
    return meta


def _setup_device(model, config: dict) -> torch.device:
    """Place model on GPU when available and keep classical_device in sync.

    HybridBreastCancerModel.forward() moves inputs to ``self.classical_device``.
    Calling only ``model.to(cuda)`` leaves that attribute on CPU and causes
    ``Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)``.
    """
    train_cfg = config.get("training", {})
    requested = str(train_cfg.get("classical_device", "auto")).lower()
    if requested in ("auto", "gpu", "cuda") or (
        requested == "cpu" and torch.cuda.is_available()
    ):
        # histopath.yaml defaults classical_device=cpu for training; export prefers GPU.
        classical = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        classical = requested
    quantum = train_cfg.get("quantum_device", "cpu")
    if hasattr(model, "set_devices"):
        model.set_devices(classical, quantum)
    else:
        model.to(torch.device(classical))
    if hasattr(model, "classical_device"):
        model.classical_device = torch.device(classical)
    print(f"  device: classical={classical} quantum={quantum}")
    return torch.device(classical)


def _scalar_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    preds = (probs >= threshold).astype(int)
    return {
        "n_samples": int(len(labels)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
    }


def export_arm(
    *,
    arm: str,
    ckpt_path: Path,
    config: dict,
    splits: dict[str, str],
    archive_path: Path,
    fold: int,
    seed: int,
    split_name: str,
    tta: bool,
    threshold: float | None,
    out_dir: Path,
    batch_size: int | None,
) -> Path:
    train_cfg = config["training"]
    data_cfg = config["data"]
    runtime = configure_runtime(config)
    bs = int(batch_size or train_cfg.get("batch_size", 64))

    loaders = create_dataloaders(
        splits,
        batch_size=bs,
        image_size=data_cfg["image_size"],
        num_workers=runtime["num_workers"],
        modality_filter=[HISTOPATH_MODALITY],
        preprocess_config=data_cfg.get("preprocessing"),
        prefetch_factor=runtime["prefetch_factor"],
        data_root=str(archive_path),
        max_samples=None,
        max_eval_samples=None,
        augment_config=None,  # eval: no train augment
    )
    if split_name not in loaders:
        raise SystemExit(f"Split {split_name!r} not in loaders ({list(loaders)})")

    model = _build_model(config, arm)
    device = _setup_device(model, config)
    meta = _load_checkpoint(model, ckpt_path)
    # Re-apply devices after load (state_dict load must not leave attrs stale).
    device = _setup_device(model, config)
    model.eval()

    stage = meta["best_stage"]
    if hasattr(model, "set_training_stage"):
        model.set_training_stage(stage)
        print(f"  eval stage = {stage}")

    thr = (
        float(threshold)
        if threshold is not None
        else float(meta["threshold"] if meta["threshold"] is not None else 0.5)
    )

    print(f"  infer {split_name} (tta={tta}, threshold={thr:.4f}) …")
    metrics = evaluate_model(
        model,
        loaders[split_name],
        device=str(device),
        threshold=thr,
        tta=tta,
    )
    labels = np.asarray(metrics["labels"], dtype=np.int8)
    probs = np.asarray(metrics["probs"], dtype=np.float32)
    summary = _scalar_metrics(labels, probs, thr)
    summary.update(
        {
            "arm": arm,
            "fold": fold,
            "seed": seed,
            "split": split_name,
            "tta": bool(tta),
            "best_stage": stage,
            "checkpoint": str(ckpt_path),
        }
    )
    print(
        f"  n={summary['n_samples']}  bal_acc={summary['balanced_accuracy']:.4f}  "
        f"F1={summary['f1']:.4f}  AUC={summary['auc']:.4f}  AUPRC={summary['auprc']:.4f}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"fold{fold}_{arm}_{split_name}_probs.npz"
    np.savez_compressed(
        npz_path,
        labels=labels,
        probs=probs,
        threshold=np.float32(thr),
        arm=np.array(arm),
        fold=np.int32(fold),
        tta=np.bool_(tta),
        best_stage=np.array(stage),
    )
    json_path = out_dir / f"fold{fold}_{arm}_{split_name}_probs_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {npz_path}")
    print(f"  wrote {json_path}")
    return npz_path


def _parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected ARM=/path/to.pt, got {spec!r}"
        )
    arm, path = spec.split("=", 1)
    arm = arm.strip()
    if arm not in ("E2", "E2b", "E3", "E4", "classical", "hybrid", "fusion"):
        raise argparse.ArgumentTypeError(f"Unknown arm {arm!r}")
    ckpt = Path(path.strip()).expanduser()
    if not ckpt.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint not found: {ckpt}")
    return arm, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export histopath fold test probabilities for ROC/PR plots."
    )
    parser.add_argument("--config", default="configs/histopath.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits-dir",
        default="data/splits/histopath",
        help="Directory with patient_stats.csv and folds/",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="IDC image root (Kaggle input). Default: resolve from split_stats.json",
    )
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        required=True,
        metavar="ARM=/path/to.pt",
        help="Repeatable, e.g. --arm E2=/path/E2.pt --arm E3=/path/E3.pt",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which fold split to score (default: test)",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average softmax over geometric TTA views (default: on)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: 0.5 or value stored in checkpoint)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/fold0_probs",
        help="Output directory for .npz + summary JSON",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--reuse-runtime-splits",
        action="store_true",
        help="Use existing splits/runtime/fold_N CSVs if present (skip rewrite)",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    with config_path.open() as f:
        config = yaml.safe_load(f)

    set_seed(args.seed + args.fold)
    splits_dir = (ROOT / args.splits_dir).resolve()
    if not (splits_dir / "patient_stats.csv").is_file():
        raise SystemExit(f"Missing patient_stats.csv under {splits_dir}")

    archive_path = resolve_archive_path(args.archive_path, splits_dir)
    print(f"Archive: {archive_path}")
    print(f"Splits:  {splits_dir}")

    runtime_dir = splits_dir / "runtime" / f"fold_{args.fold}"
    if args.reuse_runtime_splits and (runtime_dir / "test.csv").is_file():
        splits = {
            "train": str(runtime_dir / "train.csv"),
            "val": str(runtime_dir / "val.csv"),
            "test": str(runtime_dir / "test.csv"),
        }
        print(f"Reusing runtime manifests in {runtime_dir}")
    else:
        val_ratio = float(config.get("training", {}).get("val_ratio", 0.1))
        splits = _prepare_fold_manifests(
            splits_dir,
            Path(archive_path),
            args.fold,
            val_ratio=val_ratio,
            seed=args.seed,
        )
        print(f"Wrote runtime manifests for fold {args.fold}")

    out_dir = ROOT / args.out_dir
    written: list[str] = []
    for arm, ckpt in args.arm:
        print(f"\n=== {arm} ← {ckpt} ===")
        path = export_arm(
            arm=arm,
            ckpt_path=ckpt,
            config=config,
            splits=splits,
            archive_path=Path(archive_path),
            fold=args.fold,
            seed=args.seed,
            split_name=args.split,
            tta=bool(args.tta),
            threshold=args.threshold,
            out_dir=out_dir,
            batch_size=args.batch_size,
        )
        written.append(str(path))

    manifest = {
        "fold": args.fold,
        "split": args.split,
        "tta": bool(args.tta),
        "files": written,
    }
    man_path = out_dir / f"fold{args.fold}_{args.split}_probs_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {man_path}")


if __name__ == "__main__":
    main()
