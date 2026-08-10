#!/usr/bin/env python3
"""Test whether the VQC can learn fixed, balanced histopathology features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.hybrid_model import ClassicalMLPHead  # noqa: E402
from src.models.vqc import VQCHead  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


MODEL_NAMES = ("linear", "mlp", "vqc")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def balanced_indices(
    labels: torch.Tensor,
    per_class: int,
    seed: int,
) -> torch.Tensor:
    """Select the same number of class-0 and class-1 examples."""
    generator = torch.Generator().manual_seed(seed)
    selected = []
    counts = torch.bincount(labels.long(), minlength=2)
    for class_id in (0, 1):
        class_indices = torch.where(labels == class_id)[0]
        if len(class_indices) < per_class:
            raise ValueError(
                f"Requested {per_class} examples of class {class_id}, "
                f"but cache contains {len(class_indices)} "
                f"(class counts={counts.tolist()})."
            )
        order = torch.randperm(len(class_indices), generator=generator)
        selected.append(class_indices[order[:per_class]])
    combined = torch.cat(selected)
    return combined[torch.randperm(len(combined), generator=generator)]


def subset_from_cache(
    cached_split: dict[str, torch.Tensor],
    per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = cached_split["labels"].long()
    indices = balanced_indices(labels, per_class=per_class, seed=seed)
    features = cached_split["features"][indices].float()
    return features, labels[indices], indices


def make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def build_head(name: str, args, feature_dim: int) -> nn.Module:
    if name == "linear":
        return nn.Linear(feature_dim, 2)
    if name == "mlp":
        return ClassicalMLPHead(feature_dim, 2, hidden=feature_dim)
    if name == "vqc":
        if args.n_qubits != feature_dim:
            raise ValueError(
                f"VQC diagnostic requires n_qubits ({args.n_qubits}) to equal "
                f"cached feature dimension ({feature_dim})."
            )
        return VQCHead(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            num_classes=2,
            entanglement=args.entanglement,
            encoding=args.encoding,
            data_reuploading=args.data_reuploading,
            feature_norm=True,
            full_readout=True,
            backend=args.backend,
            diff_method=args.diff_method,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    for features, labels in loader:
        features = features.to(device)
        logits = model(features).float()
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        labels_all.extend(labels.numpy())
        preds_all.extend(predictions.cpu().numpy())
        probs_all.extend(probabilities[:, 1].cpu().numpy())
    return compute_metrics(labels_all, preds_all, probs_all)


def _trainable_vector(model: nn.Module) -> torch.Tensor:
    parts = [
        parameter.detach().cpu().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return torch.cat(parts) if parts else torch.empty(0)


def _gradient_stats(model: nn.Module) -> dict:
    all_gradients = []
    quantum_gradients = []
    readout_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().abs().cpu().reshape(-1)
        all_gradients.append(gradient)
        if "quantum_layer" in name:
            quantum_gradients.append(gradient)
        if "classifier" in name:
            readout_gradients.append(gradient)

    def summarize(parts: list[torch.Tensor], prefix: str) -> dict:
        if not parts:
            return {
                f"{prefix}_grad_mean": 0.0,
                f"{prefix}_grad_max": 0.0,
                f"{prefix}_near_zero_fraction": 1.0,
            }
        values = torch.cat(parts)
        return {
            f"{prefix}_grad_mean": float(values.mean()),
            f"{prefix}_grad_max": float(values.max()),
            f"{prefix}_near_zero_fraction": float((values < 1e-8).float().mean()),
        }

    return {
        **summarize(all_gradients, "all"),
        **summarize(quantum_gradients, "quantum"),
        **summarize(readout_gradients, "readout"),
    }


@torch.no_grad()
def _probe_stats(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    probe_size = min(64, len(features))
    probe = features[:probe_size].to(device)
    probe_labels = labels[:probe_size].to(device)
    logits = model(probe).float()
    result = {
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std(unbiased=False)),
    }

    if not isinstance(model, VQCHead):
        return result

    normalized = model.feature_norm(probe)
    angles = model.angle_encoder(normalized)
    quantum_output = model.quantum_layer(angles)
    if quantum_output.dim() == 1:
        quantum_output = quantum_output.unsqueeze(0)
    quantum_output = quantum_output.float()
    per_feature_variance = quantum_output.var(dim=0, unbiased=False)
    class_separation = 0.0
    if (probe_labels == 0).any() and (probe_labels == 1).any():
        class_0_mean = quantum_output[probe_labels == 0].mean(dim=0)
        class_1_mean = quantum_output[probe_labels == 1].mean(dim=0)
        class_separation = float(torch.linalg.vector_norm(class_1_mean - class_0_mean))
    result.update(
        {
            "angle_min": float(angles.min()),
            "angle_max": float(angles.max()),
            "angle_mean": float(angles.mean()),
            "angle_std": float(angles.std(unbiased=False)),
            "angle_saturation_fraction": float(
                ((angles < 0.05 * math.pi) | (angles > 0.95 * math.pi))
                .float()
                .mean()
            ),
            "quantum_output_mean": float(quantum_output.mean()),
            "quantum_output_std": float(quantum_output.std(unbiased=False)),
            "quantum_output_variance_mean": float(per_feature_variance.mean()),
            "quantum_class_mean_distance": class_separation,
            "quantum_output_near_constant_fraction": float(
                (per_feature_variance < 1e-8).float().mean()
            ),
        }
    )
    return result


def _compact_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key != "roc"
    }


def train_one(
    model_name: str,
    seed: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    args,
    output_dir: Path,
) -> dict:
    set_seed(seed)
    device = torch.device("cpu")
    model = build_head(model_name, args, train_features.shape[1]).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(
        train_features,
        train_labels,
        args.batch_size,
        shuffle=True,
        seed=seed,
    )
    train_eval_loader = make_loader(
        train_features,
        train_labels,
        args.batch_size,
        shuffle=False,
        seed=seed,
    )
    val_loader = make_loader(
        val_features,
        val_labels,
        args.batch_size,
        shuffle=False,
        seed=seed,
    )

    history = []
    best_train_score = -math.inf
    best_val_score = -math.inf
    best_state = None
    successful_epochs = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        before = _trainable_vector(model)
        epoch_loss = 0.0
        gradient_rows = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).float()
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss for {model_name}, seed={seed}, epoch={epoch}"
                )
            loss.backward()
            gradient_rows.append(_gradient_stats(model))
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())

        after = _trainable_vector(model)
        update_norm = float(torch.linalg.vector_norm(after - before))
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        probe = _probe_stats(model, train_features, train_labels, device)

        averaged_gradients = {}
        for key in gradient_rows[0]:
            averaged_gradients[key] = float(
                np.mean([row[key] for row in gradient_rows])
            )

        row = {
            "model": model_name,
            "seed": seed,
            "epoch": epoch,
            "loss": epoch_loss / max(len(train_loader), 1),
            "parameter_update_norm": update_norm,
            **averaged_gradients,
            **probe,
            **{f"train_{k}": v for k, v in _compact_metrics(train_metrics).items()},
            **{f"val_{k}": v for k, v in _compact_metrics(val_metrics).items()},
        }
        history.append(row)

        train_score = float(train_metrics["balanced_accuracy"])
        val_score = float(val_metrics["balanced_accuracy"])
        if train_score > best_train_score:
            best_train_score = train_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        best_val_score = max(best_val_score, val_score)

        print(
            f"{model_name:6s} seed={seed} epoch={epoch:03d}/{args.epochs} "
            f"loss={row['loss']:.4f} "
            f"train_bal={train_score:.3f} val_bal={val_score:.3f} "
            f"pos={train_metrics['pred_positive_rate']:.3f} "
            f"grad={row['all_grad_mean']:.2e} "
            f"update={update_norm:.2e}"
        )

        if train_score >= args.success_train_balanced_accuracy:
            successful_epochs += 1
        else:
            successful_epochs = 0
        if (
            args.stop_after_success > 0
            and successful_epochs >= args.stop_after_success
        ):
            print(
                f"{model_name} seed={seed}: trainability target sustained for "
                f"{args.stop_after_success} epochs; stopping early."
            )
            break

    checkpoint_path = output_dir / f"{model_name}_seed{seed}_best_train.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model": model_name,
            "seed": seed,
            "parameter_count": parameter_count,
            "best_train_balanced_accuracy": best_train_score,
            "best_val_balanced_accuracy": best_val_score,
            "config": vars(args),
        },
        checkpoint_path,
    )

    return {
        "model": model_name,
        "seed": seed,
        "parameter_count": parameter_count,
        "epochs_completed": len(history),
        "best_train_balanced_accuracy": best_train_score,
        "best_val_balanced_accuracy": best_val_score,
        "final": history[-1],
        "history": history,
        "checkpoint": str(checkpoint_path),
        "runtime_s": time.time() - started,
    }


def write_epoch_csv(results: list[dict], path: Path):
    rows = [row for result in results for row in result["history"]]
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_decision(results: list[dict], args) -> dict:
    by_model_seed = {
        (result["model"], result["seed"]): result
        for result in results
    }
    rows = []
    for seed in args.seeds:
        vqc = by_model_seed.get(("vqc", seed))
        mlp = by_model_seed.get(("mlp", seed))
        if vqc is None:
            continue
        vqc_score = vqc["best_train_balanced_accuracy"]
        mlp_score = mlp["best_train_balanced_accuracy"] if mlp else None
        reaches_absolute = vqc_score >= args.success_train_balanced_accuracy
        mlp_demonstrates_learnability = (
            mlp_score is not None
            and mlp_score >= args.success_train_balanced_accuracy
        )
        close_to_mlp = (
            mlp_demonstrates_learnability
            and vqc_score >= mlp_score - args.mlp_tolerance
        )
        rows.append(
            {
                "seed": seed,
                "vqc_best_train_balanced_accuracy": vqc_score,
                "mlp_best_train_balanced_accuracy": mlp_score,
                "mlp_demonstrates_learnability": mlp_demonstrates_learnability,
                "reaches_absolute_target": reaches_absolute,
                "within_mlp_tolerance": close_to_mlp,
                "passes": reaches_absolute or close_to_mlp,
            }
        )

    required = math.ceil(len(rows) / 2) if rows else 1
    passes = sum(bool(row["passes"]) for row in rows)
    return {
        "criterion": (
            f"VQC train balanced accuracy >= "
            f"{args.success_train_balanced_accuracy:.2f}, or within "
            f"{args.mlp_tolerance:.2f} of matched MLP"
        ),
        "per_seed": rows,
        "passing_seeds": passes,
        "required_passing_seeds": required,
        "proceed_to_full_benchmark": bool(rows) and passes >= required,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train linear, MLP, and VQC heads on identical balanced cached "
            "histopathology features."
        )
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/histopath/vqc_trainability"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--train-per-class", type=int, default=256)
    parser.add_argument("--val-per-class", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--entanglement",
        choices=["none", "linear", "circular"],
        default="linear",
    )
    parser.add_argument(
        "--encoding",
        choices=["angle_x", "angle_y"],
        default="angle_y",
    )
    parser.add_argument("--data-reuploading", action="store_true")
    parser.add_argument("--backend", default="default.qubit")
    parser.add_argument("--diff-method", default="backprop")
    parser.add_argument(
        "--success-train-balanced-accuracy",
        type=float,
        default=0.95,
    )
    parser.add_argument("--mlp-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--stop-after-success",
        type=int,
        default=3,
        help="Stop after the trainability target is sustained for N epochs; 0 disables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(
        args.feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    train_features, train_labels, train_indices = subset_from_cache(
        cache["train"],
        per_class=args.train_per_class,
        seed=args.sample_seed,
    )
    val_features, val_labels, val_indices = subset_from_cache(
        cache["val"],
        per_class=args.val_per_class,
        seed=args.sample_seed + 1,
    )

    manifest = {
        "feature_cache": str(args.feature_cache),
        "sample_seed": args.sample_seed,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "train_class_counts": torch.bincount(
            train_labels, minlength=2
        ).tolist(),
        "val_class_counts": torch.bincount(val_labels, minlength=2).tolist(),
        "feature_dim": int(train_features.shape[1]),
    }
    with open(args.output_dir / "sample_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    print("Fixed-feature VQC trainability diagnostic")
    print(f"  cache={args.feature_cache}")
    print(
        f"  train={len(train_labels)} {manifest['train_class_counts']} "
        f"val={len(val_labels)} {manifest['val_class_counts']} "
        f"features={manifest['feature_dim']}"
    )
    print(f"  models={args.models} seeds={args.seeds} epochs={args.epochs}")

    results = []
    for model_name in args.models:
        for seed in args.seeds:
            results.append(
                train_one(
                    model_name,
                    seed,
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    args,
                    args.output_dir,
                )
            )

    decision = build_decision(results, args)
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary = {
        "config": config_payload,
        "manifest": manifest,
        "runs": [
            {key: value for key, value in result.items() if key != "history"}
            for result in results
        ],
        "decision": decision,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    write_epoch_csv(results, args.output_dir / "epoch_metrics.csv")

    print("\nDecision:")
    print(json.dumps(decision, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
