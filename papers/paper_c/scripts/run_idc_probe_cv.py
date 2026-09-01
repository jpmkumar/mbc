#!/usr/bin/env python3
"""Run Paper C's deterministic linear-probe IDC cross-validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ALPHAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, filepath: str) -> str:
    return hashlib.sha256(f"{seed}|{filepath}".encode()).hexdigest()


def case_weights(case_ids: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(case_ids, return_counts=True)
    inverse = {case_id: 1.0 / count for case_id, count in zip(unique, counts)}
    weights = np.asarray([inverse[case_id] for case_id in case_ids], dtype=np.float64)
    return weights * len(weights) / weights.sum()


def partition_random_inner(
    frame: pd.DataFrame, outer_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train: list[int] = []
    val: list[int] = []
    cal: list[int] = []
    seed = 2042 + outer_fold
    for label in (0, 1):
        members = frame.index[frame["label"] == label].tolist()
        members.sort(key=lambda idx: stable_key(seed, frame.at[idx, "filepath"]))
        n = len(members)
        n_train = round(0.75 * n)
        n_val = round(0.125 * n)
        train.extend(members[:n_train])
        val.extend(members[n_train:n_train + n_val])
        cal.extend(members[n_train + n_val:])
    return np.asarray(train), np.asarray(val), np.asarray(cal)


def read_case_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["case_id"] for row in csv.DictReader(stream)}


def grouped_inner(
    frame: pd.DataFrame, inner_root: Path, outer_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_dir = inner_root / "folds" / f"fold_{outer_fold}"
    train_ids = read_case_ids(fold_dir / "inner_train_case_ids.csv")
    val_ids = read_case_ids(fold_dir / "inner_val_case_ids.csv")
    cal_ids = read_case_ids(fold_dir / "inner_cal_case_ids.csv")
    sets = [train_ids, val_ids, cal_ids]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"Fold {outer_fold}: inner case lists overlap.")
    train = frame.index[frame["case_id"].isin(train_ids)].to_numpy()
    val = frame.index[frame["case_id"].isin(val_ids)].to_numpy()
    cal = frame.index[frame["case_id"].isin(cal_ids)].to_numpy()
    return train, val, cal


def weighted_ap(y: np.ndarray, p: np.ndarray, cases: np.ndarray) -> float:
    return float(average_precision_score(y, p, sample_weight=case_weights(cases)))


def choose_threshold(
    y: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    candidates = np.unique(probabilities)
    if len(candidates) > 5_000:
        candidates = np.quantile(probabilities, np.linspace(0, 1, 5_001))
    best_threshold = 0.5
    best_mcc = -math.inf
    for threshold in candidates:
        mcc = matthews_corrcoef(y, probabilities >= threshold, sample_weight=weights)
        if mcc > best_mcc or (mcc == best_mcc and threshold > best_threshold):
            best_mcc = mcc
            best_threshold = float(threshold)
    return best_threshold


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> float:
    def objective(log_temperature: float) -> float:
        probabilities = expit(logits / math.exp(log_temperature))
        return float(log_loss(y, probabilities, sample_weight=weights, labels=[0, 1]))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(10.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature scaling failed: {result.message}")
    return float(math.exp(result.x))


def metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["label"].to_numpy()
    p = frame["probability"].to_numpy()
    weights = case_weights(frame["case_id"].to_numpy())
    predicted = p >= frame["threshold"].to_numpy()
    true_positive = int(np.sum(predicted & (y == 1)))
    false_negative = int(np.sum(~predicted & (y == 1)))
    true_negative = int(np.sum(~predicted & (y == 0)))
    false_positive = int(np.sum(predicted & (y == 0)))
    return {
        "case_balanced_auprc": float(average_precision_score(y, p, sample_weight=weights)),
        "unweighted_auprc": float(average_precision_score(y, p)),
        "case_balanced_auroc": float(roc_auc_score(y, p, sample_weight=weights)),
        "unweighted_auroc": float(roc_auc_score(y, p)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "sensitivity": true_positive / (true_positive + false_negative),
        "specificity": true_negative / (true_negative + false_positive),
        "brier_case_balanced": float(brier_score_loss(y, p, sample_weight=weights)),
        "log_loss_case_balanced": float(log_loss(y, p, sample_weight=weights)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--patch-manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=("grouped", "random"))
    parser.add_argument(
        "--weighting",
        choices=("case-balanced", "protocol-native"),
        default="case-balanced",
        help=(
            "case-balanced harmonises fitting, tuning, calibration and threshold "
            "weighting across both arms so the protocol contrast isolates grouping. "
            "protocol-native keeps each arm's conventional weighting and yields the "
            "preregistered bundled-regime secondary contrast."
        ),
    )
    parser.add_argument(
        "--inner-case-splits",
        type=Path,
        default=Path("data/splits/paper_c/idc"),
    )
    parser.add_argument(
        "--inner-split-summary",
        type=Path,
        default=Path("data/splits/paper_c/idc/inner_split_summary.json"),
    )
    parser.add_argument("--complete-context-k", choices=("3", "5", "9"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    if args.protocol == "grouped" and "paper_c/idc" not in str(args.inner_case_splits):
        raise SystemExit(
            "Grouped Paper C probes require data/splits/paper_c/idc; "
            "the Paper B histopath partition is forbidden."
        )
    provenance = json.loads((args.cache / "provenance.json").read_text())
    if not provenance.get("complete"):
        raise SystemExit(f"Embedding cache is incomplete: {args.cache}")
    if args.protocol == "random" and provenance.get("context_k") != 1:
        raise SystemExit(
            "Random-patch evaluation is restricted to K=1; mosaics create an "
            "additional direct neighbour-leakage channel."
        )
    if args.protocol == "random" and args.complete_context_k:
        raise SystemExit("Context-eligible analyses must use the grouped protocol.")
    embeddings = np.load(args.cache / "embeddings.npy", mmap_mode="r")
    cache_index = pd.read_csv(args.cache / "index.csv")
    if len(cache_index) != len(embeddings):
        raise SystemExit("Cache index and embedding row counts differ.")
    if list(embeddings.shape) != [
        int(provenance["n_centres"]),
        int(provenance["embedding_dim"]),
    ]:
        raise SystemExit("Embedding shape differs from cache provenance.")
    if np.dtype(embeddings.dtype).name != provenance["dtype"]:
        raise SystemExit("Embedding dtype differs from cache provenance.")
    if sha256(args.cache / "index.csv") != provenance["index_sha256"]:
        raise SystemExit("Cache index SHA-256 differs from provenance.")
    if sha256(args.cache / "embeddings.npy") != provenance["embeddings_sha256"]:
        raise SystemExit("Embedding SHA-256 differs from provenance.")
    cache_index["_embedding_row"] = np.arange(len(cache_index))
    manifest = pd.read_csv(
        args.patch_manifest,
        dtype={"filepath": str, "case_id": str},
    )
    if manifest["filepath"].duplicated().any() or cache_index["filepath"].duplicated().any():
        raise SystemExit("filepath must be unique in both cache and manifest.")
    if len(manifest) != len(cache_index) or set(manifest["filepath"]) != set(
        cache_index["filepath"]
    ):
        raise SystemExit("Cache and patch manifest filepath populations differ.")
    frame = cache_index.merge(
        manifest,
        on="filepath",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_cache", ""),
    )
    if not (frame["_merge"] == "both").all():
        raise SystemExit("Cache-to-manifest filepath join is incomplete.")
    if not (
        frame["case_id_cache"].astype(str) == frame["case_id"].astype(str)
    ).all() or not (frame["label_cache"] == frame["label"]).all():
        raise SystemExit("Cache and manifest case/label metadata disagree.")
    frame = frame.drop(columns=["_merge", "case_id_cache", "label_cache"])

    if args.complete_context_k:
        column = f"k{args.complete_context_k}_complete"
        if column not in frame:
            raise SystemExit(f"Manifest does not contain {column}.")
        frame = frame[frame[column] == 1].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise SystemExit("No observations remain after eligibility filtering.")

    fold_column = (
        "grouped_outer_fold" if args.protocol == "grouped" else "random_outer_fold"
    )
    case_weighted = args.weighting == "case-balanced" or args.protocol == "grouped"
    oof_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []

    for fold in range(5):
        outer_test = frame.index[frame[fold_column] == fold].to_numpy()
        development = frame[frame[fold_column] != fold]
        if args.protocol == "grouped":
            train, val, cal = grouped_inner(frame, args.inner_case_splits, fold)
            allowed = set(development.index)
            if (set(train) | set(val) | set(cal)) != allowed:
                raise RuntimeError(f"Fold {fold}: grouped inner sets do not cover development.")
        else:
            train, val, cal = partition_random_inner(development, fold)

        y_train = frame.loc[train, "label"].to_numpy()
        y_val = frame.loc[val, "label"].to_numpy()
        y_cal = frame.loc[cal, "label"].to_numpy()
        row_train = frame.loc[train, "_embedding_row"].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            np.asarray(embeddings[row_train], dtype=np.float32)
        ).astype(np.float32, copy=False)
        x_val = scaler.transform(
            np.asarray(
                embeddings[frame.loc[val, "_embedding_row"].to_numpy()],
                dtype=np.float32,
            )
        ).astype(np.float32, copy=False)

        if case_weighted:
            fit_weights = case_weights(frame.loc[train, "case_id"].to_numpy())
        else:
            fit_weights = compute_sample_weight("balanced", y_train)
        tuning: list[tuple[float, float, SGDClassifier]] = []
        for alpha in ALPHAS:
            classifier = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=args.max_iter,
                tol=1e-4,
                random_state=args.seed + fold,
                shuffle=True,
            )
            classifier.fit(x_train, y_train, sample_weight=fit_weights)
            val_probability = classifier.predict_proba(x_val)[:, 1]
            objective = (
                weighted_ap(
                    y_val,
                    val_probability,
                    frame.loc[val, "case_id"].to_numpy(),
                )
                if case_weighted
                else float(average_precision_score(y_val, val_probability))
            )
            tuning.append((objective, alpha, classifier))
        _, selected_alpha, model = max(tuning, key=lambda item: (item[0], -item[1]))

        x_cal = scaler.transform(
            np.asarray(
                embeddings[frame.loc[cal, "_embedding_row"].to_numpy()],
                dtype=np.float32,
            )
        ).astype(np.float32, copy=False)
        cal_logits = model.decision_function(x_cal)
        cal_weights = (
            case_weights(frame.loc[cal, "case_id"].to_numpy())
            if case_weighted
            else np.ones(len(cal))
        )
        temperature = fit_temperature(cal_logits, y_cal, cal_weights)
        val_probability = expit(model.decision_function(x_val) / temperature)
        threshold = choose_threshold(
            y_val,
            val_probability,
            case_weights(frame.loc[val, "case_id"].to_numpy())
            if case_weighted
            else None,
        )

        x_test = scaler.transform(
            np.asarray(
                embeddings[frame.loc[outer_test, "_embedding_row"].to_numpy()],
                dtype=np.float32,
            )
        ).astype(np.float32, copy=False)
        test_logits = model.decision_function(x_test)
        test_probability = expit(test_logits / temperature)
        test_frame = frame.loc[
            outer_test, ["filepath", "case_id", "label"]
        ].copy()
        test_frame["fold"] = fold
        test_frame["logit"] = test_logits
        test_frame["probability"] = test_probability
        test_frame["threshold"] = threshold
        oof_parts.append(test_frame)
        fold_reports.append({
            "fold": fold,
            "n_train": len(train),
            "n_val": len(val),
            "n_cal": len(cal),
            "n_test": len(outer_test),
            "selected_alpha": selected_alpha,
            "temperature": temperature,
            "threshold": threshold,
            "test_metrics": metric_bundle(test_frame),
        })
        print(json.dumps(fold_reports[-1]), flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != len(frame) or oof["filepath"].duplicated().any():
        raise RuntimeError("OOF predictions are not one-to-one and complete.")
    if set(oof["filepath"]) != set(frame["filepath"]):
        raise RuntimeError("OOF predictions do not cover the selected filepath population.")
    oof = oof.sort_values("filepath").reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "oof_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise SystemExit(f"Output bundle already exists: {args.output_dir}")
    oof.to_csv(prediction_path, index=False)
    summary = {
        "complete": True,
        "protocol": args.protocol,
        "weighting": args.weighting,
        "case_weighted_fitting": case_weighted,
        "cache": str(args.cache),
        "cache_provenance_sha256": sha256(args.cache / "provenance.json"),
        "cache_index_sha256": sha256(args.cache / "index.csv"),
        "patch_manifest_sha256": sha256(args.patch_manifest),
        "inner_split_summary_sha256": (
            sha256(args.inner_split_summary)
            if args.protocol == "grouped"
            else None
        ),
        "complete_context_k": args.complete_context_k,
        "seed": args.seed,
        "alpha_grid": list(ALPHAS),
        "rows": len(oof),
        "case_identifiers": int(oof["case_id"].nunique()),
        "metrics": metric_bundle(oof),
        "folds": fold_reports,
        "predictions_sha256": sha256(prediction_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
