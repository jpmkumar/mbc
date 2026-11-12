#!/usr/bin/env python3
"""Paper C+ BreakHis linear SGD probe (Paper C bridge secondary)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT / "papers" / "paper_c" / "scripts"))
from run_idc_probe_cv import (  # noqa: E402
    ALPHAS,
    case_weights,
    choose_threshold,
    fit_temperature,
    metric_bundle,
    partition_random_inner,
    sha256,
)


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
        raise RuntimeError(f"fold {outer_fold}: inner case lists overlap")
    train = frame.index[frame["case_id"].isin(train_ids)].to_numpy()
    val = frame.index[frame["case_id"].isin(val_ids)].to_numpy()
    cal = frame.index[frame["case_id"].isin(cal_ids)].to_numpy()
    return train, val, cal


def weighted_ap(y: np.ndarray, p: np.ndarray, cases: np.ndarray) -> float:
    return float(average_precision_score(y, p, sample_weight=case_weights(cases)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=("grouped", "random"))
    parser.add_argument(
        "--weighting",
        choices=("case-balanced", "protocol-native"),
        default="case-balanced",
    )
    parser.add_argument(
        "--inner-case-splits",
        type=Path,
        default=ROOT / "data" / "splits" / "paper_c_plus" / "breakhis",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    provenance = json.loads((args.cache / "provenance.json").read_text())
    if not provenance.get("complete") or provenance.get("cohort") != "BreakHis":
        raise SystemExit("BreakHis embedding cache incomplete or misidentified")
    embeddings = np.load(args.cache / "embeddings.npy", mmap_mode="r")
    cache_index = pd.read_csv(args.cache / "index.csv", dtype={"filepath": str, "case_id": str})
    manifest = pd.read_csv(args.manifest, dtype={"filepath": str, "case_id": str})
    if len(cache_index) != len(embeddings) or len(cache_index) != len(manifest):
        raise SystemExit("cache/manifest row counts differ")
    if set(cache_index["filepath"]) != set(manifest["filepath"]):
        raise SystemExit("cache and manifest filepath populations differ")
    if sha256(args.cache / "index.csv") != provenance["index_sha256"]:
        raise SystemExit("index sha mismatch")
    if sha256(args.cache / "embeddings.npy") != provenance["embeddings_sha256"]:
        raise SystemExit("embeddings sha mismatch")

    need = [
        "filepath",
        "case_id",
        "y",
        "magnification",
        "grouped_outer_fold",
        "random_outer_fold",
    ]
    frame = cache_index[["filepath"]].merge(manifest[need], on="filepath", how="left")
    if frame[need].isna().any().any():
        raise SystemExit("manifest join incomplete")
    if list(frame["filepath"]) != list(cache_index["filepath"]):
        raise SystemExit("cache index order must match manifest join order")
    frame["_embedding_row"] = np.arange(len(frame))
    frame["label"] = frame["y"].astype(int)
    frame = frame.reset_index(drop=True)

    fold_column = (
        "grouped_outer_fold" if args.protocol == "grouped" else "random_outer_fold"
    )
    case_weighted = args.weighting == "case-balanced" or args.protocol == "grouped"
    oof_parts: list[pd.DataFrame] = []
    fold_reports: list[dict] = []

    for fold in range(5):
        outer_test = frame.index[frame[fold_column] == fold].to_numpy()
        development = frame[frame[fold_column] != fold]
        if args.protocol == "grouped":
            train, val, cal = grouped_inner(frame, args.inner_case_splits, fold)
            allowed = set(development.index)
            if (set(train) | set(val) | set(cal)) != allowed:
                raise RuntimeError(f"fold {fold}: inner sets do not cover development")
        else:
            train, val, cal = partition_random_inner(development, fold)

        y_train = frame.loc[train, "label"].to_numpy()
        y_val = frame.loc[val, "label"].to_numpy()
        y_cal = frame.loc[cal, "label"].to_numpy()
        row_train = frame.loc[train, "_embedding_row"].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            np.asarray(embeddings[row_train], dtype=np.float32)
        ).astype(np.float32)
        x_val = scaler.transform(
            np.asarray(
                embeddings[frame.loc[val, "_embedding_row"].to_numpy()], dtype=np.float32
            )
        ).astype(np.float32)
        x_cal = scaler.transform(
            np.asarray(
                embeddings[frame.loc[cal, "_embedding_row"].to_numpy()], dtype=np.float32
            )
        ).astype(np.float32)
        x_test = scaler.transform(
            np.asarray(
                embeddings[frame.loc[outer_test, "_embedding_row"].to_numpy()],
                dtype=np.float32,
            )
        ).astype(np.float32)

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
                    y_val, val_probability, frame.loc[val, "case_id"].to_numpy()
                )
                if case_weighted
                else float(average_precision_score(y_val, val_probability))
            )
            tuning.append((objective, alpha, classifier))
        _, selected_alpha, model = max(tuning, key=lambda item: (item[0], -item[1]))

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
            case_weights(frame.loc[val, "case_id"].to_numpy()) if case_weighted else None,
        )
        test_logits = model.decision_function(x_test)
        test_probability = expit(test_logits / temperature)
        test_frame = frame.loc[
            outer_test, ["filepath", "case_id", "label", "magnification"]
        ].copy()
        test_frame["fold"] = fold
        test_frame["logit"] = test_logits
        test_frame["probability"] = test_probability
        test_frame["threshold"] = threshold
        oof_parts.append(test_frame)
        fold_reports.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_val": len(val),
                "n_cal": len(cal),
                "n_test": len(outer_test),
                "selected_alpha": selected_alpha,
                "temperature": temperature,
                "threshold": threshold,
                "test_metrics": metric_bundle(test_frame),
            }
        )
        print(json.dumps(fold_reports[-1]), flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != len(frame) or oof["filepath"].duplicated().any():
        raise RuntimeError("OOF predictions incomplete")
    oof = oof.sort_values("filepath").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / "oof_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    if pred_path.exists() or summary_path.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    oof.to_csv(pred_path, index=False)
    summary = {
        "complete": True,
        "paper": "C+",
        "head": "linear-SGD",
        "protocol": args.protocol,
        "weighting": args.weighting,
        "cache": str(args.cache),
        "seed": args.seed,
        "alpha_grid": list(ALPHAS),
        "rows": len(oof),
        "case_identifiers": int(oof["case_id"].nunique()),
        "metrics": metric_bundle(oof),
        "folds": fold_reports,
        "predictions_sha256": sha256(pred_path),
        "cache_provenance_sha256": sha256(args.cache / "provenance.json"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"metrics": summary["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
