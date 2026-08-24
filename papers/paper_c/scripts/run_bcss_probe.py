#!/usr/bin/env python3
"""Fit and evaluate Paper C's frozen linear probe on BCSS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from run_idc_probe_cv import (
    ALPHAS,
    case_weights,
    choose_threshold,
    fit_temperature,
    metric_bundle,
    sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--centre-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    provenance = json.loads((args.cache / "provenance.json").read_text())
    if not provenance.get("complete") or provenance.get("cohort") != "BCSS":
        raise SystemExit("BCSS embedding cache is incomplete or misidentified.")
    embeddings = np.load(args.cache / "embeddings.npy", mmap_mode="r")
    index = pd.read_csv(args.cache / "index.csv", dtype={"filepath": str})
    manifest = pd.read_csv(
        args.centre_manifest,
        dtype={"filepath": str, "patient_id": str},
    )
    if len(index) != len(embeddings):
        raise SystemExit("Cache index and embedding rows differ.")
    if sha256(args.centre_manifest) != provenance["manifest_sha256"]:
        raise SystemExit("BCSS centre manifest differs from extraction provenance.")
    if list(embeddings.shape) != [
        int(provenance["n_centres"]),
        int(provenance["embedding_dim"]),
    ]:
        raise SystemExit("BCSS embedding shape differs from provenance.")
    if np.dtype(embeddings.dtype).name != provenance["dtype"]:
        raise SystemExit("BCSS embedding dtype differs from provenance.")
    if sha256(args.cache / "index.csv") != provenance["index_sha256"]:
        raise SystemExit("BCSS index SHA-256 differs from provenance.")
    if sha256(args.cache / "embeddings.npy") != provenance["embeddings_sha256"]:
        raise SystemExit("BCSS embedding SHA-256 differs from provenance.")
    if index["filepath"].duplicated().any() or manifest["filepath"].duplicated().any():
        raise SystemExit("BCSS filepath key is not unique.")
    if len(index) != len(manifest) or set(index["filepath"]) != set(
        manifest["filepath"]
    ):
        raise SystemExit("BCSS cache and centre-manifest populations differ.")
    index["_embedding_row"] = np.arange(len(index))
    duplicate_metadata = [
        "patient_id", "site", "split", "x", "y", "label", "roi_id",
        "image_filename", "mask_filename",
    ]
    frame = index[["filepath", "_embedding_row", *duplicate_metadata]].merge(
        manifest,
        on="filepath",
        validate="one_to_one",
        how="left",
        indicator=True,
    )
    if not (frame["_merge"] == "both").all() or len(frame) != len(manifest):
        raise SystemExit("BCSS cache-to-manifest filepath join is incomplete.")
    for column in duplicate_metadata:
        left = frame[f"{column}_x"].astype(str)
        right = frame[f"{column}_y"].astype(str)
        if not (left == right).all():
            raise SystemExit(f"BCSS cached metadata differs for {column}.")
        frame[column] = frame[f"{column}_y"]
    frame = frame.drop(columns=[
        "_merge",
        *[f"{column}_{side}" for column in duplicate_metadata for side in ("x", "y")],
    ])

    partitions = {
        name: frame.index[frame["split"] == name].to_numpy()
        for name in ("train", "val", "cal", "test")
    }
    if any(len(indices) == 0 for indices in partitions.values()):
        raise SystemExit("Every BCSS train/val/cal/test partition must be nonempty.")
    patient_sets = {
        name: set(frame.loc[indices, "patient_id"])
        for name, indices in partitions.items()
    }
    names = list(patient_sets)
    if any(
        patient_sets[names[i]] & patient_sets[names[j]]
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ):
        raise SystemExit("BCSS patient partitions overlap.")
    test_sites = set(frame.loc[partitions["test"], "site"])
    development_sites = set(frame.loc[
        np.concatenate([partitions["train"], partitions["val"], partitions["cal"]]),
        "site",
    ])
    if test_sites & development_sites:
        raise SystemExit("BCSS test institutions leak into development.")

    scaler = StandardScaler()
    train = partitions["train"]
    val = partitions["val"]
    cal = partitions["cal"]
    test = partitions["test"]
    x_train = scaler.fit_transform(
        np.asarray(
            embeddings[frame.loc[train, "_embedding_row"].to_numpy()],
            dtype=np.float32,
        )
    ).astype(np.float32, copy=False)
    x_val = scaler.transform(
        np.asarray(
            embeddings[frame.loc[val, "_embedding_row"].to_numpy()],
            dtype=np.float32,
        )
    ).astype(np.float32, copy=False)
    y_train = frame.loc[train, "label"].to_numpy()
    y_val = frame.loc[val, "label"].to_numpy()
    train_weights = case_weights(frame.loc[train, "patient_id"].to_numpy())

    candidates: list[tuple[float, float, SGDClassifier]] = []
    for alpha in ALPHAS:
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=args.max_iter,
            tol=1e-4,
            random_state=args.seed,
            shuffle=True,
        )
        model.fit(x_train, y_train, sample_weight=train_weights)
        probability = model.predict_proba(x_val)[:, 1]
        objective = average_precision_score(
            y_val,
            probability,
            sample_weight=case_weights(frame.loc[val, "patient_id"].to_numpy()),
        )
        candidates.append((float(objective), alpha, model))
    _, selected_alpha, model = max(candidates, key=lambda item: (item[0], -item[1]))

    x_cal = scaler.transform(
        np.asarray(
            embeddings[frame.loc[cal, "_embedding_row"].to_numpy()],
            dtype=np.float32,
        )
    ).astype(np.float32, copy=False)
    y_cal = frame.loc[cal, "label"].to_numpy()
    temperature = fit_temperature(
        model.decision_function(x_cal),
        y_cal,
        case_weights(frame.loc[cal, "patient_id"].to_numpy()),
    )
    val_probability = expit(model.decision_function(x_val) / temperature)
    threshold = choose_threshold(y_val, val_probability)

    x_test = scaler.transform(
        np.asarray(
            embeddings[frame.loc[test, "_embedding_row"].to_numpy()],
            dtype=np.float32,
        )
    ).astype(np.float32, copy=False)
    logits = model.decision_function(x_test)
    predictions = frame.loc[test, ["filepath", "patient_id", "label", "site", "roi_id"]].copy()
    predictions = predictions.rename(columns={"patient_id": "case_id"})
    predictions["logit"] = logits
    predictions["probability"] = expit(logits / temperature)
    predictions["threshold"] = threshold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "test_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise SystemExit(f"Output bundle exists: {args.output_dir}")
    predictions.to_csv(prediction_path, index=False)
    summary = {
        "complete": True,
        "cohort": "BCSS",
        "cache_provenance_sha256": sha256(args.cache / "provenance.json"),
        "centre_manifest_sha256": sha256(args.centre_manifest),
        "seed": args.seed,
        "selected_alpha": selected_alpha,
        "temperature": temperature,
        "threshold": threshold,
        "split_rows": {name: len(indices) for name, indices in partitions.items()},
        "split_patients": {
            name: len(patients) for name, patients in patient_sets.items()
        },
        "test_sites": sorted(test_sites),
        "test_metrics": metric_bundle(predictions),
        "predictions_sha256": sha256(prediction_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
