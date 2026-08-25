#!/usr/bin/env python3
"""Average paired Paper C prediction files before case-level inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, action="append", type=Path)
    parser.add_argument("--seed", required=True, action="append", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.prediction) != len(args.seed) or len(args.seed) < 2:
        raise SystemExit("Supply matching --prediction/--seed pairs.")
    if len(set(args.seed)) != len(args.seed):
        raise SystemExit("Seed values must be unique.")

    raw_frames = [
        pd.read_csv(path, dtype={"filepath": str, "case_id": str})
        for path in args.prediction
    ]
    keys = ["filepath", "case_id", "label"]
    for optional in ("site", "roi_id"):
        present = [optional in frame for frame in raw_frames]
        if any(present) and not all(present):
            raise SystemExit(f"Optional metadata {optional} is missing from some seeds.")
        if all(present):
            keys.append(optional)
    frames = []
    for path, seed, frame in zip(args.prediction, args.seed, raw_frames):
        required = set(keys) | {"probability", "logit", "threshold"}
        if not required <= set(frame):
            raise SystemExit(f"{path} lacks {sorted(required - set(frame))}")
        if frame["filepath"].duplicated().any():
            raise SystemExit(f"{path} has duplicate filepath keys.")
        columns = keys + ["probability", "logit", "threshold"]
        if "fold" in frame:
            columns.append("fold")
        frames.append(frame[columns].rename(columns={
                "probability": f"probability_seed_{seed}",
                "logit": f"logit_seed_{seed}",
                "threshold": f"threshold_seed_{seed}",
                "fold": f"fold_seed_{seed}",
            }))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    if any(len(frame) != len(merged) for frame in frames):
        raise SystemExit("Seed prediction filepath populations differ.")
    fold_columns = [column for column in merged if column.startswith("fold_seed_")]
    if fold_columns and len(fold_columns) != len(frames):
        raise SystemExit("Fold metadata is present for only some seed files.")
    if fold_columns and not merged[fold_columns].nunique(axis=1).eq(1).all():
        raise SystemExit("Seed prediction fold assignments differ.")

    probability_columns = [
        column for column in merged if column.startswith("probability_seed_")
    ]
    logit_columns = [column for column in merged if column.startswith("logit_seed_")]
    threshold_columns = [
        column for column in merged if column.startswith("threshold_seed_")
    ]
    output = merged[keys].copy()
    if fold_columns:
        output["fold"] = merged[fold_columns[0]]
    output["logit"] = merged[logit_columns].mean(axis=1)
    output["probability"] = merged[probability_columns].mean(axis=1)
    output["threshold"] = merged[threshold_columns].mean(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    output.to_csv(args.output, index=False)
    report = {
        "seeds": args.seed,
        "rows": len(output),
        "aggregation": (
            "arithmetic mean of calibrated probabilities, logits and "
            "validation-locked fold thresholds"
        ),
        "inputs": {
            str(seed): {"path": str(path), "sha256": sha256(path)}
            for seed, path in zip(args.seed, args.prediction)
        },
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
