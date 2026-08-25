#!/usr/bin/env python3
"""Fail unless fp16-autocast and fp32-compute caches are equivalent."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def filepaths(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [row["filepath"] for row in csv.DictReader(stream)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-cache", required=True, type=Path)
    parser.add_argument("--fp32-cache", required=True, type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-absolute-error", type=float, default=0.02)
    args = parser.parse_args()

    left_paths = filepaths(args.fp16_cache / "index.csv")
    right_paths = filepaths(args.fp32_cache / "index.csv")
    if left_paths != right_paths:
        raise SystemExit("Cache filepath indexes differ; comparison would be invalid.")
    left = np.load(args.fp16_cache / "embeddings.npy", mmap_mode="r").astype(np.float32)
    right = np.load(args.fp32_cache / "embeddings.npy", mmap_mode="r").astype(np.float32)
    if left.shape != right.shape:
        raise SystemExit(f"Shape mismatch: {left.shape} versus {right.shape}")

    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    absolute = np.abs(left - right)
    report = {
        "rows": len(left_paths),
        "dimensions": left.shape[1],
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "absolute_error_max": float(absolute.max()),
        "absolute_error_mean": float(absolute.mean()),
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_absolute_error": args.max_absolute_error,
        },
    }
    report["status"] = (
        "PASS"
        if report["cosine_min"] >= args.min_cosine
        and report["absolute_error_max"] <= args.max_absolute_error
        else "FAIL"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Embedding precision equivalence failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
