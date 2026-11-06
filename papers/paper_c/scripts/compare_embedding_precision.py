#!/usr/bin/env python3
"""Fail unless fp16-autocast and fp32-compute caches are probe-equivalent.

Raw absolute/relative maxima over unnormalised features are dominated by
near-zero units (especially ReLU ResNet dims). Confirmatory gates therefore
use row cosine similarity plus absolute error after L2 normalisation — the
geometry that a linear probe sees after standardisation.
"""

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
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--min-cosine-mean", type=float, default=0.9999)
    parser.add_argument("--max-l2-abs-error", type=float, default=0.02)
    args = parser.parse_args()

    left_paths = filepaths(args.fp16_cache / "index.csv")
    right_paths = filepaths(args.fp32_cache / "index.csv")
    if left_paths != right_paths:
        raise SystemExit("Cache filepath indexes differ; comparison would be invalid.")
    left = np.load(args.fp16_cache / "embeddings.npy", mmap_mode="r").astype(np.float32)
    right = np.load(args.fp32_cache / "embeddings.npy", mmap_mode="r").astype(np.float32)
    if left.shape != right.shape:
        raise SystemExit(f"Shape mismatch: {left.shape} versus {right.shape}")

    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    cosine = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    left_unit = left / np.clip(left_norm[:, None], 1e-12, None)
    right_unit = right / np.clip(right_norm[:, None], 1e-12, None)
    l2_abs = np.abs(left_unit - right_unit)
    absolute = np.abs(left - right)
    report = {
        "rows": len(left_paths),
        "dimensions": int(left.shape[1]),
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "cosine_p01": float(np.quantile(cosine, 0.01)),
        "absolute_error_max": float(absolute.max()),
        "absolute_error_mean": float(absolute.mean()),
        "l2_normalized_abs_max": float(l2_abs.max()),
        "l2_normalized_abs_mean": float(l2_abs.mean()),
        "rows_cosine_below_0_999": int((cosine < 0.999).sum()),
        "thresholds": {
            "min_cosine": args.min_cosine,
            "min_cosine_mean": args.min_cosine_mean,
            "max_l2_abs_error": args.max_l2_abs_error,
        },
    }
    report["status"] = (
        "PASS"
        if report["cosine_min"] >= args.min_cosine
        and report["cosine_mean"] >= args.min_cosine_mean
        and report["l2_normalized_abs_max"] <= args.max_l2_abs_error
        else "FAIL"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Embedding precision equivalence failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
