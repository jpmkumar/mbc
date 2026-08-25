#!/usr/bin/env python3
"""Freeze Paper C's inner train/validation/calibration IDC case-ID lists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_ids(path: Path, ids: list[str]) -> None:
    # LF only: these artifacts are committed and their SHA-256 digests are
    # preregistered, so CRLF would not survive git's line-ending normalization.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id"])
        writer.writerows((case_id,) for case_id in ids)


def split_groups(
    stats_by_id: dict[str, dict[str, str]],
    group_ids: list[str],
    *,
    val_ratio: float,
    cal_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Stdlib mirror of ``src.data.histopath_splits.split_train_val_cal_groups``.

    This generator writes frozen provenance artifacts, so it deliberately avoids
    pandas and scikit-learn. ``tests/test_paper_c_splits.py`` asserts that this
    implementation and the library agree exactly.
    """
    by_bin: dict[int, list[str]] = {}
    for group_id in group_ids:
        if group_id not in stats_by_id:
            raise ValueError(f"Unknown case identifier: {group_id}")
        ratio_bin = int(stats_by_id[group_id]["ratio_bin"])
        by_bin.setdefault(ratio_bin, []).append(group_id)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    cal: list[str] = []
    for ratio_bin in sorted(by_bin):
        ids = sorted(by_bin[ratio_bin])
        if len(ids) < 3:
            raise ValueError(f"ratio_bin {ratio_bin} has fewer than three cases")
        rng.shuffle(ids)
        n_val = max(1, round(val_ratio * len(ids)))
        n_cal = max(1, round(cal_ratio * len(ids)))
        if n_val + n_cal >= len(ids):
            raise ValueError(f"ratio_bin {ratio_bin} is too small")
        val.extend(ids[:n_val])
        cal.extend(ids[n_val:n_val + n_cal])
        train.extend(ids[n_val + n_cal:])
    return sorted(train), sorted(val), sorted(cal)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outer-splits",
        type=Path,
        default=Path("data/splits/histopath_kaggle"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/paper_c/idc"),
    )
    parser.add_argument("--seed-base", type=int, default=1042)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--cal-ratio", type=float, default=0.125)
    args = parser.parse_args()

    stats_path = args.outer_splits / "patient_stats.csv"
    if not stats_path.is_file():
        raise SystemExit(f"Missing case statistics: {stats_path}")
    stats_rows = read_rows(stats_path)
    if len(stats_rows) != 279:
        raise SystemExit(f"Expected 279 case identifiers, found {len(stats_rows)}")
    stats_by_id = {row["patient_id"]: row for row in stats_rows}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "terminology": "public case identifiers; patient identity is not verified",
        "source_outer_splits": str(args.outer_splits),
        "source_patient_stats_sha256": sha256(stats_path),
        "seed_base": args.seed_base,
        "val_ratio": args.val_ratio,
        "cal_ratio": args.cal_ratio,
        "folds": [],
    }

    test_occurrences: Counter[str] = Counter()
    for fold in range(5):
        outer_dir = args.outer_splits / "folds" / f"fold_{fold}"
        train_path = outer_dir / "train_patients.csv"
        test_path = outer_dir / "test_patients.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise SystemExit(f"Missing frozen outer case lists under {outer_dir}")

        outer_train = [row["patient_id"] for row in read_rows(train_path)]
        outer_test = [row["patient_id"] for row in read_rows(test_path)]
        if set(outer_train) != set(stats_by_id) - set(outer_test):
            raise AssertionError(f"fold {fold}: outer train is not test complement")
        train, val, cal = split_groups(
            stats_by_id,
            outer_train,
            val_ratio=args.val_ratio,
            cal_ratio=args.cal_ratio,
            seed=args.seed_base + fold,
        )

        partitions = [set(train), set(val), set(cal), set(outer_test)]
        if any(
            partitions[i] & partitions[j]
            for i in range(len(partitions))
            for j in range(i + 1, len(partitions))
        ):
            raise AssertionError(f"fold {fold}: inner/outer partitions overlap")
        if set().union(*partitions) != set(stats_by_id):
            raise AssertionError(f"fold {fold}: partitions do not cover all cases")
        test_occurrences.update(outer_test)

        fold_dir = output / "folds" / f"fold_{fold}"
        paths = {
            "inner_train": fold_dir / "inner_train_case_ids.csv",
            "inner_val": fold_dir / "inner_val_case_ids.csv",
            "inner_cal": fold_dir / "inner_cal_case_ids.csv",
            "outer_test": fold_dir / "outer_test_case_ids.csv",
        }
        for name, ids in (
            ("inner_train", train),
            ("inner_val", val),
            ("inner_cal", cal),
            ("outer_test", sorted(outer_test)),
        ):
            write_ids(paths[name], ids)

        fold_summary = {
            "fold": fold,
            "seed": args.seed_base + fold,
            "counts": {name: len(read_rows(path)) for name, path in paths.items()},
            "sha256": {name: sha256(path) for name, path in paths.items()},
            "source_outer_train_sha256": sha256(train_path),
            "source_outer_test_sha256": sha256(test_path),
        }
        summary["folds"].append(fold_summary)  # type: ignore[union-attr]

    if set(test_occurrences) != set(stats_by_id) or any(
        count != 1 for count in test_occurrences.values()
    ):
        raise AssertionError("Every case must occur in exactly one outer test fold.")

    summary_path = output / "inner_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote frozen inner case-ID splits to {output}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
