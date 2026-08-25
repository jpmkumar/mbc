#!/usr/bin/env python3
"""Build the canonical filepath-keyed IDC protocol manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def stable_key(seed: int, filepath: str) -> str:
    return hashlib.sha256(f"{seed}|{filepath}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--case-stats", required=True, type=Path)
    parser.add_argument(
        "--outer-folds",
        type=Path,
        default=Path("data/splits/histopath_kaggle/folds"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(args.index)
    if len(rows) != 277_524:
        raise SystemExit(f"Expected 277,524 index rows, found {len(rows):,}")
    paths = [row["filepath"] for row in rows]
    if len(paths) != len(set(paths)):
        raise SystemExit("Input index filepath is not unique.")

    case_rows = read_rows(args.case_stats)
    grouped_fold = {row["patient_id"]: int(row["test_fold"]) for row in case_rows}
    if len(grouped_fold) != 279:
        raise SystemExit(f"Expected 279 case-fold assignments, found {len(grouped_fold)}")
    if {row["case_id"] for row in rows} != set(grouped_fold):
        raise SystemExit("Index case identifiers do not match frozen grouped folds.")
    frozen_fold: dict[str, int] = {}
    occurrences: Counter[str] = Counter()
    for fold in range(5):
        fold_dir = args.outer_folds / f"fold_{fold}"
        test_ids = {
            row["patient_id"] for row in read_rows(fold_dir / "test_patients.csv")
        }
        train_ids = {
            row["patient_id"] for row in read_rows(fold_dir / "train_patients.csv")
        }
        if train_ids != set(grouped_fold) - test_ids:
            raise SystemExit(f"Fold {fold}: train list is not the test complement.")
        occurrences.update(test_ids)
        frozen_fold.update({case_id: fold for case_id in test_ids})
    if set(occurrences) != set(grouped_fold) or any(
        count != 1 for count in occurrences.values()
    ):
        raise SystemExit("Frozen outer test lists are not an exact five-fold partition.")
    if frozen_fold != grouped_fold:
        raise SystemExit("patient_stats.csv test_fold disagrees with frozen fold CSVs.")

    random_folds: dict[str, int] = {}
    for label in ("0", "1"):
        members = sorted(
            (row["filepath"] for row in rows if row["label"] == label),
            key=lambda filepath: stable_key(args.random_seed, filepath),
        )
        for rank, filepath in enumerate(members):
            random_folds[filepath] = rank % 5

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filepath", "case_id", "label", "grouped_outer_fold",
        "random_outer_fold", "k3_complete", "k5_complete", "k9_complete",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "filepath": row["filepath"],
                "case_id": row["case_id"],
                "label": row["label"],
                "grouped_outer_fold": grouped_fold[row["case_id"]],
                "random_outer_fold": random_folds[row["filepath"]],
                "k3_complete": row.get("k3_complete", ""),
                "k5_complete": row.get("k5_complete", ""),
                "k9_complete": row.get("k9_complete", ""),
            })

    report = {
        "rows": len(rows),
        "case_identifiers": len(grouped_fold),
        "random_seed": args.random_seed,
        "random_assignment": (
            "within each label, sort by SHA256(seed|filepath), then rank modulo 5"
        ),
        "source_index_sha256": sha256(args.index),
        "source_case_stats_sha256": sha256(args.case_stats),
        "outer_fold_csvs_validated": True,
        "manifest_sha256": sha256(args.output),
        "output": str(args.output.resolve()),
    }
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
