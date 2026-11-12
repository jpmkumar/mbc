#!/usr/bin/env python3
"""Freeze Paper C+ BreakHis inner train/val/cal case lists per outer fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTER = ROOT / "data" / "splits" / "paper_c_plus" / "breakhis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id"])
        writer.writerows((case_id,) for case_id in ids)


def split_by_label(
    cases: list[dict[str, str]],
    *,
    val_ratio: float,
    cal_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    by_label: dict[str, list[str]] = {"benign": [], "malignant": []}
    for row in cases:
        by_label[row["label"]].append(row["case_id"])
    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    cal: list[str] = []
    for label in ("benign", "malignant"):
        ids = sorted(by_label[label])
        if len(ids) < 3:
            raise SystemExit(f"label {label} has fewer than 3 train cases")
        rng.shuffle(ids)
        n_val = max(1, round(val_ratio * len(ids)))
        n_cal = max(1, round(cal_ratio * len(ids)))
        if n_val + n_cal >= len(ids):
            n_val = 1
            n_cal = 1
        val.extend(ids[:n_val])
        cal.extend(ids[n_val : n_val + n_cal])
        train.extend(ids[n_val + n_cal :])
    return sorted(train), sorted(val), sorted(cal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-root", type=Path, default=DEFAULT_OUTER)
    parser.add_argument("--seed-base", type=int, default=1042)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--cal-ratio", type=float, default=0.125)
    args = parser.parse_args()

    summary = {"folds": [], "seed_base": args.seed_base}
    for fold in range(5):
        train_path = args.outer_root / "folds" / f"fold_{fold}" / "train_cases.csv"
        cases = read_cases(train_path)
        train, val, cal = split_by_label(
            cases,
            val_ratio=args.val_ratio,
            cal_ratio=args.cal_ratio,
            seed=args.seed_base + fold,
        )
        if set(train) | set(val) | set(cal) != {row["case_id"] for row in cases}:
            raise SystemExit(f"fold {fold}: inner sets do not cover train cases")
        if len(train) + len(val) + len(cal) != len(cases):
            raise SystemExit(f"fold {fold}: duplicate or missing inner assignment")
        out = args.outer_root / "folds" / f"fold_{fold}"
        write_ids(out / "inner_train_case_ids.csv", train)
        write_ids(out / "inner_val_case_ids.csv", val)
        write_ids(out / "inner_cal_case_ids.csv", cal)
        summary["folds"].append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_val": len(val),
                "n_cal": len(cal),
                "train_sha256": sha256(out / "inner_train_case_ids.csv"),
                "val_sha256": sha256(out / "inner_val_case_ids.csv"),
                "cal_sha256": sha256(out / "inner_cal_case_ids.csv"),
            }
        )

    summary_path = args.outer_root / "inner_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
