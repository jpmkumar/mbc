#!/usr/bin/env python3
"""Audit IDC coordinate geometry before any mosaic extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

EXCLUDED_DIR = "IDC_regular_ps50_idx5"
FNAME = re.compile(
    r"^(?P<stem>.+?)_idx(?P<idx>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_class(?P<cls>\d+)\.png$"
)
TILE = 50
KS = (1, 3, 5, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    archive = args.archive_path.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    lookup: set[tuple[str, int, int]] = set()
    by_case_x: dict[str, set[int]] = defaultdict(set)
    by_case_y: dict[str, set[int]] = defaultdict(set)
    for case_id in sorted(os.listdir(archive)):
        if case_id == EXCLUDED_DIR or not (archive / case_id).is_dir():
            continue
        for label in ("0", "1"):
            class_dir = archive / case_id / label
            if not class_dir.is_dir():
                continue
            for entry in os.scandir(class_dir):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                match = FNAME.match(entry.name)
                if not match:
                    raise RuntimeError(f"Unexpected filename: {entry.name}")
                x, y = int(match.group("x")), int(match.group("y"))
                key = (case_id, x, y)
                if key in lookup:
                    raise RuntimeError(f"Duplicate coordinate: {key}")
                lookup.add(key)
                by_case_x[case_id].add(x)
                by_case_y[case_id].add(y)
                rows.append({
                    "filepath": f"{case_id}/{label}/{entry.name}",
                    "case_id": case_id,
                    "label": int(label),
                    "x": x,
                    "y": y,
                })
    rows.sort(key=lambda row: (row["case_id"], row["y"], row["x"], row["label"]))
    if len(rows) != 277_524:
        raise RuntimeError(f"Expected 277,524 patches, found {len(rows):,}")

    x_steps: Counter[int] = Counter()
    y_steps: Counter[int] = Counter()
    for case_id in by_case_x:
        xs, ys = sorted(by_case_x[case_id]), sorted(by_case_y[case_id])
        x_steps.update(b - a for a, b in zip(xs, xs[1:]) if b > a)
        y_steps.update(b - a for a, b in zip(ys, ys[1:]) if b > a)

    counts = {str(k): 0 for k in KS}
    index_path = output / "idc_context_eligibility.csv"
    fields = ["filepath", "case_id", "label", "x", "y"]
    for k in KS:
        fields.extend([f"k{k}_available", f"k{k}_complete"])

    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for k in KS:
                half = k // 2
                available = sum(
                    (
                        row["case_id"],
                        int(row["x"]) + gx * TILE,
                        int(row["y"]) + gy * TILE,
                    ) in lookup
                    for gy in range(-half, half + 1)
                    for gx in range(-half, half + 1)
                )
                complete = int(available == k * k)
                row[f"k{k}_available"] = available
                row[f"k{k}_complete"] = complete
                counts[str(k)] += complete
            writer.writerow(row)

    report = {
        "archive_path": str(archive),
        "patches": len(rows),
        "case_identifiers": len(by_case_x),
        "nominal_tile_pixels": TILE,
        "complete_context_counts": counts,
        "x_step_histogram_top10": x_steps.most_common(10),
        "y_step_histogram_top10": y_steps.most_common(10),
        "coordinate_grid_50px_supported": (
            bool(x_steps) and bool(y_steps)
            and x_steps.most_common(1)[0][0] == TILE
            and y_steps.most_common(1)[0][0] == TILE
        ),
        "eligibility_index": str(index_path),
        "eligibility_index_sha256": sha256(index_path),
    }
    report_path = output / "idc_context_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
