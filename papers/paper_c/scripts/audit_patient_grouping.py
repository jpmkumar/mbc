#!/usr/bin/env python3
"""Phase 0 — resolve the 162-slides versus 279-directories discrepancy.

The IDC archive is laid out as ``<archive>/<dir_id>/<class>/<file>.png`` where
each filename follows ``{dir_id}_idx{N}_x{X}_y{Y}_class{C}.png``. The source
description records 162 whole-mount slides; the Kaggle release exposes 279
top-level directories. Published papers cite both numbers interchangeably, so
the grouping variable behind any "patient-level" claim is not consistent across
the literature.

This script answers, from the archive alone:

1. How many top-level directories exist?
2. How many distinct ``idx`` values appear, and does any directory contain more
   than one? If directories alias slides, grouping must happen at the coarser
   level.
3. Do any two directories share a filename stem prefix, which would indicate one
   physical case split across directories?
4. What is the patch-coordinate extent per directory (a proxy for whole-mount
   area) and does it look like one mount or several?

No GPU, no labels beyond the directory class, no model. Safe to run before
preregistration.

Usage:
    python audit_patient_grouping.py --archive-path /path/to/archive \
        --output ../results/phase0_patient_grouping.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

EXCLUDED_DIR = "IDC_regular_ps50_idx5"
FNAME = re.compile(r"^(?P<stem>.+?)_idx(?P<idx>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_class(?P<cls>\d+)\.png$")


def scan(archive: Path) -> tuple[dict, list[str]]:
    dirs = sorted(
        name for name in os.listdir(archive)
        if (archive / name).is_dir() and name != EXCLUDED_DIR
    )

    per_dir: dict[str, dict] = {}
    anomalies: list[str] = []

    for dir_id in dirs:
        idx_values: set[str] = set()
        stems: set[str] = set()
        xs: list[int] = []
        ys: list[int] = []
        counts = {"0": 0, "1": 0}
        unparsed = 0

        for cls in ("0", "1"):
            cls_dir = archive / dir_id / cls
            if not cls_dir.is_dir():
                continue
            for entry in os.scandir(cls_dir):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                counts[cls] += 1
                m = FNAME.match(entry.name)
                if not m:
                    unparsed += 1
                    continue
                idx_values.add(m.group("idx"))
                stems.add(m.group("stem"))
                xs.append(int(m.group("x")))
                ys.append(int(m.group("y")))

        total = counts["0"] + counts["1"]
        if unparsed:
            anomalies.append(f"{dir_id}: {unparsed} filenames did not match the expected pattern")
        if len(idx_values) > 1:
            anomalies.append(f"{dir_id}: multiple idx values {sorted(idx_values)} — directory aliases >1 slide")
        if len(stems) > 1:
            anomalies.append(f"{dir_id}: multiple filename stems {sorted(stems)}")
        if stems and dir_id not in stems:
            anomalies.append(f"{dir_id}: filename stem {sorted(stems)} does not match directory name")

        per_dir[dir_id] = {
            "n0": counts["0"],
            "n1": counts["1"],
            "total": total,
            "idc_ratio": (counts["1"] / total) if total else 0.0,
            "idx_values": sorted(idx_values),
            "stems": sorted(stems),
            "x_range": [min(xs), max(xs)] if xs else None,
            "y_range": [min(ys), max(ys)] if ys else None,
            "unparsed": unparsed,
        }

    return per_dir, anomalies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-path", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    archive = args.archive_path.expanduser().resolve()
    if not archive.is_dir():
        raise SystemExit(f"Archive not found: {archive}")

    per_dir, anomalies = scan(archive)

    total_patches = sum(d["total"] for d in per_dir.values())
    total_pos = sum(d["n1"] for d in per_dir.values())
    stem_to_dirs: dict[str, list[str]] = defaultdict(list)
    for dir_id, d in per_dir.items():
        for stem in d["stems"]:
            stem_to_dirs[stem].append(dir_id)
    shared_stems = {s: v for s, v in stem_to_dirs.items() if len(v) > 1}

    all_idx = sorted({i for d in per_dir.values() for i in d["idx_values"]})

    summary = {
        "archive_path": str(archive),
        "n_directories": len(per_dir),
        "total_patches": total_patches,
        "total_idc_positive": total_pos,
        "idc_ratio": (total_pos / total_patches) if total_patches else 0.0,
        "distinct_idx_values": all_idx,
        "directories_with_multiple_idx": [k for k, v in per_dir.items() if len(v["idx_values"]) > 1],
        "stems_shared_across_directories": shared_stems,
        "anomalies": anomalies,
    }

    print(json.dumps(summary, indent=2)[:4000])
    print()
    print(f"directories                : {summary['n_directories']}")
    print(f"total patches              : {total_patches:,}  (expected 277,524)")
    print(f"IDC-positive patches       : {total_pos:,}  (expected 78,786)")
    print(f"distinct idx values        : {all_idx}")
    print(f"dirs aliasing >1 slide     : {len(summary['directories_with_multiple_idx'])}")
    print(f"stems spanning >1 directory: {len(shared_stems)}")

    print()
    if not summary["directories_with_multiple_idx"] and not shared_stems:
        print("VERDICT: each directory maps to exactly one slide stem.")
        print("  The directory is a valid grouping unit. The 162 figure refers to")
        print("  whole-mount slides in the source study, not to this distribution.")
        print("  -> group by directory; state both numbers explicitly in the paper.")
    else:
        print("VERDICT: directories DO NOT map one-to-one onto slides.")
        print("  Patient-disjoint folds must group at the coarser unit.")
        print("  -> the existing frozen folds must be re-audited before Phase 1.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "per_directory": per_dir}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
