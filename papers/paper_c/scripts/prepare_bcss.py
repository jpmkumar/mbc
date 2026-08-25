#!/usr/bin/env python3
"""Materialize the revision-pinned CC0 BCSS mirror as RGB/mask PNG pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

REPO_ID = "MedOtter/BCSS"
REVISION = "502d4a3fbc77dbaca6f4664c19e2379ff077d418"


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "provenance.json"
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/huggingface"))
    args = parser.parse_args()

    image_dir = args.output / "images"
    mask_dir = args.output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        REPO_ID,
        revision=REVISION,
        split="train",
        cache_dir=str(args.cache_dir),
    )
    if len(dataset) != 151:
        raise RuntimeError(f"Expected 151 BCSS ROIs, found {len(dataset)}")

    records = []
    for row in dataset:
        stem = row["image_id"]
        image_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        row["image"].convert("RGB").save(image_path)
        row["mask"].convert("L").save(mask_path)
        records.append({
            "image_id": stem,
            "patient_id": row["patient_id"],
            "xmin": int(row["xmin"]),
            "ymin": int(row["ymin"]),
        })

    metadata_path = args.output / "mirror_rows.json"
    metadata_path.write_text(json.dumps(records, indent=2) + "\n")
    provenance = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "CC0-1.0",
        "rows": len(records),
        "materialized_tree_sha256": tree_hash(args.output),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
