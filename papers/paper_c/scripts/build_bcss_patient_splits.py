#!/usr/bin/env python3
"""Freeze exact BCSS patient/site partitions before centre generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

TEST_SITES = {"OL", "LL", "E2", "EW", "GM", "S3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def site_from_patient(patient_id: str) -> str:
    parts = patient_id.split("-")
    if len(parts) != 3 or parts[0] != "TCGA":
        raise ValueError(f"Invalid TCGA patient barcode: {patient_id}")
    return parts[1]


def split_for(patient_id: str, site: str) -> str:
    if site in TEST_SITES:
        return "test"
    value = int(hashlib.sha256(f"32026|{patient_id}".encode()).hexdigest(), 16) % 100
    return "train" if value < 70 else ("val" if value < 85 else "cal")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = json.loads(args.metadata.read_text())
    if len(records) != 151:
        raise SystemExit(f"Expected 151 BCSS ROI metadata rows, found {len(records)}")
    patient_by_roi: dict[str, str] = {}
    patients: dict[str, str] = {}
    for record in records:
        roi = str(record["image_id"])
        patient = str(record["patient_id"])
        if roi in patient_by_roi:
            raise SystemExit(f"Duplicate BCSS image_id: {roi}")
        patient_by_roi[roi] = patient
        patients[patient] = site_from_patient(patient)

    rows = [
        {
            "patient_id": patient,
            "site": site,
            "split": split_for(patient, site),
        }
        for patient, site in sorted(patients.items())
    ]
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val", "cal", "test")
    }
    if any(count == 0 for count in split_counts.values()):
        raise SystemExit(f"BCSS patient partition is empty: {split_counts}")
    development_sites = {
        row["site"] for row in rows if row["split"] != "test"
    }
    test_sites = {row["site"] for row in rows if row["split"] == "test"}
    if development_sites & test_sites or test_sites != TEST_SITES:
        raise SystemExit(
            f"BCSS institution holdout invalid: dev={development_sites}, test={test_sites}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["patient_id", "site", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "metadata_sha256": sha256(args.metadata),
        "patient_split_sha256": sha256(args.output),
        "patients": len(rows),
        "rois": len(records),
        "test_sites": sorted(TEST_SITES),
        "split_patient_counts": split_counts,
        "development_assignment": (
            'SHA256("32026|" + patient_id) mod 100: '
            "0-69 train, 70-84 val, 85-99 cal"
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
