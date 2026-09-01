#!/usr/bin/env python3
"""Build the patient/site-held-out BCSS centre manifest from RGB/mask pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_CODES = {0, 7, 15}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--patient-splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--centre-size", type=int, default=50)
    parser.add_argument("--context-size", type=int, default=450)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-purity", type=float, default=0.8)
    args = parser.parse_args()

    image_paths = {path.stem: path for path in args.images.glob("*.png")}
    mask_paths = {path.stem: path for path in args.masks.glob("*.png")}
    if set(image_paths) != set(mask_paths):
        raise SystemExit("BCSS image and mask stems do not match exactly.")
    if len(image_paths) != 151:
        raise SystemExit(f"Expected 151 BCSS ROIs, found {len(image_paths)}")
    metadata_rows = json.loads(args.metadata.read_text())
    metadata_by_roi = {
        str(row["image_id"]): str(row["patient_id"]) for row in metadata_rows
    }
    if len(metadata_by_roi) != 151 or set(metadata_by_roi) != set(image_paths):
        raise SystemExit("BCSS mirror metadata does not match RGB/mask ROI stems.")
    with args.patient_splits.open(newline="", encoding="utf-8") as stream:
        patient_split_rows = list(csv.DictReader(stream))
    split_by_patient = {
        row["patient_id"]: (row["site"], row["split"])
        for row in patient_split_rows
    }
    if len(split_by_patient) != len(patient_split_rows):
        raise SystemExit("BCSS patient split artifact contains duplicates.")

    half_centre = args.centre_size // 2
    half_context = args.context_size // 2
    rows: list[dict[str, object]] = []
    roi_records: list[dict[str, object]] = []
    patient_roles: dict[str, str] = {}
    for stem in sorted(image_paths):
        patient_id = metadata_by_roi[stem]
        if patient_id not in split_by_patient:
            raise RuntimeError(f"No frozen BCSS split for {patient_id}")
        site, split = split_by_patient[patient_id]
        previous = patient_roles.setdefault(patient_id, split)
        if previous != split:
            raise RuntimeError(f"Patient {patient_id} appears in multiple splits.")
        with Image.open(image_paths[stem]) as image:
            width, height = image.size
        with Image.open(mask_paths[stem]) as mask_image:
            mask = np.asarray(mask_image)
        if mask.shape[:2] != (height, width):
            raise RuntimeError(f"RGB/mask shape mismatch for {stem}")

        accepted = 0
        for y in range(half_context, height - half_context + 1, args.stride):
            for x in range(half_context, width - half_context + 1, args.stride):
                centre = mask[
                    y - half_centre:y + half_centre,
                    x - half_centre:x + half_centre,
                ]
                if centre.shape != (args.centre_size, args.centre_size):
                    continue
                valid = ~np.isin(centre, list(IGNORE_CODES))
                valid_fraction = float(valid.mean())
                if valid_fraction < args.minimum_valid_fraction:
                    continue
                valid_labels = centre[valid]
                tumour_fraction = float(np.mean(valid_labels == 1))
                other_fraction = 1.0 - tumour_fraction
                purity = max(tumour_fraction, other_fraction)
                if purity < args.minimum_purity:
                    continue
                label = int(tumour_fraction >= args.minimum_purity)
                rows.append({
                    "filepath": f"{stem}:x{x}:y{y}",
                    "image_filename": image_paths[stem].name,
                    "mask_filename": mask_paths[stem].name,
                    "roi_id": stem,
                    "patient_id": patient_id,
                    "site": site,
                    "split": split,
                    "x": x,
                    "y": y,
                    "label": label,
                    "valid_fraction": valid_fraction,
                    "tumour_fraction": tumour_fraction,
                    "context_complete": 1,
                })
                accepted += 1
        roi_records.append({
            "roi_id": stem,
            "patient_id": patient_id,
            "site": site,
            "split": split,
            "width": width,
            "height": height,
            "accepted_centres": accepted,
        })

    if not rows:
        raise SystemExit("No BCSS centres passed the preregistered criteria.")
    split_label_counts = {
        split: {
            label: sum(
                row["split"] == split and row["label"] == label for row in rows
            )
            for label in (0, 1)
        }
        for split in ("train", "val", "cal", "test")
    }
    if any(
        split_label_counts[split][label] == 0
        for split in split_label_counts
        for label in (0, 1)
    ):
        raise RuntimeError(
            f"BCSS split lacks a class after centre filtering: {split_label_counts}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    roi_path = args.output.with_name(args.output.stem + "_rois.csv")
    with roi_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(roi_records[0]))
        writer.writeheader()
        writer.writerows(roi_records)

    split_patients = {
        split: sorted(patient for patient, role in patient_roles.items() if role == split)
        for split in ("train", "val", "cal", "test")
    }
    if any(
        set(split_patients[left]) & set(split_patients[right])
        for left in split_patients
        for right in split_patients
        if left < right
    ):
        raise RuntimeError("BCSS patient split overlap.")
    report = {
        "images": len(image_paths),
        "patients": len(patient_roles),
        "test_sites": sorted(
            {row["site"] for row in rows if row["split"] == "test"}
        ),
        "development_sites": sorted(
            {row["site"] for row in rows if row["split"] != "test"}
        ),
        "split_patient_counts": {
            split: len(patients) for split, patients in split_patients.items()
        },
        "centres": len(rows),
        "positive_centres": sum(int(row["label"]) for row in rows),
        "split_label_counts": split_label_counts,
        "stride": args.stride,
        "centre_size": args.centre_size,
        "context_size": args.context_size,
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "minimum_purity": args.minimum_purity,
        "ignore_codes": sorted(IGNORE_CODES),
        "mirror_metadata_sha256": sha256(args.metadata),
        "patient_split_sha256": sha256(args.patient_splits),
        "manifest_sha256": sha256(args.output),
        "roi_manifest_sha256": sha256(roi_path),
    }
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
