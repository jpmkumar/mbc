"""Stratified train/val/test split generation."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import MODALITIES


def _collect_images(data_root: Path, modality: str) -> list[dict]:
    """Collect image paths from processed folder layout."""
    records = []
    mod_dir = data_root / modality
    if not mod_dir.exists():
        return records

    for split in ("train", "val", "test"):
        split_dir = mod_dir / split
        if not split_dir.exists():
            continue
        for label_name in ("benign", "malignant"):
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            label = 0 if label_name == "benign" else 1
            for img_path in sorted(label_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                patient_id = f"{modality}_{img_path.stem}"
                records.append({
                    "filepath": str(img_path.relative_to(data_root)),
                    "modality": modality,
                    "label": label,
                    "patient_id": patient_id,
                })
    return records


def create_stratified_splits(
    data_root: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create patient-level stratified splits per modality."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    all_records = []
    for modality in MODALITIES:
        mod_dir = data_root / modality
        if not mod_dir.exists():
            continue

        # Flat layout: modality/{benign,malignant}/*.png
        flat_records = []
        for label_name in ("benign", "malignant"):
            label_dir = mod_dir / label_name
            if not label_dir.exists():
                continue
            label = 0 if label_name == "benign" else 1
            for img_path in sorted(label_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                flat_records.append({
                    "filepath": str(img_path.relative_to(data_root)),
                    "modality": modality,
                    "label": label,
                    "patient_id": f"{modality}_{img_path.stem}",
                })

        if not flat_records:
            # Already split layout
            all_records.extend(_collect_images(data_root, modality))
            continue

        df = pd.DataFrame(flat_records)
        patients = df["patient_id"].astype(str).unique().tolist()

        train_p, temp_p = train_test_split(
            patients, test_size=(1 - train_ratio), random_state=seed, shuffle=True
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_p, test_p = train_test_split(
            temp_p, test_size=(1 - val_size), random_state=seed, shuffle=True
        )

        split_map = {}
        for p in train_p:
            split_map[p] = "train"
        for p in val_p:
            split_map[p] = "val"
        for p in test_p:
            split_map[p] = "test"

        df["split"] = df["patient_id"].map(split_map)
        all_records.extend(df.to_dict("records"))

    manifest = pd.DataFrame(all_records)
    if manifest.empty:
        raise FileNotFoundError(f"No images found under {data_root}")

    splits = {}
    for split_name in ("train", "val", "test"):
        split_df = manifest[manifest["split"] == split_name].drop(columns=["split"])
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        splits[split_name] = str(out_path)

    stats = {}
    for modality in MODALITIES:
        mod_df = manifest[manifest["modality"] == modality]
        if mod_df.empty:
            continue
        stats[modality] = {
            "total": len(mod_df),
            "benign": int((mod_df["label"] == 0).sum()),
            "malignant": int((mod_df["label"] == 1).sum()),
            "train": int((mod_df["split"] == "train").sum()),
            "val": int((mod_df["split"] == "val").sum()),
            "test": int((mod_df["split"] == "test").sum()),
        }

    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    splits["stats"] = str(stats_path)
    return splits


def load_splits(splits_dir: str) -> dict[str, str]:
    splits_dir = Path(splits_dir)
    return {
        "train": str(splits_dir / "train.csv"),
        "val": str(splits_dir / "val.csv"),
        "test": str(splits_dir / "test.csv"),
    }
