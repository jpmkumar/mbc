#!/usr/bin/env python3
"""
Dataset acquisition guide and split creation.

Public datasets:
  - Mammography: CBIS-DDSM (TCIA) — https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
  - Ultrasound: BUSI (Kaggle) — https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
  - Thermography: DMR-IR or Breast Thermography Database (Kaggle)

Expected folder layout after download:
  data/processed/
    mammo/{benign,malignant}/*.png
    ultrasound/{benign,malignant}/*.png
    thermo/{benign,malignant}/*.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.splits import create_stratified_splits


DATASET_SOURCES = {
    "mammo": {
        "name": "CBIS-DDSM",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM",
        "access": "TCIA registration required; use NBIA Data Retriever",
        "kaggle_alt": None,
    },
    "ultrasound": {
        "name": "BUSI",
        "url": "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset",
        "access": "kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset",
        "kaggle_alt": "aryashah2k/breast-ultrasound-images-dataset",
    },
    "thermo": {
        "name": "Breast Thermography Database",
        "url": "https://www.kaggle.com/datasets/viznote/breast-thermography-database",
        "access": "kaggle datasets download -d viznote/breast-thermography-database",
        "kaggle_alt": "viznote/breast-thermography-database",
    },
}


def print_download_instructions():
    print("=" * 70)
    print("DATASET ACQUISITION GUIDE")
    print("=" * 70)
    for mod, info in DATASET_SOURCES.items():
        print(f"\n[{mod.upper()}] {info['name']}")
        print(f"  URL: {info['url']}")
        print(f"  Access: {info['access']}")
    print("\n" + "=" * 70)
    print("After downloading, organize images into:")
    print("  data/processed/<modality>/{benign,malignant}/")
    print("\nFor immediate testing, run:")
    print("  python data/download/generate_synthetic.py --samples 50")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instructions-only", action="store_true")
    args = parser.parse_args()

    print_download_instructions()

    if args.instructions_only:
        return

    data_root = Path(args.data_root)
    if not data_root.exists() or not any(data_root.iterdir()):
        print("\nNo processed data found. Generating synthetic dataset...")
        from data.download.generate_synthetic import generate_dataset
        generate_dataset(str(data_root), samples_per_class=50, seed=args.seed)

    splits = create_stratified_splits(
        data_root=str(data_root),
        output_dir=args.splits_dir,
        seed=args.seed,
    )
    print(f"\nSplits created: {splits}")


if __name__ == "__main__":
    main()
