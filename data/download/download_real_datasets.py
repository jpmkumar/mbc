#!/usr/bin/env python3
"""
Download real breast imaging datasets from Hugging Face and organize for training.

Sources:
  - Ultrasound: Angelou0516/BUSI (780 images, benign/malignant/normal)
  - Mammography: dbaek111/CBIS-DDSM_1024 (cancer / not_cancer PNGs)
  - Thermography: SemilleroCV/DMR-IR (IR TIFF images + mastectomy label)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.splits import create_stratified_splits


def _save_pil(img, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    img.save(out_path)


def download_busi(output_root: Path) -> dict:
    from datasets import load_dataset

    print("\n[1/3] Downloading BUSI ultrasound dataset...")
    ds = load_dataset("Angelou0516/BUSI", split="train")
    stats = {"benign": 0, "malignant": 0, "skipped_normal": 0}

    for i, row in enumerate(tqdm(ds, desc="BUSI")):
        label_name = row["class_label"]
        if label_name == "normal":
            stats["skipped_normal"] += 1
            continue
        if label_name not in ("benign", "malignant"):
            continue
        img_id = row.get("image_id", f"busi_{i}")
        safe_name = str(img_id).replace("/", "_").replace(" ", "_")
        out_path = output_root / "ultrasound" / label_name / f"{safe_name}.png"
        _save_pil(row["image"], out_path)
        stats[label_name] += 1

    return stats


def download_cbis_ddsm(output_root: Path) -> dict:
    from datasets import load_dataset

    print("\n[2/3] Downloading CBIS-DDSM mammography dataset...")
    ds = load_dataset("dbaek111/CBIS-DDSM_1024", split="train", trust_remote_code=True)
    stats = {"benign": 0, "malignant": 0}

    # Inspect label column
    sample = ds[0]
    label_key = None
    for k in ("label", "labels", "class", "category", "path"):
        if k in sample:
            label_key = k
            break

    for i, row in enumerate(tqdm(ds, desc="CBIS-DDSM")):
        img = row.get("image") or row.get("img")
        if img is None:
            continue

        # Infer label from folder structure or explicit label field
        label = row.get("label") or row.get("labels") or row.get("class")
        if label is None:
            path_str = str(row.get("path", row.get("file_name", ""))).lower()
            if "cancer" in path_str or "malignant" in path_str:
                label_name = "malignant"
            elif "not_cancer" in path_str or "benign" in path_str or "normal" in path_str:
                label_name = "benign"
            else:
                continue
        else:
            label_str = str(label).lower()
            if label_str in ("1", "cancer", "malignant", "positive"):
                label_name = "malignant"
            else:
                label_name = "benign"

        out_path = output_root / "mammo" / label_name / f"mammo_{i:05d}.png"
        _save_pil(img, out_path)
        stats[label_name if label_name == "malignant" else "benign"] += 1

    return stats


def download_dmr_ir(output_root: Path, max_samples: int | None = None) -> dict:
    from datasets import load_dataset

    print("\n[3/3] Downloading DMR-IR thermography dataset...")
    ds = load_dataset("SemilleroCV/DMR-IR", split="train")
    stats = {"benign": 0, "malignant": 0, "skipped": 0}

    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i, row in enumerate(tqdm(ds.select(range(n)), desc="DMR-IR")):
        img = row.get("image") or row.get("ir_image") or row.get("thermal_image")
        if img is None:
            stats["skipped"] += 1
            continue

        # mastectomy column: 1 = malignant indicator; 0 = benign/healthy
        mastectomy = row.get("mastectomy")
        if mastectomy is None:
            # fallback: check diagnosis-related fields
            diag = str(row.get("diagnosis", row.get("label", ""))).lower()
            if any(x in diag for x in ("malign", "cancer", "carcinoma")):
                label_name = "malignant"
            elif any(x in diag for x in ("benign", "normal", "healthy")):
                label_name = "benign"
            else:
                stats["skipped"] += 1
                continue
        else:
            label_name = "malignant" if int(mastectomy) == 1 else "benign"

        patient_id = row.get("patient_id", row.get("id", i))
        out_path = output_root / "thermo" / label_name / f"thermo_{patient_id}_{i}.png"
        _save_pil(img, out_path)
        stats[label_name] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--raw-backup", default="data/raw")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--modalities", nargs="+", default=["mammo", "ultrasound", "thermo"])
    parser.add_argument("--thermo-max", type=int, default=None, help="Limit thermo samples (large dataset)")
    parser.add_argument("--clear", action="store_true", help="Clear existing processed data first")
    args = parser.parse_args()

    output_root = ROOT / args.output
    if args.clear and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    if "ultrasound" in args.modalities:
        all_stats["ultrasound"] = download_busi(output_root)
    if "mammo" in args.modalities:
        all_stats["mammo"] = download_cbis_ddsm(output_root)
    if "thermo" in args.modalities:
        all_stats["thermo"] = download_dmr_ir(output_root, max_samples=args.thermo_max)

    stats_path = ROOT / args.raw_backup / "download_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print("\nCreating stratified splits...")
    splits = create_stratified_splits(
        data_root=str(output_root),
        output_dir=str(ROOT / args.splits_dir),
        seed=42,
    )

    split_stats = json.loads(Path(splits["stats"]).read_text())
    print("\n" + "=" * 60)
    print("REAL DATASET DOWNLOAD COMPLETE")
    print("=" * 60)
    print(json.dumps({"download": all_stats, "splits": split_stats}, indent=2))
    print(f"\nProcessed data: {output_root}")
    print(f"Splits: {ROOT / args.splits_dir}")


if __name__ == "__main__":
    main()
