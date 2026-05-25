#!/usr/bin/env python3
"""
Download CBIS-DDSM mammography from TCIA without NBIA Data Retriever.

Uses:
  - Label CSVs from https://www.cancerimagingarchive.net/collection/cbis-ddsm/
  - TCIA NBIA REST API: getImage?SeriesInstanceUID=...

Organizes cropped mammograms into:
  data/processed/mammo/benign/*.png
  data/processed/mammo/malignant/*.png

Citation (required):
  Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016).
  Curated Breast Imaging Subset of DDSM [Data set]. TCIA.
  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
"""

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.splits import create_stratified_splits

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
CSV_URLS = {
    "mass_train": "https://www.cancerimagingarchive.net/wp-content/uploads/mass_case_description_train_set.csv",
    "mass_test": "https://www.cancerimagingarchive.net/wp-content/uploads/mass_case_description_test_set.csv",
    "calc_train": "https://www.cancerimagingarchive.net/wp-content/uploads/calc_case_description_train_set.csv",
    "calc_test": "https://www.cancerimagingarchive.net/wp-content/uploads/calc_case_description_test_set.csv",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def download_csvs(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in CSV_URLS.items():
        out = raw_dir / f"{name}.csv"
        if not out.exists():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            out.write_bytes(r.content)
        paths.append(out)
    return paths


def load_cases(csv_paths: list[Path], use_cropped: bool = True) -> pd.DataFrame:
    rows = []
    path_col = "cropped_image_file_path" if use_cropped else "image_file_path"
    for p in csv_paths:
        df = _normalize_columns(pd.read_csv(p))
        if path_col not in df.columns:
            continue
        for _, row in df.iterrows():
            pathology = str(row.get("pathology", "")).strip().upper()
            if pathology not in ("BENIGN", "MALIGNANT"):
                continue
            dcm_path = str(row[path_col]).strip().strip('"').strip()
            if not dcm_path or dcm_path == "nan":
                continue
            parts = dcm_path.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            series_uid = parts[-2]
            patient = str(row.get("patient_id", parts[0]))
            rows.append({
                "patient_id": patient,
                "pathology": pathology.lower(),
                "series_uid": series_uid,
                "dcm_path": dcm_path,
                "source_csv": p.name,
            })
    cases = pd.DataFrame(rows).drop_duplicates(subset=["series_uid"])
    return cases


def _dicom_to_png(dcm_bytes: bytes) -> Image.Image:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
    arr = ds.pixel_array.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype("uint8")
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def download_series(series_uid: str, session: requests.Session, retries: int = 3) -> bytes | None:
    url = f"{TCIA_BASE}/getImage"
    for attempt in range(retries):
        try:
            r = session.get(url, params={"SeriesInstanceUID": series_uid}, timeout=300)
            if r.status_code == 200 and r.content[:2] == b"PK":
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def extract_first_dicom(zip_bytes: bytes) -> bytes | None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith(".dcm"):
                return zf.read(name)
    return None


def main():
    parser = argparse.ArgumentParser(description="Download CBIS-DDSM via TCIA REST API")
    parser.add_argument("--raw-dir", default="data/raw/cbis-ddsm")
    parser.add_argument("--output", default="data/processed/mammo")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limit images per class (for testing)")
    parser.add_argument("--full", action="store_true",
                        help="Download all cases (~3.5k cropped images, large)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--recreate-splits", action="store_true")
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    output_root = ROOT / args.output
    progress_file = raw_dir / "download_progress.json"

    print("=" * 60)
    print("CBIS-DDSM Download (TCIA REST API — no NBIA Retriever needed)")
    print("Collection: https://www.cancerimagingarchive.net/collection/cbis-ddsm/")
    print("=" * 60)

    csv_paths = download_csvs(raw_dir)
    cases = load_cases(csv_paths, use_cropped=True)
    print(f"Total unique cropped cases: {len(cases)}")
    print(cases["pathology"].value_counts().to_string())

    if not args.full and args.max_per_class is None:
        args.max_per_class = 100
        print(f"\nDefault limit: {args.max_per_class} per class (use --full for all)")

    if args.max_per_class:
        selected = []
        for label in ("benign", "malignant"):
            sub = cases[cases["pathology"] == label].head(args.max_per_class)
            selected.append(sub)
        cases = pd.concat(selected, ignore_index=True)
        print(f"Downloading subset: {len(cases)} images")

    done = set()
    if args.resume and progress_file.exists():
        done = set(json.loads(progress_file.read_text()))

    session = requests.Session()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for _, row in tqdm(cases.iterrows(), total=len(cases), desc="CBIS-DDSM"):
        series_uid = row["series_uid"]
        label = row["pathology"]
        out_path = output_root / label / f"{row['patient_id']}_{series_uid[-8:]}.png"

        if out_path.exists() or series_uid in done:
            stats["skipped"] += 1
            continue

        zip_bytes = download_series(series_uid, session)
        if not zip_bytes:
            stats["failed"] += 1
            continue

        dcm_bytes = extract_first_dicom(zip_bytes)
        if not dcm_bytes:
            stats["failed"] += 1
            continue

        try:
            img = _dicom_to_png(dcm_bytes)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            stats["downloaded"] += 1
            done.add(series_uid)
            if stats["downloaded"] % 10 == 0:
                progress_file.write_text(json.dumps(list(done)))
        except Exception:
            stats["failed"] += 1

    progress_file.write_text(json.dumps(list(done)))
    cases.to_csv(raw_dir / "cases_manifest.csv", index=False)

    print("\nDownload stats:", stats)
    print(f"Saved to: {output_root}")

    if stats["downloaded"] > 0 or args.recreate_splits:
        splits = create_stratified_splits(
            data_root=str(ROOT / "data/processed"),
            output_dir=str(ROOT / args.splits_dir),
            seed=42,
        )
        print("Splits updated:", splits)


if __name__ == "__main__":
    main()
