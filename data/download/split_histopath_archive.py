#!/usr/bin/env python3
"""
Patient-level splits for Breast Histopathology (IDC) archive.

Modes:
  holdout  — single 80/20 split stratified by IDC-ratio quartile (debugging)
  cv       — patient-level StratifiedGroupKFold (default: 5 folds)

CV uses sklearn.model_selection.StratifiedGroupKFold with:
  - group = patient_id (no patient leakage across train/test)
  - stratify label = IDC-ratio quartile bin

Expected layout:
  archive/
    <patient_id>/
      0/   # non-IDC patches
      1/   # IDC patches
    IDC_regular_ps50_idx5/   # excluded
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.histopath_splits import (
    assign_ratio_bins,
    build_patch_manifest,
    collect_patient_stats,
    create_stratified_patient_folds,
    summarize_patients,
    write_cv_fold_manifests,
    write_split_metadata,
)

DEFAULT_ARCHIVE_CANDIDATES = (
    ROOT / "archive",
    Path.home() / "Downloads" / "archive",
    Path.home() / "Downloads" / "Histopathology-dataset",
)


def resolve_archive_path(path: str | None) -> Path:
    if path:
        archive = Path(path).expanduser().resolve()
    else:
        archive = next((p for p in DEFAULT_ARCHIVE_CANDIDATES if p.is_dir()), None)
        if archive is None:
            raise FileNotFoundError(
                "Could not find archive folder. Pass --archive-path "
                "(e.g. ~/Downloads/archive)."
            )
    if not archive.is_dir():
        raise FileNotFoundError(f"Archive path not found: {archive}")
    return archive


def split_patients_holdout(
    patient_df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
    n_bins: int = 4,
) -> tuple[list[str], list[str], pd.DataFrame]:
    df = assign_ratio_bins(patient_df, n_bins=n_bins)
    rng = random.Random(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []

    for ratio_bin in sorted(df["ratio_bin"].dropna().unique()):
        bin_patients = df[df["ratio_bin"] == ratio_bin]["patient_id"].tolist()
        rng.shuffle(bin_patients)
        split_at = int(train_ratio * len(bin_patients))
        train_ids.extend(bin_patients[:split_at])
        test_ids.extend(bin_patients[split_at:])

    return train_ids, test_ids, df


def run_holdout(
    archive_path: Path,
    patient_df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    n_bins: int,
) -> None:
    train_ids, test_ids, patient_df = split_patients_holdout(
        patient_df, train_ratio=train_ratio, seed=seed, n_bins=n_bins
    )

    print(f"Train patients: {len(train_ids)}, Test patients: {len(test_ids)}")
    train_summary = summarize_patients(patient_df, train_ids)
    test_summary = summarize_patients(patient_df, test_ids)
    print(
        f"Train IDC ratio: {train_summary['idc_ratio']:.3f}, "
        f"Test IDC ratio: {test_summary['idc_ratio']:.3f}"
    )

    patient_df["split"] = patient_df["patient_id"].map(
        {pid: "train" for pid in train_ids} | {pid: "test" for pid in test_ids}
    )

    manifest = build_patch_manifest(archive_path, patient_df, train_ids, test_ids)
    for split_name in ("train", "test"):
        split_df = manifest[manifest["split"] == split_name].drop(columns=["split"])
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    write_split_metadata(
        archive_path,
        patient_df.drop(columns=["split"], errors="ignore"),
        output_dir,
        mode="holdout",
        seed=seed,
        n_bins=n_bins,
        holdout={"train_ratio": train_ratio, "train": train_summary, "test": test_summary},
    )


def run_cv(
    archive_path: Path,
    patient_df: pd.DataFrame,
    output_dir: Path,
    n_folds: int,
    seed: int,
    n_bins: int,
    skip_existing: bool = False,
) -> None:
    patient_df, folds = create_stratified_patient_folds(
        patient_df, n_splits=n_folds, seed=seed, n_bins=n_bins
    )

    print(f"{n_folds}-fold patient-level CV (StratifiedGroupKFold)")
    for fold in folds:
        print(
            f"  fold {fold['fold']}: "
            f"train {fold['train']['patients']} patients / "
            f"{fold['train']['patches']:,} patches (IDC={fold['train']['idc_ratio']:.3f}), "
            f"test {fold['test']['patients']} patients / "
            f"{fold['test']['patches']:,} patches (IDC={fold['test']['idc_ratio']:.3f})"
        )

    test_ratios = [fold["test"]["idc_ratio"] for fold in folds]
    print(
        f"Test IDC ratio across folds: "
        f"mean={sum(test_ratios)/len(test_ratios):.3f}, "
        f"std={pd.Series(test_ratios).std(ddof=0):.3f}"
    )

    write_cv_fold_manifests(
        archive_path, patient_df, folds, output_dir, skip_existing=skip_existing
    )
    write_split_metadata(
        archive_path,
        patient_df,
        output_dir,
        mode="cv",
        seed=seed,
        n_bins=n_bins,
        folds=folds,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Patient-level IDC-ratio-stratified splits for histopath archive"
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Path to archive/ (default: ./archive or ~/Downloads/archive)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/histopath",
        help="Directory for split CSVs and stats",
    )
    parser.add_argument(
        "--mode",
        choices=("cv", "holdout"),
        default="cv",
        help="cv = StratifiedGroupKFold (for mean±std / Friedman); holdout = single 80/20",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip fold CSV generation when train.csv and test.csv already exist",
    )
    args = parser.parse_args()

    archive_path = resolve_archive_path(args.archive_path)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_df = collect_patient_stats(archive_path)
    print(f"Archive: {archive_path}")
    print(f"Patient folders: {len(patient_df)}")
    print(f"Total patches: {int(patient_df['total'].sum()):,}")

    if args.mode == "cv":
        run_cv(
            archive_path,
            patient_df,
            output_dir,
            args.folds,
            args.seed,
            args.bins,
            skip_existing=args.skip_existing,
        )
        print(f"\nSaved fold manifests under: {output_dir / 'folds'}")
    else:
        run_holdout(
            archive_path, patient_df, output_dir,
            args.train_ratio, args.seed, args.bins,
        )
        print(f"\nSaved holdout manifests: {output_dir / 'train.csv'}, {output_dir / 'test.csv'}")

    print(f"Saved patient stats: {output_dir / 'patient_stats.csv'}")
    print(f"Saved stats: {output_dir / 'split_stats.json'}")


if __name__ == "__main__":
    main()
