"""Patient-level stratified splits for IDC histopathology archive data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

EXCLUDED_DIR = "IDC_regular_ps50_idx5"


def _list_patches(cls_dir: Path) -> list[str]:
    if not cls_dir.is_dir():
        return []
    return sorted(
        f.name for f in cls_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )


def collect_patient_stats(archive_path: Path) -> pd.DataFrame:
    archive_path = Path(archive_path)
    patient_ids = sorted(
        name for name in os.listdir(archive_path)
        if (archive_path / name).is_dir() and name != EXCLUDED_DIR
    )

    rows = []
    for pid in patient_ids:
        n0 = len(_list_patches(archive_path / pid / "0"))
        n1 = len(_list_patches(archive_path / pid / "1"))
        total = n0 + n1
        ratio = n1 / total if total > 0 else 0.0
        rows.append({
            "patient_id": pid,
            "n0": n0,
            "n1": n1,
            "total": total,
            "idc_ratio": ratio,
        })
    return pd.DataFrame(rows)


def assign_ratio_bins(patient_df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    df = patient_df.copy()
    df["ratio_bin"] = pd.qcut(
        df["idc_ratio"], q=n_bins, labels=False, duplicates="drop"
    )
    return df


def summarize_patients(patient_df: pd.DataFrame, patient_ids: list[str]) -> dict:
    sub = patient_df[patient_df["patient_id"].isin(patient_ids)]
    total = int(sub["total"].sum())
    n1 = int(sub["n1"].sum())
    return {
        "patients": len(patient_ids),
        "patches": total,
        "n0": int(sub["n0"].sum()),
        "n1": n1,
        "idc_ratio": (n1 / total) if total else 0.0,
        "mean_patient_idc_ratio": float(sub["idc_ratio"].mean()),
    }


def create_stratified_patient_folds(
    patient_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    n_bins: int = 4,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return patient table with ratio bins and per-fold train/test patient lists."""
    df = assign_ratio_bins(patient_df, n_bins=n_bins)
    if df["ratio_bin"].isna().any():
        raise ValueError("Could not assign IDC ratio bins for all patients.")

    x = df[["patient_id"]].to_numpy()
    y = df["ratio_bin"].astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    folds: list[dict] = []
    fold_test_col = pd.Series(index=df.index, dtype="Int64")

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups)):
        train_ids = df.iloc[train_idx]["patient_id"].tolist()
        test_ids = df.iloc[test_idx]["patient_id"].tolist()
        fold_test_col.iloc[test_idx] = fold_idx
        folds.append({
            "fold": fold_idx,
            "train_patient_ids": train_ids,
            "test_patient_ids": test_ids,
            "train": summarize_patients(df, train_ids),
            "test": summarize_patients(df, test_ids),
        })

    df["test_fold"] = fold_test_col.astype("Int64")
    return df, folds


def build_master_patch_index(
    archive_path: Path,
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """Scan the archive once and return one row per patch."""
    archive_path = Path(archive_path)
    records = []
    for row in patient_df.itertuples(index=False):
        patient_id = str(row.patient_id)
        for label, cls in ((0, "0"), (1, "1")):
            cls_dir = archive_path / patient_id / cls
            for fname in _list_patches(cls_dir):
                records.append({
                    "filepath": f"{patient_id}/{cls}/{fname}",
                    "modality": "histopath",
                    "label": label,
                    "patient_id": patient_id,
                })
    return pd.DataFrame(records)


def build_patch_manifest(
    archive_path: Path,
    patient_df: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    patch_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    train_ids = [str(pid) for pid in train_ids]
    test_ids = [str(pid) for pid in test_ids]

    if patch_index is None:
        patch_index = build_master_patch_index(archive_path, patient_df)

    manifest = patch_index.copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["split"] = manifest["patient_id"].map(
        {pid: "train" for pid in train_ids} | {pid: "test" for pid in test_ids}
    )
    return manifest[manifest["split"].notna()].reset_index(drop=True)


def write_cv_fold_manifests(
    archive_path: Path,
    patient_df: pd.DataFrame,
    folds: list[dict],
    output_dir: Path,
    skip_existing: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[dict] = []
    pending_folds = []
    for fold in folds:
        fold_idx = fold["fold"]
        fold_dir = folds_dir / f"fold_{fold_idx}"
        train_path = fold_dir / "train.csv"
        test_path = fold_dir / "test.csv"
        if skip_existing and train_path.exists() and test_path.exists():
            manifest_paths.append({
                "fold": fold_idx,
                "train": str(train_path),
                "test": str(test_path),
            })
            continue
        pending_folds.append(fold)

    patch_index = None
    if pending_folds:
        print(f"Indexing {int(patient_df['total'].sum()):,} patches (one archive scan)...")
        patch_index = build_master_patch_index(archive_path, patient_df)
        print(f"Writing {len(pending_folds)} fold manifest(s)...")

    for fold in pending_folds:
        fold_idx = fold["fold"]
        fold_dir = folds_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        manifest = build_patch_manifest(
            archive_path,
            patient_df,
            fold["train_patient_ids"],
            fold["test_patient_ids"],
            patch_index=patch_index,
        )
        train_path = fold_dir / "train.csv"
        test_path = fold_dir / "test.csv"
        manifest[manifest["split"] == "train"].drop(columns=["split"]).to_csv(
            train_path, index=False
        )
        manifest[manifest["split"] == "test"].drop(columns=["split"]).to_csv(
            test_path, index=False
        )
        manifest_paths.append({
            "fold": fold_idx,
            "train": str(train_path),
            "test": str(test_path),
        })
        print(f"  saved fold {fold_idx}: {train_path.name}, {test_path.name}")

    manifest_paths.sort(key=lambda item: item["fold"])
    return {"folds": manifest_paths}


EXCLUDED_DIR = "IDC_regular_ps50_idx5"
DEFAULT_ARCHIVE_CANDIDATES = (
    Path("archive"),
    Path.home() / "Downloads" / "archive",
    Path.home() / "Downloads" / "Histopathology-dataset",
)


def discover_archive_path() -> Path | None:
    """Find IDC archive under common cloud notebook mount points."""
    search_roots = [
        Path("/kaggle/input"),
        Path("/content"),
        Path.cwd(),
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for current, dirnames, _ in os.walk(root):
            current_path = Path(current)
            if EXCLUDED_DIR in dirnames:
                return current_path
            if "10253" in dirnames and (current_path / "10253" / "0").is_dir():
                return current_path
    return None


def resolve_archive_path(
    archive_path: str | Path | None = None,
    splits_dir: str | Path | None = None,
) -> Path:
    if archive_path:
        path = Path(archive_path).expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"Archive path not found: {path}")

    if splits_dir is not None:
        stats_path = Path(splits_dir) / "split_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            saved = Path(stats["archive_path"])
            if saved.is_dir():
                return saved.resolve()

    discovered = discover_archive_path()
    if discovered is not None:
        return discovered.resolve()

    for candidate in DEFAULT_ARCHIVE_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate histopath archive. Pass --archive-path or regenerate splits."
    )


def split_train_val_patients(
    patient_df: pd.DataFrame,
    train_patient_ids: list[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Hold out a patient-level validation set from fold train patients."""
    import random

    sub = patient_df.copy()
    sub["patient_id"] = sub["patient_id"].astype(str)
    sub = sub[sub["patient_id"].isin([str(pid) for pid in train_patient_ids])].copy()
    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []

    for ratio_bin in sorted(sub["ratio_bin"].dropna().unique()):
        bin_patients = sub[sub["ratio_bin"] == ratio_bin]["patient_id"].tolist()
        rng.shuffle(bin_patients)
        if len(bin_patients) == 1:
            train_ids.extend(bin_patients)
            continue
        split_at = max(1, int((1 - val_ratio) * len(bin_patients)))
        if split_at >= len(bin_patients):
            split_at = len(bin_patients) - 1
        train_ids.extend(bin_patients[:split_at])
        val_ids.extend(bin_patients[split_at:])

    return train_ids, val_ids


def write_fold_split_manifests(
    archive_path: Path,
    patient_df: pd.DataFrame,
    fold_dir: Path,
    train_patient_ids: list[str],
    val_patient_ids: list[str],
    test_patient_ids: list[str],
) -> dict[str, Path]:
    split_map = (
        {pid: "train" for pid in train_patient_ids}
        | {pid: "val" for pid in val_patient_ids}
        | {pid: "test" for pid in test_patient_ids}
    )
    records = []
    for row in patient_df.itertuples(index=False):
        patient_id = str(row.patient_id)
        split = split_map.get(patient_id)
        if split is None:
            continue
        for label, cls in ((0, "0"), (1, "1")):
            cls_dir = Path(archive_path) / patient_id / cls
            for fname in _list_patches(cls_dir):
                records.append({
                    "filepath": f"{patient_id}/{cls}/{fname}",
                    "modality": "histopath",
                    "label": label,
                    "patient_id": patient_id,
                    "split": split,
                })

    manifest = pd.DataFrame(records)
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name in ("train", "val", "test"):
        out = fold_dir / f"{split_name}.csv"
        manifest[manifest["split"] == split_name].drop(columns=["split"]).to_csv(
            out, index=False
        )
        paths[split_name] = out
    return paths


def load_histopath_folds(splits_dir: str | Path) -> list[dict[str, str]]:
    splits_dir = Path(splits_dir)
    folds_dir = splits_dir / "folds"
    if not folds_dir.exists():
        raise FileNotFoundError(
            f"No k-fold manifests found under {folds_dir}. "
            "Run split_histopath_archive.py --mode cv first."
        )

    fold_paths = []
    for fold_dir in sorted(folds_dir.glob("fold_*")):
        fold_idx = int(fold_dir.name.split("_")[-1])
        fold_paths.append({
            "fold": fold_idx,
            "train": str(fold_dir / "train.csv"),
            "test": str(fold_dir / "test.csv"),
        })
    return fold_paths


def write_split_metadata(
    archive_path: Path,
    patient_df: pd.DataFrame,
    output_dir: Path,
    *,
    mode: str,
    seed: int,
    n_bins: int,
    folds: list[dict] | None = None,
    holdout: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_df.to_csv(output_dir / "patient_stats.csv", index=False)

    stats = {
        "archive_path": str(Path(archive_path).resolve()),
        "mode": mode,
        "seed": seed,
        "ratio_bins": n_bins,
        "total_patients": len(patient_df),
        "total_patches": int(patient_df["total"].sum()),
    }

    if mode == "cv" and folds is not None:
        stats["n_folds"] = len(folds)
        stats["folds"] = [
            {"fold": fold["fold"], "train": fold["train"], "test": fold["test"]}
            for fold in folds
        ]
        ratio_spread = [fold["test"]["idc_ratio"] for fold in folds]
        stats["test_idc_ratio_mean"] = float(sum(ratio_spread) / len(ratio_spread))
        stats["test_idc_ratio_std"] = float(pd.Series(ratio_spread).std(ddof=0))
    elif holdout is not None:
        stats.update(holdout)

    stats_path = output_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats_path
