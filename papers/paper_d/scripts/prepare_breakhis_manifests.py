#!/usr/bin/env python3
"""Build BreakHis image manifests for one outer fold (train/val/test).

Reads frozen case lists under data/splits/breakhis/ and the extracted archive
under papers/paper_d/data/BreaKHis_v1/. Writes CSVs compatible with
UnifiedBreastDataset (filepath, modality, label, patient_id).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTRACTED = ROOT / "papers" / "paper_d" / "data" / "BreaKHis_v1"
DEFAULT_SPLITS = ROOT / "data" / "splits" / "breakhis"
PATH_RE = re.compile(
    r"histology_slides/breast/(?P<label>benign|malignant)/SOB/"
    r"(?P<subtype>[^/]+)/(?P<case>[^/]+)/(?P<mag>\d+X)/(?P<fname>[^/]+\.png)$",
    re.IGNORECASE,
)


def index_archive(extracted: Path) -> pd.DataFrame:
    rows = []
    for png in sorted(extracted.rglob("*.png")):
        rel = png.relative_to(extracted).as_posix()
        m = PATH_RE.search(rel)
        if not m:
            continue
        rows.append(
            {
                "filepath": f"BreaKHis_v1/{rel}",
                "modality": "histopath",
                "label": 1 if m.group("label").lower() == "malignant" else 0,
                "patient_id": m.group("case"),
                "subtype": m.group("subtype"),
                "magnification": m.group("mag").upper(),
            }
        )
    if not rows:
        raise SystemExit(f"no PNGs matched under {extracted}")
    return pd.DataFrame(rows)


def load_case_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_val_cases(
    train_cases: pd.DataFrame, val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    ids = train_cases["case_id"].astype(str).tolist()
    y = (train_cases["label"].astype(str) == "malignant").astype(int).to_numpy()
    if len(ids) < 4:
        raise SystemExit("too few train cases to form a validation holdout")
    tr, va = train_test_split(
        ids, test_size=val_ratio, random_state=seed, stratify=y
    )
    return sorted(tr), sorted(va)


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "modality", "label", "patient_id", "subtype", "magnification"]
    df[cols].to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--val-case-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/splits/breakhis/runtime/fold_{N}",
    )
    args = parser.parse_args()

    fold_dir = args.splits_dir / f"fold_{args.fold}"
    train_cases = load_case_table(fold_dir / "train_cases.csv")
    test_cases = load_case_table(fold_dir / "test_cases.csv")

    inventory = index_archive(args.extracted)
    print(
        f"indexed images={len(inventory)} cases={inventory['patient_id'].nunique()}"
    )

    tr_ids, va_ids = split_val_cases(
        train_cases, args.val_case_ratio, seed=args.seed + args.fold
    )
    te_ids = test_cases["case_id"].astype(str).tolist()

    # Leakage guards
    assert not set(tr_ids) & set(va_ids)
    assert not set(tr_ids) & set(te_ids)
    assert not set(va_ids) & set(te_ids)

    out = args.out_dir or (args.splits_dir / "runtime" / f"fold_{args.fold}")
    out = out.resolve()
    write_manifest(inventory[inventory["patient_id"].isin(tr_ids)], out / "train.csv")
    write_manifest(inventory[inventory["patient_id"].isin(va_ids)], out / "val.csv")
    write_manifest(inventory[inventory["patient_id"].isin(te_ids)], out / "test.csv")

    meta = {
        "fold": args.fold,
        "n_train_cases": len(tr_ids),
        "n_val_cases": len(va_ids),
        "n_test_cases": len(te_ids),
        "n_train_images": int((inventory["patient_id"].isin(tr_ids)).sum()),
        "n_val_images": int((inventory["patient_id"].isin(va_ids)).sum()),
        "n_test_images": int((inventory["patient_id"].isin(te_ids)).sum()),
    }
    print(meta)
    (out / "split_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote manifests under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
