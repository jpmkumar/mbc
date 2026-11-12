#!/usr/bin/env python3
"""Freeze Paper C+ BreakHis case-grouped and random-patch five-fold manifests.

Distinct from Paper D: output under data/splits/paper_c_plus/breakhis/ with
seed 32026. Does not overwrite data/splits/breakhis/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = ROOT / "papers" / "paper_d" / "data" / "BreaKHis_v1.tar.gz"
OUT_DIR = ROOT / "data" / "splits" / "paper_c_plus" / "breakhis"
EXPECTED_SHA256 = (
    "d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640"
)

PATH_RE = re.compile(
    r"BreaKHis_v1/histology_slides/breast/"
    r"(?P<label>benign|malignant)/SOB/"
    r"(?P<subtype>[^/]+)/"
    r"(?P<case>[^/]+)/"
    r"(?P<mag>\d+X)/"
    r"(?P<fname>[^/]+\.png)$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_members(archive: Path) -> list[dict]:
    rows: list[dict] = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            match = PATH_RE.search(name)
            if not match:
                continue
            label = match.group("label").lower()
            rows.append(
                {
                    "filepath": name,
                    "label": label,
                    "subtype": match.group("subtype"),
                    "case_id": match.group("case"),
                    "magnification": match.group("mag").upper(),
                    "filename": match.group("fname"),
                    "y": 1 if label == "malignant" else 0,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=32026)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"missing archive: {archive}")

    digest = sha256_file(archive)
    print(f"archive={archive}")
    print(f"sha256={digest}")
    if args.expected_sha256 and digest != args.expected_sha256:
        raise SystemExit(
            f"archive digest mismatch: got {digest}, expected {args.expected_sha256}"
        )

    print("Indexing PNG members...")
    rows = parse_members(archive)
    if not rows:
        raise SystemExit("no BreakHis PNG members matched the expected path pattern")

    cases = sorted({row["case_id"] for row in rows})
    case_label: dict[str, int] = {}
    case_counts = Counter(row["case_id"] for row in rows)
    for row in rows:
        case_label[row["case_id"]] = row["y"]
    for case_id in cases:
        labels = {row["y"] for row in rows if row["case_id"] == case_id}
        if len(labels) != 1:
            raise SystemExit(f"case {case_id} has mixed labels: {labels}")

    y_case = [case_label[case_id] for case_id in cases]
    grouped = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    grouped_fold: dict[str, int] = {}
    dummy = [[0] for _ in cases]
    for fold_i, (_, test_idx) in enumerate(grouped.split(dummy, y_case, cases)):
        for index in test_idx:
            grouped_fold[cases[index]] = fold_i
    if len(grouped_fold) != len(cases):
        raise SystemExit("grouped fold assignment incomplete")

    # Random-patch outer folds: image-level stratified, ignores case grouping.
    image_y = [row["y"] for row in rows]
    random_fold = [-1] * len(rows)
    random_splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    image_index = list(range(len(rows)))
    for fold_i, (_, test_idx) in enumerate(
        random_splitter.split(image_index, image_y)
    ):
        for index in test_idx:
            random_fold[index] = fold_i
    if any(fold < 0 for fold in random_fold):
        raise SystemExit("random fold assignment incomplete")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold_i in range(args.n_splits):
        fold_dir = out_dir / "folds" / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_cases = sorted(c for c, fold in grouped_fold.items() if fold == fold_i)
        train_cases = sorted(c for c, fold in grouped_fold.items() if fold != fold_i)
        for name, case_list in (
            ("test_cases.csv", test_cases),
            ("train_cases.csv", train_cases),
        ):
            path = fold_dir / name
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["case_id", "label", "n_images"])
                for case_id in case_list:
                    writer.writerow(
                        [
                            case_id,
                            "malignant" if case_label[case_id] else "benign",
                            case_counts[case_id],
                        ]
                    )

    manifest_path = out_dir / "image_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filepath",
                "case_id",
                "label",
                "y",
                "subtype",
                "magnification",
                "filename",
                "grouped_outer_fold",
                "random_outer_fold",
            ],
        )
        writer.writeheader()
        # Original parse order so random_fold indices align.
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "filepath": row["filepath"],
                    "case_id": row["case_id"],
                    "label": row["label"],
                    "y": row["y"],
                    "subtype": row["subtype"],
                    "magnification": row["magnification"],
                    "filename": row["filename"],
                    "grouped_outer_fold": grouped_fold[row["case_id"]],
                    "random_outer_fold": random_fold[index],
                }
            )

    fold_summary = []
    for fold_i in range(args.n_splits):
        test_cases = [c for c, fold in grouped_fold.items() if fold == fold_i]
        fold_summary.append(
            {
                "fold": fold_i,
                "n_test_cases": len(test_cases),
                "n_test_benign_cases": sum(1 for c in test_cases if case_label[c] == 0),
                "n_test_malignant_cases": sum(1 for c in test_cases if case_label[c] == 1),
                "n_test_images": sum(case_counts[c] for c in test_cases),
                "n_random_test_images": sum(
                    1 for fold in random_fold if fold == fold_i
                ),
            }
        )

    # Compare overlap with Paper D folds if present.
    paper_d_stats = ROOT / "data" / "splits" / "breakhis" / "split_stats.json"
    paper_d_note = None
    if paper_d_stats.is_file():
        d_stats = json.loads(paper_d_stats.read_text(encoding="utf-8"))
        paper_d_note = {
            "paper_d_seed": d_stats.get("seed"),
            "paper_c_plus_seed": args.seed,
            "same_archive_sha256": d_stats.get("archive_sha256") == digest,
            "note": "C+ uses a distinct seed so case folds differ from Paper D.",
        }

    stats = {
        "paper": "C+",
        "archive": (
            str(archive.relative_to(ROOT))
            if str(archive).startswith(str(ROOT))
            else str(archive)
        ),
        "archive_sha256": digest,
        "n_images": len(rows),
        "n_cases": len(cases),
        "images_by_label": dict(Counter(row["label"] for row in rows)),
        "images_by_magnification": dict(
            sorted(Counter(row["magnification"] for row in rows).items())
        ),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "grouped_splitter": "StratifiedGroupKFold",
        "random_splitter": "StratifiedKFold",
        "group_unit": "case_id",
        "magnification_rule": "pooled_all_mags_primary",
        "fov_co_primary": False,
        "fov_reason": "BreakHis public tiles lack neighbour coordinates",
        "folds": fold_summary,
        "paper_d_relation": paper_d_note,
        "manifest": str(manifest_path.relative_to(ROOT)),
    }
    stats_path = out_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# BreakHis splits (Paper C+)",
                "",
                "Distinct from Paper D (`data/splits/breakhis/`).",
                "",
                f"- Seed: `{args.seed}`",
                f"- Archive SHA-256: `{digest}`",
                "- Grouped: `StratifiedGroupKFold` on case_id",
                "- Random: `StratifiedKFold` on images (protocol-optimism arm)",
                "- Magnification: pooled (40X/100X/200X/400X)",
                "- FOV co-primary: no",
                "",
                "Do not overwrite Paper D lists.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"images={len(rows)} cases={len(cases)}")
    print(f"wrote {stats_path.relative_to(ROOT)}")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
