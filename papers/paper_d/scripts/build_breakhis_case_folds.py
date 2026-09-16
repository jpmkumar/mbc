#!/usr/bin/env python3
"""Audit BreakHis archive and freeze case-ID five-fold manifests (Paper D).

Builds from the tarball listing (no extract required for split freeze).
Does not train or score any head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = ROOT / "papers" / "paper_d" / "data" / "BreaKHis_v1.tar.gz"
OUT_DIR = ROOT / "data" / "splits" / "breakhis"
AUDIT_OUT = ROOT / "papers" / "paper_d" / "results" / "breakhis_archive_audit.md"

# .../benign|malignant/SOB/<subtype>/<case_folder>/<mag>/<file>.png
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_members(archive: Path) -> list[dict]:
    rows: list[dict] = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            m = PATH_RE.search(name)
            if not m:
                continue
            rows.append(
                {
                    "path": name,
                    "label": m.group("label").lower(),
                    "subtype": m.group("subtype"),
                    "case_id": m.group("case"),
                    "magnification": m.group("mag").upper(),
                    "filename": m.group("fname"),
                    "y": 1 if m.group("label").lower() == "malignant" else 0,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-sha256",
        default="",
        help="If set, refuse to proceed when archive digest differs.",
    )
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

    print("Indexing PNG members (may take a few minutes)...")
    rows = parse_members(archive)
    if not rows:
        raise SystemExit("no BreakHis PNG members matched the expected path pattern")

    n_images = len(rows)
    cases = sorted({r["case_id"] for r in rows})
    n_cases = len(cases)
    by_label = Counter(r["label"] for r in rows)
    by_mag = Counter(r["magnification"] for r in rows)
    case_label = {}
    case_counts = Counter(r["case_id"] for r in rows)
    for r in rows:
        case_label[r["case_id"]] = r["y"]
    # Ambiguous case labels should not occur in BreakHis.
    for cid in cases:
        ys = {r["y"] for r in rows if r["case_id"] == cid}
        if len(ys) != 1:
            raise SystemExit(f"case {cid} has mixed labels: {ys}")

    print(f"images={n_images} cases={n_cases}")
    print(f"labels={dict(by_label)}")
    print(f"mags={dict(sorted(by_mag.items()))}")

    # One row per case for grouping CV; stratify on case-level binary label.
    case_ids = cases
    y_case = [case_label[c] for c in case_ids]
    groups = case_ids  # group == case

    sgkf = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    # sklearn wants X array-like; dummy features fine.
    X = [[0] for _ in case_ids]
    fold_assign = {}
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y_case, groups)):
        for j in test_idx:
            fold_assign[case_ids[j]] = fold_i

    if len(fold_assign) != n_cases:
        raise SystemExit("fold assignment incomplete")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Per-fold case lists (test fold membership), matching histopath style.
    for fold_i in range(args.n_splits):
        fold_dir = OUT_DIR / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_cases = sorted(c for c, f in fold_assign.items() if f == fold_i)
        train_cases = sorted(c for c, f in fold_assign.items() if f != fold_i)
        for name, clist in (("test_cases.csv", test_cases), ("train_cases.csv", train_cases)):
            path = fold_dir / name
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["case_id", "label", "n_images"])
                for c in clist:
                    w.writerow([c, "malignant" if case_label[c] else "benign", case_counts[c]])

    # Image-level inventory (rebuildable; local under papers/paper_d/results/).
    inventory = (
        ROOT / "papers" / "paper_d" / "results" / "breakhis_image_inventory.csv"
    )
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "case_id",
                "label",
                "subtype",
                "magnification",
                "filename",
                "fold",
            ],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["path"]):
            w.writerow(
                {
                    **{
                        k: r[k]
                        for k in (
                            "path",
                            "case_id",
                            "label",
                            "subtype",
                            "magnification",
                            "filename",
                        )
                    },
                    "fold": fold_assign[r["case_id"]],
                }
            )

    fold_summary = []
    for fold_i in range(args.n_splits):
        test_c = [c for c, f in fold_assign.items() if f == fold_i]
        n_mal = sum(case_label[c] for c in test_c)
        n_ben = len(test_c) - n_mal
        n_img = sum(case_counts[c] for c in test_c)
        fold_summary.append(
            {
                "fold": fold_i,
                "n_test_cases": len(test_c),
                "n_test_benign_cases": n_ben,
                "n_test_malignant_cases": n_mal,
                "n_test_images": n_img,
            }
        )

    stats = {
        "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": digest,
        "n_images": n_images,
        "n_cases": n_cases,
        "images_by_label": dict(by_label),
        "images_by_magnification": dict(sorted(by_mag.items())),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "sklearn_splitter": "StratifiedGroupKFold",
        "group_unit": "case_id",
        "folds": fold_summary,
        "case_lists_glob": "data/splits/breakhis/fold_*/{train,test}_cases.csv",
    }
    stats_path = OUT_DIR / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BreakHis archive audit (Paper D)",
        "",
        f"- Archive: `{stats['archive']}`",
        f"- SHA-256: `{digest}`",
        f"- Images: **{n_images}**",
        f"- Cases: **{n_cases}**",
        f"- Labels (images): {dict(by_label)}",
        f"- Magnifications: {dict(sorted(by_mag.items()))}",
        f"- Splitter: StratifiedGroupKFold(n_splits={args.n_splits}, seed={args.seed})",
        "",
        "| Fold | Test cases | Benign cases | Malignant cases | Test images |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for fs in fold_summary:
        lines.append(
            f"| {fs['fold']} | {fs['n_test_cases']} | {fs['n_test_benign_cases']} | "
            f"{fs['n_test_malignant_cases']} | {fs['n_test_images']} |"
        )
    lines.append("")
    lines.append("Case lists written under `data/splits/breakhis/fold_*/`.")
    lines.append("")
    AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {stats_path.relative_to(ROOT)}")
    print(f"wrote {AUDIT_OUT.relative_to(ROOT)}")
    print("fold case lists ready under data/splits/breakhis/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
