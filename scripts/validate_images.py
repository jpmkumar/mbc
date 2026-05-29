#!/usr/bin/env python3
"""Find corrupt or unreadable images under data/processed/."""

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate(root: Path) -> list[Path]:
    bad = []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in exts:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.convert("RGB")
        except Exception:
            bad.append(path)
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/processed/mammo",
        help="Folder to scan (default: data/processed/mammo)",
    )
    parser.add_argument("--delete", action="store_true", help="Delete corrupt files")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    bad = validate(root)
    print(f"Scanned {root}")
    print(f"Corrupt/unreadable: {len(bad)}")
    for p in bad[:20]:
        print(" ", p, p.stat().st_size if p.exists() else "missing", "bytes")
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more")

    if args.delete and bad:
        for p in bad:
            p.unlink(missing_ok=True)
        print(f"Deleted {len(bad)} files. Re-run: python data/download/setup_datasets.py")


if __name__ == "__main__":
    main()
