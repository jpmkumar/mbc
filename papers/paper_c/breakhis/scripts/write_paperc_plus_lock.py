#!/usr/bin/env python3
"""Write preregistration/paper_c_plus_lock.json after Stage-1 LOCKED commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "preregistration" / "paper_c_plus_protocol.md"
SPLITS = ROOT / "data" / "splits" / "paper_c_plus" / "breakhis"
OUT = ROOT / "preregistration" / "paper_c_plus_lock.json"
REGISTRY = ROOT / "papers" / "paper_c" / "config" / "model_registry.json"

REQUIRED = [
    PROTOCOL,
    REGISTRY,
    SPLITS / "split_stats.json",
    SPLITS / "image_manifest.csv",
    SPLITS / "inner_split_summary.json",
    ROOT / "papers" / "paper_c_plus" / "PLAN.md",
    ROOT / "papers" / "paper_c_plus" / "scripts" / "extract_breakhis_embeddings.py",
    ROOT / "papers" / "paper_c_plus" / "scripts" / "run_breakhis_mlp_cv.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--breakhis-archive-sha256",
        default="d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640",
    )
    parser.add_argument("--container-image-id", default="")
    parser.add_argument("--dependency-lock-sha256", default="")
    args = parser.parse_args()

    missing = [p for p in REQUIRED if not p.is_file()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1

    status = next(
        (
            line
            for line in PROTOCOL.read_text(encoding="utf-8").splitlines()
            if line.startswith("**Status:**")
        ),
        "",
    )
    if "LOCKED" not in status:
        print("protocol is not LOCKED", file=sys.stderr)
        return 1

    tracked = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in REQUIRED}
    split_artifacts = {}
    for path in sorted(SPLITS.rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            split_artifacts[path.relative_to(ROOT).as_posix()] = sha256_file(path)

    manifest = {
        "paper": "C+",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage1_commit": git_commit(),
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": tracked[PROTOCOL.relative_to(ROOT).as_posix()],
        "tracked_file_sha256": tracked,
        "split_artifact_sha256": split_artifacts,
        "breakhis_archive_sha256": args.breakhis_archive_sha256,
        "container_image_id": args.container_image_id or None,
        "dependency_lock_sha256": args.dependency_lock_sha256 or None,
        "confirmatory_head": "F-MLP",
        "notes": "Option B confirmatory F-MLP; FOV not co-primary on BreakHis",
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"protocol_sha256={manifest['protocol_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
