#!/usr/bin/env python3
"""Write preregistration/paper_d_lock.json (Paper D Stage-2 lock manifest).

Run after case-ID manifests exist under data/splits/breakhis/ and before any
labelled comparative BreakHis run. Digests are over exact file bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "preregistration" / "paper_d_breakhis_protocol.md"
REGISTRY = ROOT / "papers" / "paper_d" / "config" / "circuit_registry.json"
SPLITS = ROOT / "data" / "splits" / "breakhis"
OUT = ROOT / "preregistration" / "paper_d_lock.json"

REQUIRED_TRACKED = [
    PROTOCOL,
    REGISTRY,
    ROOT / "papers" / "paper_d" / "README.md",
    ROOT / "papers" / "paper_d" / "CODE_AVAILABILITY.md",
    ROOT / "papers" / "paper_d" / "PLAN.md",
    SPLITS / "README.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def collect_split_artifacts() -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if not SPLITS.is_dir():
        return artifacts
    for path in sorted(SPLITS.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "README.md":
            continue
        # Case lists and stats only — skip large rebuildable CSVs if present.
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        artifacts[rel] = sha256_file(path)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-empty-splits",
        action="store_true",
        help="Write a partial lock before BreakHis case lists exist (Phase 0 only).",
    )
    parser.add_argument(
        "--archive-sha256",
        default="",
        help="SHA-256 of the BreakHis archive bytes (required for confirmatory lock).",
    )
    parser.add_argument(
        "--container-image-digest",
        default="",
        help="Container image digest used for confirmatory runs, if any.",
    )
    parser.add_argument(
        "--dependency-lock-sha256",
        default="",
        help="SHA-256 of the pip/conda lock file used for confirmatory runs.",
    )
    args = parser.parse_args()

    missing = [p for p in REQUIRED_TRACKED if not p.is_file()]
    if missing:
        for p in missing:
            print(f"missing required file: {p.relative_to(ROOT)}", file=sys.stderr)
        return 1

    split_artifacts = collect_split_artifacts()
    if not split_artifacts and not args.allow_empty_splits:
        print(
            "no BreakHis split artifacts under data/splits/breakhis/ "
            "(*.csv / *.json). Pass --allow-empty-splits for a Phase-0 stub, "
            "or freeze case lists first.",
            file=sys.stderr,
        )
        return 1

    if not args.allow_empty_splits and not args.archive_sha256:
        print(
            "confirmatory lock requires --archive-sha256 <BreakHis archive digest>",
            file=sys.stderr,
        )
        return 1

    tracked = {
        p.relative_to(ROOT).as_posix(): sha256_file(p) for p in REQUIRED_TRACKED
    }

    manifest = {
        "paper": "D",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage1_commit": git_commit(),
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": tracked[PROTOCOL.relative_to(ROOT).as_posix()],
        "tracked_file_sha256": tracked,
        "split_artifact_sha256": split_artifacts,
        "breakhis_archive_sha256": args.archive_sha256 or None,
        "container_image_digest": args.container_image_digest or None,
        "dependency_lock_sha256": args.dependency_lock_sha256 or None,
        "partial_lock": bool(args.allow_empty_splits and not split_artifacts),
        "notes": (
            "partial Phase-0 stub; re-run without --allow-empty-splits after "
            "case lists and archive digest are available"
            if args.allow_empty_splits and not split_artifacts
            else "confirmatory lock candidate — verify before labelled runs"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"protocol_sha256={manifest['protocol_sha256']}")
    print(f"split_artifacts={len(split_artifacts)}")
    print(f"partial_lock={manifest['partial_lock']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
