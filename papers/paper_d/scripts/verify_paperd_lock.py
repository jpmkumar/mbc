#!/usr/bin/env python3
"""Recompute digests in preregistration/paper_d_lock.json and abort on mismatch.

Run from a clean checkout of the Stage-1 commit before the first labelled
BreakHis comparative run. Any mismatch voids the lock.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "preregistration" / "paper_d_lock.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not LOCK.is_file():
        print(f"missing lock manifest: {LOCK.relative_to(ROOT)}", file=sys.stderr)
        print("run papers/paper_d/scripts/write_paperd_lock.py first", file=sys.stderr)
        return 1

    manifest = json.loads(LOCK.read_text(encoding="utf-8"))
    errors: list[str] = []

    if manifest.get("partial_lock"):
        errors.append(
            "lock is marked partial_lock=true; freeze splits + archive digest "
            "and rewrite before labelled runs"
        )

    for rel, expected in manifest.get("tracked_file_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing tracked file: {rel}")
            continue
        got = sha256_file(path)
        if got != expected:
            errors.append(f"digest mismatch {rel}: got {got}, expected {expected}")

    for rel, expected in manifest.get("split_artifact_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing split artifact: {rel}")
            continue
        got = sha256_file(path)
        if got != expected:
            errors.append(f"digest mismatch {rel}: got {got}, expected {expected}")

    protocol_rel = manifest.get("protocol_path")
    protocol_expected = manifest.get("protocol_sha256")
    if protocol_rel and protocol_expected:
        path = ROOT / protocol_rel
        if path.is_file():
            got = sha256_file(path)
            if got != protocol_expected:
                errors.append(
                    f"protocol digest mismatch: got {got}, expected {protocol_expected}"
                )

    if errors:
        print("LOCK VERIFY FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("LOCK VERIFY OK")
    print(f"stage1_commit={manifest.get('stage1_commit')}")
    print(f"protocol_sha256={manifest.get('protocol_sha256')}")
    print(f"split_artifacts={len(manifest.get('split_artifact_sha256') or {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
