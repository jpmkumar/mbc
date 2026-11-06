#!/usr/bin/env python3
"""Write preregistration/paper_c_lock.json (Paper C Stage-2 lock manifest).

Run after Stage-1 commit (protocol status LOCKED) and before any labelled
comparative probe. Digests are over exact file bytes. Server-generated
artifacts are recorded by absolute path via --server-artifact PATH=SHA256.
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
PROTOCOL = ROOT / "preregistration" / "paper_c_protocol.md"
REGISTRY = ROOT / "papers" / "paper_c" / "config" / "model_registry.json"
INNER_SUMMARY = ROOT / "data" / "splits" / "paper_c" / "idc" / "inner_split_summary.json"
OUT = ROOT / "preregistration" / "paper_c_lock.json"

REQUIRED_TRACKED = [
    PROTOCOL,
    REGISTRY,
    INNER_SUMMARY,
    ROOT / "papers" / "paper_c" / "PLAN.md",
    ROOT / "papers" / "paper_c" / "SERVER_PAPER_C.md",
    ROOT / "papers" / "paper_c" / "scripts" / "extract_embeddings.py",
    ROOT / "papers" / "paper_c" / "scripts" / "build_bcss_centres.py",
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


def collect_idc_inner_artifacts() -> dict[str, str]:
    artifacts: dict[str, str] = {}
    splits = ROOT / "data" / "splits" / "paper_c" / "idc"
    if not splits.is_dir():
        return artifacts
    for path in sorted(splits.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        artifacts[rel] = sha256_file(path)
    return artifacts


def parse_kv_sha(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected PATH=SHA256, got: {item}")
        path, digest = item.split("=", 1)
        path = path.strip()
        digest = digest.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SystemExit(f"invalid sha256 for {path}: {digest}")
        out[path] = digest
    return out


def encoder_revisions(registry: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, cfg in registry.items():
        out[name] = {
            "status": cfg.get("status"),
            "provider": cfg.get("provider"),
            "repo_id": cfg.get("repo_id"),
            "revision": cfg.get("revision"),
            "weights": cfg.get("weights"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idc-archive-sha256",
        default="",
        help="SHA-256 of the IDC dataset archive bytes.",
    )
    parser.add_argument(
        "--container-image-id",
        default="",
        help="Docker image id / digest used for confirmatory runs.",
    )
    parser.add_argument(
        "--dependency-lock-sha256",
        default="",
        help="SHA-256 of docker/requirements-paperc.lock.",
    )
    parser.add_argument(
        "--server-artifact",
        action="append",
        default=[],
        help="Absolute server path=sha256 (repeatable).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing archive/image/server digests (engineering only).",
    )
    args = parser.parse_args()

    missing = [p for p in REQUIRED_TRACKED if not p.is_file()]
    if missing:
        for p in missing:
            print(f"missing required file: {p.relative_to(ROOT)}", file=sys.stderr)
        return 1

    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    status_line = next(
        (ln for ln in protocol_text.splitlines() if ln.startswith("**Status:**")),
        "",
    )
    if "LOCKED" not in status_line:
        print(
            "protocol status is not LOCKED; flip Stage 1 before writing lock",
            file=sys.stderr,
        )
        return 1

    tracked = {
        p.relative_to(ROOT).as_posix(): sha256_file(p) for p in REQUIRED_TRACKED
    }
    split_artifacts = collect_idc_inner_artifacts()
    server_artifacts = parse_kv_sha(args.server_artifact)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    partial = bool(
        args.allow_partial
        or not args.idc_archive_sha256
        or not args.container_image_id
        or not args.dependency_lock_sha256
        or not server_artifacts
    )
    if partial and not args.allow_partial:
        print(
            "confirmatory lock needs --idc-archive-sha256, --container-image-id, "
            "--dependency-lock-sha256, and at least one --server-artifact; "
            "pass --allow-partial for an engineering stub",
            file=sys.stderr,
        )
        return 1

    manifest = {
        "paper": "C",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage1_commit": git_commit(),
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": tracked[PROTOCOL.relative_to(ROOT).as_posix()],
        "tracked_file_sha256": tracked,
        "split_artifact_sha256": split_artifacts,
        "server_artifact_sha256": server_artifacts,
        "encoder_revisions": encoder_revisions(registry),
        "idc_archive_sha256": args.idc_archive_sha256 or None,
        "container_image_id": args.container_image_id or None,
        "dependency_lock_sha256": args.dependency_lock_sha256 or None,
        "partial_lock": partial,
        "notes": (
            "partial engineering stub; rewrite without --allow-partial before "
            "labelled probes"
            if partial
            else "confirmatory lock — verify before labelled runs; "
            "label-blind embedding extraction may proceed"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"protocol_sha256={manifest['protocol_sha256']}")
    print(f"split_artifacts={len(split_artifacts)}")
    print(f"server_artifacts={len(server_artifacts)}")
    print(f"partial_lock={manifest['partial_lock']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
