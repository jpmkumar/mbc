#!/usr/bin/env python3
"""Verify a Paper C embedding cache is complete and row-aligned."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--expect-n", type=int, default=0)
    args = parser.parse_args()

    cache = args.cache_dir
    errors: list[str] = []
    prov_path = cache / "provenance.json"
    index_path = cache / "index.csv"
    emb_path = cache / "embeddings.npy"
    shards_path = cache / "shards_done.json"

    if not prov_path.is_file():
        errors.append("missing provenance.json")
    if not index_path.is_file():
        errors.append("missing index.csv")
    if not emb_path.is_file():
        errors.append("missing embeddings.npy")
    if errors:
        print("CACHE VERIFY FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    with index_path.open(encoding="utf-8") as f:
        n_index = sum(1 for _ in f) - 1
    emb = np.load(emb_path, mmap_mode="r")
    n_emb = int(emb.shape[0])
    n_prov = int(prov.get("n_centres") or 0)
    expect = args.expect_n or n_prov

    if not prov.get("complete"):
        errors.append(f"provenance.complete is {prov.get('complete')!r}")
    if n_index != expect:
        errors.append(f"index rows={n_index}, expected={expect}")
    if n_emb != expect:
        errors.append(f"embeddings rows={n_emb}, expected={expect}")
    if n_prov and n_prov != expect:
        errors.append(f"provenance n_centres={n_prov}, expected={expect}")
    if shards_path.is_file():
        shards = json.loads(shards_path.read_text(encoding="utf-8"))
        completed = shards.get("completed") or {}
        # full run: ceil(n/shard_size) shards; just report count
        print(f"shards_completed={len(completed)}")

    if errors:
        print("CACHE VERIFY FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("CACHE VERIFY OK")
    print(f"cache={cache}")
    print(f"n={expect}")
    print(f"encoder={prov.get('encoder')}")
    print(f"transform={prov.get('transform') or prov.get('context_k')}")
    print(f"dim={emb.shape[1] if emb.ndim == 2 else emb.shape}")
    print(f"dtype={emb.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
