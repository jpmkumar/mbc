#!/usr/bin/env python3
"""Label-blind smoke: load DINOv2/ResNet and embed a few BreakHis tiles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "papers" / "paper_c" / "scripts"))

from extract_embeddings import load_encoder, registry_entry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        required=True,
        help="Path containing histology_slides/ (BreaKHis_v1 root).",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", default="dinov2_vitl14")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "papers" / "paper_c" / "config" / "model_registry.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/paperc_plus_smoke_cache"),
    )
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: args.n]
    if len(rows) < args.n:
        raise SystemExit(f"manifest has only {len(rows)} rows")

    spec, _registry_sha = registry_entry(args.registry, args.encoder)
    if spec.get("status") == "unavailable":
        raise SystemExit(f"encoder unavailable: {args.encoder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_encoder(spec, device=device, cache_dir=args.cache_dir)
    bundle.model.to(device).eval()

    root = args.extracted_root
    if (root / "histology_slides").is_dir():
        slide_root = root
    elif (root / "BreaKHis_v1" / "histology_slides").is_dir():
        slide_root = root / "BreaKHis_v1"
    else:
        raise SystemExit(f"cannot find histology_slides under {root}")

    batch = []
    paths = []
    for row in rows:
        rel = row["filepath"]
        if rel.startswith("BreaKHis_v1/"):
            rel = rel[len("BreaKHis_v1/") :]
        path = slide_root / rel
        if not path.is_file():
            raise SystemExit(f"missing tile: {path}")
        image = Image.open(path).convert("RGB")
        batch.append(bundle.transform(image))
        paths.append(str(path))

    tensor = torch.stack(batch).to(device)
    with torch.inference_mode():
        emb = bundle.forward(tensor)
    print(
        json.dumps(
            {
                "encoder": args.encoder,
                "device": str(device),
                "n": len(paths),
                "embedding_shape": list(emb.shape),
                "embedding_dtype": str(emb.dtype),
                "expected_dim": spec.get("expected_dim"),
                "sample_paths": paths[:3],
                "ok": True,
            },
            indent=2,
        )
    )
    if int(emb.shape[-1]) != int(spec["expected_dim"]):
        raise SystemExit(
            f"dim mismatch: got {emb.shape[-1]}, expected {spec['expected_dim']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
