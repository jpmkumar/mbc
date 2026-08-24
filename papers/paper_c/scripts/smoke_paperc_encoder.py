#!/usr/bin/env python3
"""Qualify one pinned encoder on real IDC images without using labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from extract_embeddings import (
    DEFAULT_REGISTRY,
    build_patch_index,
    load_encoder,
    registry_entry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--archive-path", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--hf-cache", type=Path, default=Path("/cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for server qualification.")
    device = torch.device("cuda")
    spec, registry_sha = registry_entry(args.registry, args.encoder)
    bundle = load_encoder(spec, device, args.hf_cache)
    bundle.model.eval().to(device)
    batch_size = args.batch_size or int(spec["batch_size_a4000"])

    index = build_patch_index(args.archive_path)
    images = []
    for row in index[:batch_size]:
        with Image.open(args.archive_path / row["filepath"]) as image:
            images.append(bundle.transform(image.convert("RGB")))
    batch = torch.stack(images).to(device)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = bundle.forward(batch)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(output).all())
    expected = (batch_size, int(spec["expected_dim"]))
    status = "PASS" if finite and tuple(output.shape) == expected else "FAIL"
    report = {
        "status": status,
        "encoder": args.encoder,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "registry_sha256": registry_sha,
        "batch_size": batch_size,
        "output_shape": list(output.shape),
        "expected_shape": list(expected),
        "finite": finite,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "artifact_sha256": bundle.artifact_hashes,
    }
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit("Encoder qualification failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
