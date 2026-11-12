#!/usr/bin/env python3
"""Extract frozen Paper C+ BreakHis embeddings (upsample to 224, no mosaics)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "papers" / "paper_c" / "scripts"))

from extract_embeddings import load_encoder, registry_entry, sha256  # noqa: E402


class BreakHisDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, transform):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        rel = self.rows[index]["filepath"]
        if rel.startswith("BreaKHis_v1/"):
            rel = rel[len("BreaKHis_v1/") :]
        path = self.root / rel
        image = Image.open(path).convert("RGB")
        return self.transform(image), index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "papers" / "paper_c" / "config" / "model_registry.json",
    )
    parser.add_argument("--hf-cache", type=Path, default=Path("/cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("MBC_DATASET_SHA"):
        raise SystemExit("Full extraction requires MBC_DATASET_SHA")

    root = args.extracted_root
    if (root / "histology_slides").is_dir():
        slide_root = root
    elif (root / "BreaKHis_v1" / "histology_slides").is_dir():
        slide_root = root / "BreaKHis_v1"
    else:
        raise SystemExit(f"histology_slides not found under {root}")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7909:
        raise SystemExit(f"expected 7909 BreakHis rows, found {len(rows)}")

    spec, registry_sha = registry_entry(args.registry, args.encoder)
    if spec.get("status") == "unavailable":
        raise SystemExit(f"encoder unavailable: {args.encoder}")

    out = args.output_dir / f"breakhis_{args.encoder}_upsample224"
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / "embeddings.npy"
    index_path = out / "index.csv"
    provenance_path = out / "provenance.json"
    state_path = out / "shards_done.json"

    if provenance_path.exists() and json.loads(provenance_path.read_text()).get("complete"):
        raise SystemExit(f"completed cache exists: {out}")

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    index_sha = sha256(index_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_encoder(spec, device, args.hf_cache)
    bundle.model.eval().to(device)
    batch_size = args.batch_size or int(spec["batch_size_a4000"])
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        dim = int(bundle.forward(probe).shape[1])
    if dim != int(spec["expected_dim"]):
        raise SystemExit(f"dim {dim} != expected {spec['expected_dim']}")

    storage_dtype = np.float32 if args.fp32 else np.float16
    request = {
        "paper": "C+",
        "cohort": "BreakHis",
        "encoder": args.encoder,
        "transform": "upsample224",
        "n_centres": len(rows),
        "embedding_dim": dim,
        "dtype": np.dtype(storage_dtype).name,
        "compute_precision": "fp32" if args.fp32 else "autocast-fp16",
        "model_registry_sha256": registry_sha,
        "index_sha256": index_sha,
        "manifest_sha256": sha256(args.manifest),
        "dataset_sha256": os.environ.get("MBC_DATASET_SHA"),
        "git_commit": os.environ.get("MBC_GIT_COMMIT"),
        "image_id": os.environ.get("MBC_IMAGE_ID"),
        "environment_lock_sha256": os.environ.get("MBC_ENV_LOCK_SHA"),
        "complete": False,
    }
    provenance_path.write_text(json.dumps(request, indent=2) + "\n")

    memmap = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=storage_dtype, shape=(len(rows), dim)
    )
    dataset = BreakHisDataset(rows, slide_root, bundle.transform)
    shards = [
        (start, min(start + args.shard_size, len(rows)))
        for start in range(0, len(rows), args.shard_size)
    ]
    state = {"completed": {}}
    start_time = time.time()
    for shard_id, (start, stop) in enumerate(shards):
        loader = DataLoader(
            torch.utils.data.Subset(dataset, range(start, stop)),
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            shuffle=False,
        )
        outputs = []
        with torch.inference_mode():
            for batch, _ in loader:
                batch = batch.to(device, non_blocking=True)
                if device.type == "cuda" and not args.fp32:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs.append(bundle.forward(batch).float().cpu().numpy())
                else:
                    outputs.append(bundle.forward(batch).float().cpu().numpy())
        block = np.concatenate(outputs, axis=0)
        memmap[start:stop] = block.astype(storage_dtype, copy=False)
        memmap.flush()
        digest = hashlib.sha256(np.asarray(memmap[start:stop]).tobytes()).hexdigest()
        state["completed"][str(shard_id)] = digest
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        rate = (stop) / max(time.time() - start_time, 1e-6)
        print(
            f"  shard {shard_id + 1}/{len(shards)} complete "
            f"[{stop:,}/{len(rows):,}] {rate:.1f} images/s",
            flush=True,
        )

    request["complete"] = True
    request["embeddings_sha256"] = sha256(emb_path)
    provenance_path.write_text(json.dumps(request, indent=2) + "\n")
    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
