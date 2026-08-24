#!/usr/bin/env python3
"""Extract pinned Paper C embeddings for preregistered BCSS centres."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from extract_embeddings import (
    DEFAULT_REGISTRY,
    json_safe,
    load_encoder,
    registry_entry,
    sha256,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    paths = [row["filepath"] for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("BCSS centre filepath key is not unique.")
    return rows


class BCSSDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        context_size: int,
    ):
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform
        self.context_size = context_size
        self._images: OrderedDict[str, Image.Image] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, filename: str) -> Image.Image:
        if filename not in self._images:
            with Image.open(self.image_dir / filename) as image:
                self._images[filename] = image.convert("RGB")
            if len(self._images) > 2:
                self._images.popitem(last=False)
        return self._images[filename]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = self._image(row["image_filename"])
        half = self.context_size // 2
        x, y = int(row["x"]), int(row["y"])
        crop = image.crop((x - half, y - half, x + half, y + half))
        if crop.size != (self.context_size, self.context_size):
            raise RuntimeError(f"Incomplete BCSS context for {row['filepath']}")
        return self.transform(crop), index


def write_index(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--context", required=True, choices=("k1", "k9"))
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--hf-cache", type=Path, default=Path("/cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--centre-limit", type=int, default=0)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if args.centre_limit and not args.run_name:
        raise SystemExit("--centre-limit requires a separate --run-name.")
    if not args.centre_limit and not os.environ.get("MBC_DATASET_SHA"):
        raise SystemExit("Full extraction requires MBC_DATASET_SHA provenance.")
    context_size = 50 if args.context == "k1" else 450
    spec, registry_sha = registry_entry(args.registry, args.encoder)
    rows = read_manifest(args.manifest)
    if args.centre_limit:
        rows = rows[: args.centre_limit]
    if not rows:
        raise SystemExit("BCSS centre manifest is empty.")

    name = f"bcss_{args.encoder}_{args.context}"
    if args.run_name:
        name += f"__{args.run_name}"
    output = args.output_dir.resolve() / name
    output.mkdir(parents=True, exist_ok=True)
    emb_path = output / "embeddings.npy"
    index_path = output / "index.csv"
    state_path = output / "shards_done.json"
    provenance_path = output / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()).get("complete"):
        raise SystemExit(f"Completed cache exists at {output}; refusing overwrite.")
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as stream:
            observed = [row["filepath"] for row in csv.DictReader(stream)]
        if observed != [row["filepath"] for row in rows]:
            raise RuntimeError("Existing BCSS cache index differs from request.")
        index_sha = sha256(index_path)
    else:
        index_sha = write_index(index_path, rows)

    storage_dtype = np.float32 if args.fp32 else np.float16
    storage_dtype_name = np.dtype(storage_dtype).name
    request = {
        "complete": False,
        "cohort": "BCSS",
        "encoder": args.encoder,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "context": args.context,
        "context_size_pixels": context_size,
        "n_centres": len(rows),
        "manifest_sha256": sha256(args.manifest),
        "index_sha256": index_sha,
        "model_registry_sha256": registry_sha,
        "compute_precision": "fp32" if args.fp32 else "autocast-fp16",
        "storage_dtype": storage_dtype_name,
        "shard_size": args.shard_size,
        "dataset_sha256": os.environ.get("MBC_DATASET_SHA"),
        "git_commit": os.environ.get("MBC_GIT_COMMIT"),
        "image_id": os.environ.get("MBC_IMAGE_ID"),
        "environment_lock_sha256": os.environ.get("MBC_ENV_LOCK_SHA"),
    }

    if not torch.cuda.is_available():
        raise SystemExit("BCSS confirmatory extraction requires CUDA.")
    device = torch.device("cuda")
    bundle = load_encoder(spec, device, args.hf_cache)
    bundle.model.eval().to(device)
    batch_size = args.batch_size or int(spec["batch_size_a4000"])
    input_size = bundle.data_config.get("input_size", [3, 224, 224])
    with torch.inference_mode():
        probe = bundle.forward(torch.zeros(1, *input_size, device=device))
    dim = int(probe.shape[1])
    if probe.ndim != 2 or dim != int(spec["expected_dim"]):
        raise RuntimeError(f"Unexpected output shape: {tuple(probe.shape)}")
    request["embedding_dim"] = dim
    request["model_artifact_sha256"] = bundle.artifact_hashes
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        for key, value in request.items():
            if old.get(key) != value:
                raise RuntimeError(f"BCSS resume provenance mismatch for {key}.")
    else:
        provenance_path.write_text(
            json.dumps({"complete": False, **request}, indent=2) + "\n"
        )

    n = len(rows)
    if emb_path.exists():
        existing = np.load(emb_path, mmap_mode="r")
        if existing.shape != (n, dim) or existing.dtype != storage_dtype:
            raise RuntimeError(
                f"Existing BCSS memmap is {existing.shape}/{existing.dtype}; "
                f"expected {(n, dim)}/{storage_dtype_name}."
            )
        del existing
    memmap = np.lib.format.open_memmap(
        emb_path,
        mode="r+" if emb_path.exists() else "w+",
        dtype=storage_dtype,
        shape=(n, dim),
    )
    request_sha = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"request_sha256": request_sha, "completed": {}}
    )
    if state.get("request_sha256") != request_sha:
        raise RuntimeError("BCSS shard-state request hash differs from this run.")
    completed_hashes = {
        int(shard): digest for shard, digest in state.get("completed", {}).items()
    }
    for shard_id, digest in completed_hashes.items():
        start = shard_id * args.shard_size
        stop = min(start + args.shard_size, n)
        observed = hashlib.sha256(
            np.asarray(memmap[start:stop]).tobytes()
        ).hexdigest()
        if observed != digest:
            raise RuntimeError(f"Completed BCSS shard {shard_id} failed integrity.")
    done = set(completed_hashes)
    shards = [
        (start, min(start + args.shard_size, n))
        for start in range(0, n, args.shard_size)
    ]
    dataset = BCSSDataset(rows, args.images, bundle.transform, context_size)
    started = time.time()
    for shard_id, (start, stop) in enumerate(shards):
        if shard_id in done:
            continue
        loader = DataLoader(
            Subset(dataset, range(start, stop)),
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False,
        )
        cursor = start
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=not args.fp32
                ):
                    features = bundle.forward(images)
                if features.shape[1] != dim:
                    raise RuntimeError("Embedding dimension changed during extraction.")
                count = len(features)
                memmap[cursor:cursor + count] = (
                    features.float().cpu().numpy().astype(storage_dtype)
                )
                cursor += count
        if cursor != stop:
            raise RuntimeError(f"Shard {shard_id} row-count mismatch.")
        memmap.flush()
        done.add(shard_id)
        completed_hashes[shard_id] = hashlib.sha256(
            np.asarray(memmap[start:stop]).tobytes()
        ).hexdigest()
        state_path.write_text(json.dumps({
            "request_sha256": request_sha,
            "completed": {
                str(shard): completed_hashes[shard]
                for shard in sorted(completed_hashes)
            },
        }, indent=2) + "\n")
        print(f"shard {shard_id + 1}/{len(shards)} complete", flush=True)
    del memmap
    if done != set(range(len(shards))):
        raise RuntimeError("Not every BCSS shard completed.")

    provenance = {
        **request,
        "complete": True,
        "embedding_dim": dim,
        "dtype": storage_dtype_name,
        "batch_size": batch_size,
        "model_artifact_sha256": bundle.artifact_hashes,
        "data_config": json_safe(bundle.data_config),
        "embeddings_sha256": sha256(emb_path),
        "dataset_sha256": os.environ.get("MBC_DATASET_SHA"),
        "git_commit": os.environ.get("MBC_GIT_COMMIT"),
        "image_id": os.environ.get("MBC_IMAGE_ID"),
        "environment_lock_sha256": os.environ.get("MBC_ENV_LOCK_SHA"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "wall_clock_seconds": round(time.time() - started, 3),
        "engineering_subset": args.centre_limit or None,
        "run_name": args.run_name or None,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {output} ({n:,} x {dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
