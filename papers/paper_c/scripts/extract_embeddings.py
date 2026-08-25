#!/usr/bin/env python3
"""Extract revision-pinned frozen embeddings for Paper C.

The cache is fold-independent and label-blind with respect to model selection.
Every consumer must join ``index.csv`` to split manifests on ``filepath``;
embedding row order is not the fold-manifest order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXCLUDED_DIR = "IDC_regular_ps50_idx5"
FNAME = re.compile(
    r"^(?P<stem>.+?)_idx(?P<idx>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_class(?P<cls>\d+)\.png$"
)
TILE = 50
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "papers/paper_c/config/model_registry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def registry_entry(path: Path, name: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    registry = json.loads(raw)
    if name not in registry:
        raise SystemExit(f"Unknown encoder {name!r}; choose from {sorted(registry)}")
    return registry[name], hashlib.sha256(raw).hexdigest()


def model_file_hashes(snapshot: Path) -> dict[str, str]:
    loading_suffixes = {".json", ".safetensors", ".bin", ".py"}
    hashes = {
        str(path.relative_to(snapshot)): sha256(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.suffix in loading_suffixes
    }
    if "config.json" not in hashes or not any(
        name.endswith((".safetensors", ".bin")) for name in hashes
    ):
        raise RuntimeError(f"Incomplete model snapshot at {snapshot}")
    return hashes


def pool_virchow2_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (261, 1280):
        raise RuntimeError(
            "Virchow2 must return [batch, 261, 1280] tokens; "
            f"received {tuple(tokens.shape)}"
        )
    return torch.cat([tokens[:, 0], tokens[:, 5:].mean(dim=1)], dim=-1)


class EncoderBundle:
    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        forward: Callable[[torch.Tensor], torch.Tensor],
        data_config: dict[str, Any],
        artifact_hashes: dict[str, str],
        snapshot: Path,
    ):
        self.model = model
        self.transform = transform
        self.forward = forward
        self.data_config = data_config
        self.artifact_hashes = artifact_hashes
        self.snapshot = snapshot


def load_encoder(
    spec: dict[str, Any],
    device: torch.device,
    cache_dir: Path,
) -> EncoderBundle:
    """Download an exact snapshot, then load only from that local directory."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        cache_dir=cache_dir,
    ))
    hashes = model_file_hashes(snapshot)
    provider = spec["provider"]

    if provider == "timm":
        import timm
        from timm.data import create_transform, resolve_data_config

        kwargs: dict[str, Any] = {}
        if spec["pooling"] == "vector":
            kwargs["num_classes"] = 0
        if spec["pooling"] == "cls_plus_mean_skip_5":
            kwargs["mlp_layer"] = timm.layers.SwiGLUPacked
            kwargs["act_layer"] = torch.nn.SiLU

        model = timm.create_model(
            f"local-dir:{snapshot}",
            pretrained=True,
            **kwargs,
        )
        config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**config, is_training=False)

        if spec["pooling"] == "cls_plus_mean_skip_5":
            def forward(batch: torch.Tensor) -> torch.Tensor:
                return pool_virchow2_tokens(model(batch))
        else:
            def forward(batch: torch.Tensor) -> torch.Tensor:
                output = model(batch)
                if output.ndim != 2:
                    raise RuntimeError(
                        f"Expected a pooled feature matrix, got {tuple(output.shape)}"
                    )
                return output

        return EncoderBundle(
            model, transform, forward, dict(config), hashes, snapshot
        )

    if provider == "transformers":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            snapshot, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )

        def transform(image: Image.Image) -> torch.Tensor:
            return processor(images=image, return_tensors="pt")["pixel_values"][0]

        def forward(batch: torch.Tensor) -> torch.Tensor:
            output = model(pixel_values=batch).last_hidden_state
            if output.ndim != 3:
                raise RuntimeError(
                    f"Expected transformer tokens, got {tuple(output.shape)}"
                )
            return output[:, 0]

        config = {
            "input_size": [3, processor.size.get("height", 224), processor.size.get("width", 224)],
            "image_mean": list(processor.image_mean),
            "image_std": list(processor.image_std),
        }
        return EncoderBundle(model, transform, forward, config, hashes, snapshot)

    raise RuntimeError(f"Unsupported model provider: {provider}")


def build_patch_index(archive: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids = sorted(
        name
        for name in os.listdir(archive)
        if (archive / name).is_dir() and name != EXCLUDED_DIR
    )
    for case_id in case_ids:
        for cls in ("0", "1"):
            cls_dir = archive / case_id / cls
            if not cls_dir.is_dir():
                continue
            for entry in os.scandir(cls_dir):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                match = FNAME.match(entry.name)
                if not match:
                    raise RuntimeError(f"Unexpected patch filename: {entry.name}")
                rows.append({
                    "filepath": f"{case_id}/{cls}/{entry.name}",
                    "case_id": case_id,
                    "label": int(cls),
                    "x": int(match.group("x")),
                    "y": int(match.group("y")),
                })
    rows.sort(key=lambda row: (row["case_id"], row["y"], row["x"], row["label"]))
    filepaths = [row["filepath"] for row in rows]
    if len(filepaths) != len(set(filepaths)):
        raise RuntimeError("Archive index contains duplicate filepaths.")
    return rows


def add_context_metadata(
    centres: list[dict[str, Any]],
    lookup: dict[tuple[str, int, int], str],
    k: int,
) -> None:
    half = k // 2
    total = k * k
    for row in centres:
        available = sum(
            (row["case_id"], row["x"] + gx * TILE, row["y"] + gy * TILE) in lookup
            for gy in range(-half, half + 1)
            for gx in range(-half, half + 1)
        )
        row["context_k"] = k
        row["context_available"] = available
        row["context_total"] = total
        row["context_complete"] = int(available == total)
        row["padding_fraction"] = (total - available) / total


class PatchDataset(Dataset):
    def __init__(
        self,
        centres: list[dict[str, Any]],
        archive: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        k: int,
        lookup: dict[tuple[str, int, int], str],
    ):
        self.centres = centres
        self.archive = archive
        self.transform = transform
        self.k = k
        self.lookup = lookup

    def __len__(self) -> int:
        return len(self.centres)

    def _tile(self, case_id: str, x: int, y: int) -> Image.Image | None:
        relpath = self.lookup.get((case_id, x, y))
        if relpath is None:
            return None
        with Image.open(self.archive / relpath) as image:
            return image.convert("RGB").resize((TILE, TILE), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.centres[index]
        if self.k == 1:
            with Image.open(self.archive / row["filepath"]) as image:
                transformed = self.transform(image.convert("RGB"))
            return transformed, index

        half = self.k // 2
        canvas = Image.new("RGB", (TILE * self.k, TILE * self.k), (255, 255, 255))
        for gy in range(-half, half + 1):
            for gx in range(-half, half + 1):
                tile = self._tile(
                    row["case_id"],
                    row["x"] + gx * TILE,
                    row["y"] + gy * TILE,
                )
                if tile is not None:
                    canvas.paste(tile, ((gx + half) * TILE, (gy + half) * TILE))
        return self.transform(canvas), index


def write_index(path: Path, rows: list[dict[str, Any]]) -> str:
    fields = [
        "filepath", "case_id", "label", "x", "y", "context_k",
        "context_available", "context_total", "context_complete",
        "padding_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return sha256(path)


def assert_existing_index(path: Path, centres: list[dict[str, Any]]) -> str:
    with path.open(newline="", encoding="utf-8") as stream:
        existing = list(csv.DictReader(stream))
    expected = [row["filepath"] for row in centres]
    observed = [row["filepath"] for row in existing]
    if observed != expected:
        raise RuntimeError(
            "Existing cache index differs from the requested centre population."
        )
    return sha256(path)


def parse_transform(name: str) -> int:
    if name == "upsample224":
        return 1
    match = re.fullmatch(r"mosaic(3|5|9)", name)
    if not match:
        raise SystemExit("Transform must be upsample224, mosaic3, mosaic5 or mosaic9.")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--archive-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--hf-cache", type=Path, default=Path("/cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument(
        "--centre-limit",
        type=int,
        default=0,
        help="Engineering smoke subset; neighbour lookup still uses the full archive.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Required for smoke subsets so production cache paths are untouched.",
    )
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    k = parse_transform(args.transform)
    if args.centre_limit and not args.run_name:
        raise SystemExit("--centre-limit requires --run-name (for example smoke-5000).")
    if not args.centre_limit and not os.environ.get("MBC_DATASET_SHA"):
        raise SystemExit("Full extraction requires MBC_DATASET_SHA provenance.")

    archive = args.archive_path.expanduser().resolve()
    if not archive.is_dir():
        raise SystemExit(f"Archive not found: {archive}")
    spec, registry_sha = registry_entry(args.registry, args.encoder)
    cache_name = f"{args.encoder}_{args.transform}"
    if args.run_name:
        cache_name += f"__{args.run_name}"
    output = args.output_dir.expanduser().resolve() / cache_name
    output.mkdir(parents=True, exist_ok=True)
    emb_path = output / "embeddings.npy"
    index_path = output / "index.csv"
    provenance_path = output / "provenance.json"
    state_path = output / "shards_done.json"

    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if old.get("complete"):
            raise SystemExit(f"Completed cache exists at {output}; refusing overwrite.")

    print("Indexing archive...", flush=True)
    full_index = build_patch_index(archive)
    if len(full_index) != 277_524:
        raise RuntimeError(f"Expected 277,524 patches, found {len(full_index):,}")
    lookup = {
        (row["case_id"], row["x"], row["y"]): row["filepath"]
        for row in full_index
    }
    if len(lookup) != len(full_index):
        raise RuntimeError("Duplicate case/x/y coordinates found in archive.")

    centres = full_index[: args.centre_limit] if args.centre_limit else full_index
    centres = [dict(row) for row in centres]
    add_context_metadata(centres, lookup, k)
    complete = sum(row["context_complete"] for row in centres)
    print(
        f"  {len(centres):,} centres; {complete:,} have complete K={k} context",
        flush=True,
    )

    if index_path.exists():
        index_sha = assert_existing_index(index_path, centres)
    else:
        index_sha = write_index(index_path, centres)
    storage_dtype = np.float32 if args.fp32 else np.float16
    storage_dtype_name = np.dtype(storage_dtype).name
    request = {
        "encoder": args.encoder,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "transform": args.transform,
        "context_k": k,
        "n_centres": len(centres),
        "compute_precision": "fp32" if args.fp32 else "autocast-fp16",
        "storage_dtype": storage_dtype_name,
        "shard_size": args.shard_size,
        "model_registry_sha256": registry_sha,
        "index_sha256": index_sha,
        "dataset_sha256": os.environ.get("MBC_DATASET_SHA"),
        "git_commit": os.environ.get("MBC_GIT_COMMIT"),
        "image_id": os.environ.get("MBC_IMAGE_ID"),
        "environment_lock_sha256": os.environ.get("MBC_ENV_LOCK_SHA"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable; extraction is not qualified.", file=sys.stderr)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(
        f"Loading {spec['repo_id']} at revision {spec['revision']}...",
        flush=True,
    )
    bundle = load_encoder(spec, device, args.hf_cache)
    bundle.model.eval().to(device)
    batch_size = args.batch_size or int(spec["batch_size_a4000"])

    input_size = bundle.data_config.get("input_size", [3, 224, 224])
    with torch.inference_mode():
        probe = torch.zeros(1, *input_size, device=device)
        features = bundle.forward(probe)
    if features.ndim != 2 or features.shape[0] != 1:
        raise RuntimeError(f"Invalid encoder output shape: {tuple(features.shape)}")
    dim = int(features.shape[1])
    if dim != int(spec["expected_dim"]):
        raise RuntimeError(
            f"{args.encoder} returned {dim} dimensions; "
            f"registry requires {spec['expected_dim']}."
        )
    request["embedding_dim"] = dim
    request["model_artifact_sha256"] = bundle.artifact_hashes
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        mismatches = {
            key: (old.get(key), value)
            for key, value in request.items()
            if old.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Incomplete cache provenance does not match request: {mismatches}"
            )
    else:
        provenance_path.write_text(
            json.dumps({"complete": False, **request}, indent=2) + "\n"
        )
    print(f"  output dim={dim}; batch={batch_size}; input={input_size}", flush=True)

    n = len(centres)
    if emb_path.exists():
        existing = np.load(emb_path, mmap_mode="r")
        if existing.shape != (n, dim) or existing.dtype != storage_dtype:
            raise RuntimeError(
                f"Existing memmap has shape/dtype {existing.shape}/{existing.dtype}; "
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
        raise RuntimeError("Shard-state request hash differs from this extraction.")
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
            raise RuntimeError(f"Completed shard {shard_id} failed integrity check.")
    completed_shards = set(completed_hashes)
    shards = [
        (start, min(start + args.shard_size, n))
        for start in range(0, n, args.shard_size)
    ]
    dataset = PatchDataset(centres, archive, bundle.transform, k, lookup)
    start_time = time.time()
    processed = 0

    for shard_id, (start, stop) in enumerate(shards):
        if shard_id in completed_shards:
            continue
        loader = DataLoader(
            torch.utils.data.Subset(dataset, range(start, stop)),
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            shuffle=False,
        )
        cursor = start
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=device.type == "cuda" and not args.fp32,
                ):
                    batch_features = bundle.forward(images)
                if batch_features.shape[1] != dim:
                    raise RuntimeError("Embedding dimension changed during extraction.")
                count = batch_features.shape[0]
                memmap[cursor:cursor + count] = (
                    batch_features.float().cpu().numpy().astype(storage_dtype)
                )
                cursor += count
                processed += count
        if cursor != stop:
            raise RuntimeError(f"Shard {shard_id} wrote {cursor-start}, expected {stop-start}")
        memmap.flush()
        completed_shards.add(shard_id)
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
        rate = processed / max(time.time() - start_time, 1e-6)
        print(
            f"  shard {shard_id + 1}/{len(shards)} complete "
            f"[{stop:,}/{n:,}] {rate:.1f} images/s",
            flush=True,
        )

    memmap.flush()
    del memmap
    if completed_shards != set(range(len(shards))):
        raise RuntimeError("Not every shard completed.")

    provenance = {
        "complete": True,
        "encoder": args.encoder,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "license": spec["license"],
        "redistribute_embeddings": spec["redistribute_embeddings"],
        "model_artifact_sha256": bundle.artifact_hashes,
        "model_snapshot": str(bundle.snapshot),
        "model_registry_sha256": registry_sha,
        "transform": args.transform,
        "context_k": k,
        "n_centres": n,
        "n_complete_context": complete,
        "embedding_dim": dim,
        "dtype": storage_dtype_name,
        "compute_precision": "fp32" if args.fp32 else "autocast-fp16",
        "batch_size": batch_size,
        "data_config": json_safe(bundle.data_config),
        "index_sha256": index_sha,
        "embeddings_sha256": sha256(emb_path),
        "dataset_sha256": os.environ.get("MBC_DATASET_SHA"),
        "git_commit": os.environ.get("MBC_GIT_COMMIT"),
        "image_id": os.environ.get("MBC_IMAGE_ID"),
        "environment_lock_sha256": os.environ.get("MBC_ENV_LOCK_SHA"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "wall_clock_seconds": round(time.time() - start_time, 3),
        "engineering_subset": args.centre_limit or None,
        "run_name": args.run_name or None,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {output} ({n:,} × {dim})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
