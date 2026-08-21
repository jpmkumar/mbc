#!/usr/bin/env python3
"""Cache frozen foundation-model embeddings for the IDC patch archive.

This is a pure forward pass. It consumes no labels for any decision, performs no
model selection, and is independent of the fold structure -- the embedding of a
patch is a function of the image alone. It is therefore safe to run before the
Phase 1 preregistration is filed.

Two transform families are supported, and they must be cached separately because
the difference between them *is* the C2 experiment:

``upsample224``
    The 50x50 patch resized directly to the encoder's input resolution. This is
    what the published IDC literature does, and it feeds a context-starved image
    to an encoder pretrained on 224x224 fields of view.

``mosaicK``
    A KxK neighbourhood assembled from spatially adjacent patches using the
    ``x``/``y`` coordinates in the filename, then resized. This restores genuine
    field of view at native magnification. The label is always that of the
    **centre** patch; neighbours contribute context only.

    Edge tiles and gaps are filled with white (slide background). Individual
    tiles are resized to 50x50 before assembly because the public archive
    contains truncated patches at mount boundaries.

.. warning::
   Mosaic transforms must not be paired with the *random patch split* arm of the
   C1 leakage experiment. A neighbour of a test patch can land in the training
   set, which adds a second leakage channel and confounds the measurement. Keep
   C1 on ``upsample224``. (That mosaics make random patch splitting even leakier
   is itself a reportable observation.)

Usage
-----
Smoke test on a subset, then the full pass::

    python extract_embeddings.py --encoder uni --transform upsample224 \\
        --archive-path /datasets/histopath --output-dir /outputs/emb --limit 5000

    python extract_embeddings.py --encoder uni --transform upsample224 \\
        --archive-path /datasets/histopath --output-dir /outputs/emb
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXCLUDED_DIR = "IDC_regular_ps50_idx5"
FNAME = re.compile(r"^(?P<stem>.+?)_idx(?P<idx>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_class(?P<cls>\d+)\.png$")
TILE = 50  # nominal patch edge in pixels; the archive's generation parameter


# --------------------------------------------------------------------------
# Encoder registry
# --------------------------------------------------------------------------

@dataclass
class Encoder:
    repo: str
    dim: int
    create_kwargs: dict = field(default_factory=dict)
    pool: str = "cls"  # "cls" or "cls_plus_mean" (Virchow's documented recipe)
    note: str = ""


def encoder_registry() -> dict[str, Encoder]:
    """Built lazily so ``timm`` is only imported when a model is actually needed."""
    import timm
    import torch.nn as nn

    return {
        "uni": Encoder(
            repo="hf-hub:MahmoodLab/uni",
            dim=1024,
            create_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
            pool="cls",
            note="ViT-L/16 @224, CC-BY-NC-ND, gated",
        ),
        "virchow": Encoder(
            repo="hf-hub:paige-ai/Virchow",
            dim=2560,
            create_kwargs={
                "mlp_layer": timm.layers.SwiGLUPacked,
                "act_layer": nn.SiLU,
            },
            pool="cls_plus_mean",
            note="ViT-H/14 @224, gated; documented embedding is cls||mean(patch tokens)",
        ),
    }


def load_encoder(name: str, device: torch.device):
    import timm

    reg = encoder_registry()
    if name not in reg:
        raise SystemExit(
            f"Unknown encoder {name!r}. Available: {sorted(reg)}.\n"
            "UNI2-h and CONCH are deliberately not wired up yet; see the paper plan."
        )
    spec = reg[name]
    model = timm.create_model(spec.repo, pretrained=True, num_classes=0, **spec.create_kwargs)
    model.eval().to(device)

    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, spec, cfg


def pool_output(out: torch.Tensor, pool: str) -> torch.Tensor:
    """Reduce encoder output to one vector per image."""
    if out.ndim == 2:  # already pooled by num_classes=0
        return out
    if pool == "cls_plus_mean":
        cls = out[:, 0]
        patches = out[:, 1:].mean(dim=1)
        return torch.cat([cls, patches], dim=-1)
    return out[:, 0]


# --------------------------------------------------------------------------
# Patch index
# --------------------------------------------------------------------------

def build_patch_index(archive: Path) -> pd.DataFrame:
    rows = []
    dirs = sorted(
        d for d in os.listdir(archive)
        if (archive / d).is_dir() and d != EXCLUDED_DIR
    )
    for d in dirs:
        for cls in ("0", "1"):
            cls_dir = archive / d / cls
            if not cls_dir.is_dir():
                continue
            for entry in os.scandir(cls_dir):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                m = FNAME.match(entry.name)
                if not m:
                    continue
                rows.append({
                    "patient_id": d,
                    "label": int(cls),
                    "x": int(m.group("x")),
                    "y": int(m.group("y")),
                    "filepath": f"{d}/{cls}/{entry.name}",
                })
    df = pd.DataFrame(rows)
    # Deterministic order. Every downstream artifact is aligned to this ordering,
    # so it must not depend on filesystem iteration order.
    return df.sort_values(["patient_id", "y", "x", "label"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class PatchDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        archive: Path,
        transform,
        mosaic_k: int = 1,
        neighbour_lookup: dict[tuple[str, int, int], str] | None = None,
    ):
        self.index = index.reset_index(drop=True)
        self.archive = archive
        self.transform = transform
        self.k = mosaic_k
        self.lookup = neighbour_lookup or {}

    def __len__(self) -> int:
        return len(self.index)

    def _tile(self, patient: str, x: int, y: int) -> Image.Image | None:
        rel = self.lookup.get((patient, x, y))
        if rel is None:
            return None
        try:
            with Image.open(self.archive / rel) as im:
                return im.convert("RGB").resize((TILE, TILE), Image.BILINEAR)
        except (OSError, ValueError):
            return None

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        patient, x, y = row["patient_id"], int(row["x"]), int(row["y"])

        if self.k == 1:
            with Image.open(self.archive / row["filepath"]) as im:
                img = im.convert("RGB")
        else:
            half = self.k // 2
            canvas = Image.new("RGB", (TILE * self.k, TILE * self.k), (255, 255, 255))
            for gy in range(-half, half + 1):
                for gx in range(-half, half + 1):
                    tile = self._tile(patient, x + gx * TILE, y + gy * TILE)
                    if tile is not None:
                        canvas.paste(tile, ((gx + half) * TILE, (gy + half) * TILE))
            img = canvas

        return self.transform(img), i


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--transform", required=True, help="upsample224 or mosaic3 / mosaic5 ...")
    ap.add_argument("--archive-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N patches")
    ap.add_argument("--fp32", action="store_true", help="disable autocast (debugging only)")
    args = ap.parse_args()

    if args.transform == "upsample224":
        mosaic_k = 1
    else:
        m = re.fullmatch(r"mosaic(\d+)", args.transform)
        if not m:
            raise SystemExit(f"Unrecognised transform {args.transform!r}")
        mosaic_k = int(m.group(1))
        if mosaic_k % 2 == 0:
            raise SystemExit("Mosaic K must be odd so the labelled patch is the centre tile.")

    archive = args.archive_path.expanduser().resolve()
    if not archive.is_dir():
        raise SystemExit(f"Archive not found: {archive}")

    out = args.output_dir.expanduser().resolve() / f"{args.encoder}_{args.transform}"
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / "embeddings.npy"
    idx_path = out / "index.csv"
    prov_path = out / "provenance.json"
    state_path = out / "shards_done.json"

    if prov_path.exists() and json.loads(prov_path.read_text()).get("complete"):
        raise SystemExit(f"A completed cache already exists at {out}. Refusing to overwrite.")

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; this will be extremely slow.", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Indexing archive...", flush=True)
    index = build_patch_index(archive)
    if args.limit:
        index = index.iloc[: args.limit].reset_index(drop=True)
    n = len(index)
    print(f"  {n:,} patches, {index['patient_id'].nunique()} patients", flush=True)

    lookup = None
    if mosaic_k > 1:
        lookup = {
            (r.patient_id, r.x, r.y): r.filepath
            for r in index.itertuples(index=False)
        }

    print(f"Loading encoder {args.encoder}...", flush=True)
    model, transform, spec, cfg = load_encoder(args.encoder, device)

    # Probe the true output width rather than trusting the registry constant.
    with torch.no_grad():
        probe = torch.zeros(1, *cfg["input_size"], device=device)
        dim = int(pool_output(model(probe), spec.pool).shape[-1])
    if dim != spec.dim:
        print(f"  note: probed dim {dim} differs from registry value {spec.dim}", flush=True)
    print(f"  embedding dim {dim}, input {cfg['input_size']}", flush=True)

    if not idx_path.exists():
        index.to_csv(idx_path, index=False)

    mm = np.lib.format.open_memmap(
        emb_path, mode="r+" if emb_path.exists() else "w+",
        dtype=np.float16, shape=(n, dim),
    )
    done: set[int] = set(json.loads(state_path.read_text())) if state_path.exists() else set()

    shards = [(s, min(s + args.shard_size, n)) for s in range(0, n, args.shard_size)]
    pending = [(a, b) for k, (a, b) in enumerate(shards) if k not in done]
    print(f"{len(pending)}/{len(shards)} shards pending", flush=True)

    t0 = time.time()
    seen = 0
    for k, (a, b) in enumerate(shards):
        if k in done:
            continue
        loader = DataLoader(
            PatchDataset(index.iloc[a:b], archive, transform, mosaic_k, lookup),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False,
        )
        with torch.no_grad():
            for batch, local in loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32 and device.type == "cuda"):
                    feats = pool_output(model(batch), spec.pool)
                mm[a + local.numpy()] = feats.float().cpu().numpy().astype(np.float16)
                seen += len(local)

        done.add(k)
        state_path.write_text(json.dumps(sorted(done)))
        mm.flush()
        rate = seen / max(time.time() - t0, 1e-6)
        eta_min = (n - b) / max(rate, 1e-6) / 60
        print(
            f"  shard {k + 1}/{len(shards)} done  [{b:,}/{n:,}]  "
            f"{rate:.0f} img/s  eta {eta_min:.1f} min",
            flush=True,
        )

    mm.flush()
    del mm

    prov = {
        "complete": True,
        "encoder": args.encoder,
        "encoder_repo": spec.repo,
        "encoder_note": spec.note,
        "pool": spec.pool,
        "transform": args.transform,
        "mosaic_k": mosaic_k,
        "tile_px": TILE,
        "embedding_dim": dim,
        "n_patches": n,
        "n_patients": int(index["patient_id"].nunique()),
        "input_size": list(cfg["input_size"]),
        "data_config": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in cfg.items()},
        "dtype": "float16",
        "archive_path": str(archive),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": os.environ.get("MBC_GIT_COMMIT"),
        "image_id": os.environ.get("MBC_IMAGE_ID"),
        "env_lock_sha": os.environ.get("MBC_ENV_LOCK_SHA"),
        "dataset_sha": os.environ.get("MBC_DATASET_SHA"),
        "wall_clock_s": round(time.time() - t0, 1),
        "limit_applied": args.limit or None,
    }
    prov_path.write_text(json.dumps(prov, indent=2))

    print(f"\nWrote {emb_path}  ({n:,} x {dim}, float16)")
    print(f"Wrote {idx_path}")
    print(f"Wrote {prov_path}")
    print(f"Elapsed {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
