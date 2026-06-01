# Training performance

## Why it used to take hours

Hybrid training was slow for four stacked reasons:

| Bottleneck | Effect |
|------------|--------|
| **Entire model on CPU** | EfficientNet + Transformer ran on CPU because PennyLane VQC needs CPU |
| **Stage B re-ran backbone every batch** | Frozen backbone still processed every image, every epoch (~137 steps × 20–30 epochs) |
| **PennyLane `default.qubit`** | Pure Python statevector simulator for 8-qubit backprop |
| **`num_workers=0`** | Image loading blocked the training loop |

Example: Stage B alone ≈ **3,340 forward passes** through EfficientNet @ 224² on CPU → **1–4+ hours**.

## What we changed

1. **Split devices** — classical backbone on GPU/MPS; VQC head on CPU only  
2. **Feature cache (Stage B)** — extract 8-d compressed features once; train VQC on tensors  
3. **`lightning.qubit` + `adjoint`** — faster C++ PennyLane simulator (auto-fallback to `default.qubit` + `backprop`)  
4. **`num_workers` from config** — parallel image loading (Colab: use `2`, not `4`)  
5. **Optional `val_interval` / `checkpoint_interval`** — skip some val/checkpoint epochs  

## Expected speedups

| Stage | Before | After (Colab T4 + cache) |
|-------|--------|--------------------------|
| **A** (classical) | ~30–60 min CPU | ~5–15 min GPU |
| **B** (VQC, 20 ep) | ~1–4 h CPU | ~15–45 min |
| **Full hybrid** | ~3–8 h | ~30–90 min |

Stage B is still slower than Stage A because **VQC backprop on CPU** remains the bottleneck.

## Colab tips

```yaml
# configs/benedetti_vqc.yaml
data:
  num_workers: 2          # 4 can hang on Colab
training:
  cache_frozen_backbone_features: true
  classical_device: auto  # uses CUDA on Colab
  val_interval: 2         # optional: validate every 2 epochs
```

After `git pull`, confirm split devices in logs:

```text
Hybrid devices: classical=cuda, quantum=cpu
Pre-extracting frozen backbone features (one-time, ~2-5 min on GPU)...
```

Feature cache is saved under `results/feature_cache/`. Delete it if you change the Stage A checkpoint.

## Semantic note

With `cache_frozen_backbone_features: true`, Stage B train uses **eval transforms** (no random flip/rotation). This is standard for frozen-feature quantum head training and matches the Benedetti “fixed classical encoder” setup. Set to `false` to keep augmentations (slower).

## Parallel processing & AMP (latest)

| Setting | What it does |
|---------|----------------|
| `num_workers: 2` | **2 CPU workers** load + CLAHE in parallel while GPU trains |
| `prefetch_factor: 2` | Each worker pre-loads 2 batches ahead |
| `use_amp: true` | **Mixed precision FP16** on GPU (~1.3–1.8× Stage A) |
| `batch_size: 32` | Fewer steps/epoch (~65 vs 130 at 2k samples) |
| `feature_cache_batch_size: 64` | Faster one-time backbone extract for Stage B |
| `val_interval: 2` | Skip validation every other epoch |

**Not parallelizable:** PennyLane VQC backprop on CPU (Stage B) — use feature cache instead.

Expected Stage A after pull: **~0.5–0.7 s/it** (was ~1.2 s/it), **~25–40 min** for 30 epochs.

If OOM at batch 32, set `batch_size: 16` in `mammo_enhanced.yaml`.

Log line to confirm: `Mixed precision (AMP) enabled on cuda`.
