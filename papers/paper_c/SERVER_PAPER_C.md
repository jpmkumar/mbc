# Paper C — RTX A4000 execution runbook

This runbook executes the preregistered classical foundation-model study on the
campus RTX A4000. Paper B images, caches and result values are not reused.

## Current gate

**2026-11-06:** protocol status is `LOCKED`. Confirmatory encoders
(`dinov2_vitl14`, `resnet50`) passed smoke + fp16 precision; BCSS centres are
stride-150; IDC/BCSS manifests are frozen. Label-blind IDC `K=1` extraction may
run. Do **not** run labelled probes until `preregistration/paper_c_lock.json`
passes `verify_paperc_lock.py`.

**ntfy:** same topic file as Path A / Paper D — `~/.config/mbc/ntfy_topic`.
Live embedding runs: `papers/paper_c/scripts/watch_paperc_ntfy.sh` (shards,
START/DONE, cache verify, failures).

**Ungated redesign:** UNI2-h, Virchow2 and Phikon-v2 are declared unavailable
(institutional Hugging Face gated access cannot be completed). No Hugging Face
Read token is required for the confirmatory panel. Weights load from `torch.hub`
(DINOv2) and `torchvision` (ResNet-50).

Caches and manifests join only on unique `filepath`. The image builder refuses
to run until `docker/requirements-paperc.lock` has a hard-coded qualified SHA.

## 1. Server preparation

```bash
cd "$HOME/projects/mbc"
git pull --ff-only
export MBC_PRIMARY_ROOT="$HOME/mbc-primary"
mkdir -p "$MBC_PRIMARY_ROOT"/{cache,embeddings,logs,provenance/paper-c,results/paper-c}
```

No Hugging Face token is required for the confirmatory panel. Server outbound
HTTPS must reach the PyTorch / GitHub endpoints used by `torchvision` and
`torch.hub`. Gated UNI2-h / Virchow2 / Phikon-v2 downloads are not attempted.

## 2. Resolve and pin the environment

Run once on the qualified server:

```bash
papers/paper_c/scripts/resolve_paperc_lock.sh
```

Copy the printed SHA-256 into `EXPECTED_LOCK_SHA` in
`papers/paper_c/scripts/build_paperc_image.sh`, review
`docker/requirements-paperc.lock`, then build:

```bash
papers/paper_c/scripts/build_paperc_image.sh
```

The builder requires a clean tracked tree, uses the digest-pinned NGC 26.06
base, installs the lock without dependency resolution, archives the exact source
commit and writes image metadata under the gitignored
`results/server_setup/current_paperc_image.env`.

## 3. Verify IDC data and freeze context artifacts

The canonical IDC archive is:

```bash
export MBC_DATASET_DIR="$MBC_PRIMARY_ROOT/datasets/breast-histopathology-images"
test -d "$MBC_DATASET_DIR"
test -f "$MBC_PRIMARY_ROOT/provenance/dataset_archive_sha256.txt"
```

Run the audit and filepath-keyed manifest builder:

```bash
python3 papers/paper_c/scripts/audit_idc_context.py \
  --archive-path "$MBC_DATASET_DIR" \
  --output-dir "$MBC_PRIMARY_ROOT/provenance/paper-c/idc-context"

python3 papers/paper_c/scripts/build_idc_patch_manifest.py \
  --index "$MBC_PRIMARY_ROOT/provenance/paper-c/idc-context/idc_context_eligibility.csv" \
  --case-stats data/splits/histopath_kaggle/patient_stats.csv \
  --output "$MBC_PRIMARY_ROOT/provenance/paper-c/idc_patch_manifest.csv"
```

The grouped protocol reads frozen inner case-ID splits from
`data/splits/paper_c/idc`. They are committed, so regenerate them only to confirm
reproducibility, and stop if the tree differs:

```bash
python3 papers/paper_c/scripts/build_idc_inner_splits.py \
  --output /tmp/paper_c_inner_splits_check
diff -r data/splits/paper_c/idc /tmp/paper_c_inner_splits_check
```

Expected pre-outcome context counts are `K1=277524`, `K3=162018`,
`K5=108196`, `K9=49798`. Stop on any difference.

## 4. Qualify encoders

The confirmatory registry uses openly downloadable DINOv2 and ResNet-50
weights (no Hugging Face token). Do not inspect labelled probe performance.

### 4a. Confirmatory smoke (no HF token)

```bash
cd "$HOME/projects/mbc"
export MBC_PRIMARY_ROOT="$HOME/mbc-primary"
source results/server_setup/current_paperc_image.env

# Sync the 2026-11-06 ungated registry + extract_embeddings.py first.
chmod +x papers/paper_c/scripts/run_smoke_paperc_encoder.sh \
         papers/paper_c/scripts/run_extract_embeddings.sh

for encoder in dinov2_vitl14 resnet50; do
  bash papers/paper_c/scripts/run_smoke_paperc_encoder.sh "$encoder"
done
```

Expected embedding dimensions:

- DINOv2 ViT-L/14: 1024
- ResNet-50: 2048

The report must show `PASS`, finite output and a batch peak below the 16 GB
A4000 limit. Lower the model's `batch_size_a4000` in the registry if necessary,
rerun every smoke test, and update the registry hash in the preregistration
before lock. Attempts to smoke `uni2_h`, `virchow2` or `phikon_v2` must exit
with the registry unavailable message.

### 4b. Precision check (both confirmatory encoders)

```bash
for encoder in dinov2_vitl14 resnet50; do
  papers/paper_c/scripts/run_extract_embeddings.sh "$encoder" upsample224 \
    --centre-limit 1000 --run-name precision-fp16
  papers/paper_c/scripts/run_extract_embeddings.sh "$encoder" upsample224 \
    --centre-limit 1000 --run-name precision-fp32 --fp32
  docker run --rm \
    --mount type=bind,src="$MBC_PRIMARY_ROOT/embeddings",dst=/outputs,readonly \
    "$MBC_IMAGE" \
    python papers/paper_c/scripts/compare_embedding_precision.py \
      --fp16-cache /outputs/${encoder}_upsample224__precision-fp16 \
      --fp32-cache /outputs/${encoder}_upsample224__precision-fp32
done
```

Equivalence gates are `cosine_min >= 0.995`, `cosine_mean >= 0.9999`, and
`l2_normalized_abs_max <= 0.02`. Raw absolute maxima are reported only; they
are dominated by near-zero feature units and are not the confirmatory fail
criterion.

Smoke caches have unique namespaces and never require deleting a completed
production cache.

Run unit tests inside the qualified image:

```bash
docker run --rm "$MBC_IMAGE" \
  python -m pytest \
    tests/test_paper_c_splits.py \
    tests/test_paper_c_embedding_pipeline.py -q
```

## 5. Lock the preregistration

Record:

- source commit and image ID;
- dependency-lock SHA;
- IDC dataset, outer/inner split, context-index and patch-manifest hashes;
- BCSS dataset and centre-manifest hashes;
- model-registry and protocol hashes.

Change only the protocol status from `DRAFT` to `LOCKED`, commit that lock, and
rebuild the source image. Do not inspect comparative labelled performance before
this point.

## 6. IDC K=1 extraction

```bash
for encoder in dinov2_vitl14 resnet50; do
  papers/paper_c/scripts/run_extract_embeddings.sh "$encoder" upsample224
done
```

Every full cache must contain 277,524 rows and report a complete provenance file.
The cache row order is not a split order.

## 7. IDC protocol experiment

For each encoder, run paired random and grouped out-of-fold probes:

```bash
MANIFEST="$MBC_PRIMARY_ROOT/provenance/paper-c/idc_patch_manifest.csv"
for encoder in dinov2_vitl14 resnet50; do
  CACHE="$MBC_PRIMARY_ROOT/embeddings/${encoder}_upsample224"
  for seed in 42 43 44; do
    # Co-primary: both arms harmonised, so the contrast isolates grouping.
    python papers/paper_c/scripts/run_idc_probe_cv.py \
      --cache "$CACHE" --patch-manifest "$MANIFEST" --protocol random \
      --weighting case-balanced --seed "$seed" \
      --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/idc/${encoder}/random-k1/seed-${seed}"
    python papers/paper_c/scripts/run_idc_probe_cv.py \
      --cache "$CACHE" --patch-manifest "$MANIFEST" --protocol grouped \
      --weighting case-balanced --seed "$seed" \
      --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/idc/${encoder}/grouped-k1/seed-${seed}"
    # Ordered secondary: conventional random-patch recipe, for the bundled
    # protocol-regime contrast against the same grouped arm.
    python papers/paper_c/scripts/run_idc_probe_cv.py \
      --cache "$CACHE" --patch-manifest "$MANIFEST" --protocol random \
      --weighting protocol-native --seed "$seed" \
      --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/idc/${encoder}/random-k1-bundled/seed-${seed}"
  done
done
```

Each bundle's `summary.json` records its `weighting`. Never pair predictions
from different regimes.

Average each condition's seed-42/43/44 OOF files before any bootstrap or metric
comparison:

```bash
IDC_ROOT="$MBC_PRIMARY_ROOT/results/paper-c/idc"
for encoder in dinov2_vitl14 resnet50; do
  for condition in random-k1 grouped-k1 random-k1-bundled; do
    python papers/paper_c/scripts/average_seed_predictions.py \
      --prediction "$IDC_ROOT/${encoder}/${condition}/seed-42/oof_predictions.csv" --seed 42 \
      --prediction "$IDC_ROOT/${encoder}/${condition}/seed-43/oof_predictions.csv" --seed 43 \
      --prediction "$IDC_ROOT/${encoder}/${condition}/seed-44/oof_predictions.csv" --seed 44 \
      --output "$IDC_ROOT/${encoder}/${condition}/seed_mean_predictions.csv"
  done
done
```

## 8. IDC context extraction and probes

The confirmatory reference contrast requires DINOv2 ViT-L/14 `K=9`; the full panel is an
ordered secondary analysis:

```bash
for encoder in dinov2_vitl14 resnet50; do
  for context in mosaic3 mosaic5 mosaic9; do
    papers/paper_c/scripts/run_extract_embeddings.sh "$encoder" "$context"
  done
done
```

Run grouped probes on the identical complete-`K=9` population:

```bash
for encoder in dinov2_vitl14 resnet50; do
  for transform in upsample224 mosaic3 mosaic5 mosaic9; do
    for seed in 42 43 44; do
      python papers/paper_c/scripts/run_idc_probe_cv.py \
        --cache "$MBC_PRIMARY_ROOT/embeddings/${encoder}_${transform}" \
        --patch-manifest "$MANIFEST" --protocol grouped --complete-context-k 9 \
        --seed "$seed" \
        --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/idc/${encoder}/grouped-${transform}-complete-k9/seed-${seed}"
    done
  done
done
```

Do not compare the all-patch `K=1` score to the 49,798-centre `K=9` score.

## 9. BCSS preparation and extraction

Materialize the revision-pinned CC0 mirror:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env HF_HOME=/cache/huggingface \
  --mount type=bind,src="$MBC_PRIMARY_ROOT/cache",dst=/cache \
  --mount type=bind,src="$MBC_PRIMARY_ROOT/datasets",dst=/datasets \
  "$MBC_IMAGE" \
  python papers/paper_c/scripts/prepare_bcss.py \
    --output /datasets/bcss --cache-dir /cache/huggingface

python papers/paper_c/scripts/build_bcss_patient_splits.py \
  --metadata "$MBC_PRIMARY_ROOT/datasets/bcss/mirror_rows.json" \
  --output "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_patient_splits.csv"

# Archive any stride-50 manifest before rebuilding (protocol: stride 150).
if [[ -f "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv" ]]; then
  mv "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv" \
     "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.stride50.bak.csv"
  mv "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.summary.json" \
     "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.stride50.bak.summary.json" \
     2>/dev/null || true
fi

python papers/paper_c/scripts/build_bcss_centres.py \
  --images "$MBC_PRIMARY_ROOT/datasets/bcss/images" \
  --masks "$MBC_PRIMARY_ROOT/datasets/bcss/masks" \
  --metadata "$MBC_PRIMARY_ROOT/datasets/bcss/mirror_rows.json" \
  --patient-splits "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_patient_splits.csv" \
  --stride 150 \
  --output "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv"

python3 - <<'PY'
import json
from pathlib import Path
summary = Path.home() / "mbc-primary/provenance/paper-c/bcss_centres.summary.json"
print(summary.read_text())
PY
```

Expect roughly 60k–80k eligible centres at stride 150 (not ~608k). Record the
printed SHA-256 in the lock manifest.

Extract the same BCSS centres at both fields of view:

```bash
for encoder in dinov2_vitl14 resnet50; do
  papers/paper_c/scripts/run_extract_bcss_embeddings.sh "$encoder" k1
  papers/paper_c/scripts/run_extract_bcss_embeddings.sh "$encoder" k9
done
```

Then fit probes:

```bash
for seed in 42 43 44; do
  python papers/paper_c/scripts/run_bcss_probe.py \
    --cache "$MBC_PRIMARY_ROOT/embeddings/bcss_dinov2_vitl14_k1" \
    --centre-manifest "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv" \
    --seed "$seed" \
    --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/bcss/dinov2_vitl14/k1/seed-${seed}"

  python papers/paper_c/scripts/run_bcss_probe.py \
    --cache "$MBC_PRIMARY_ROOT/embeddings/bcss_dinov2_vitl14_k9" \
    --centre-manifest "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv" \
    --seed "$seed" \
    --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/bcss/dinov2_vitl14/k9/seed-${seed}"
done
```

Average the BCSS seeds as well, so inference consumes one bundle per condition:

```bash
BCSS_ROOT="$MBC_PRIMARY_ROOT/results/paper-c/bcss"
for context in k1 k9; do
  python papers/paper_c/scripts/average_seed_predictions.py \
    --prediction "$BCSS_ROOT/dinov2_vitl14/${context}/seed-42/test_predictions.csv" --seed 42 \
    --prediction "$BCSS_ROOT/dinov2_vitl14/${context}/seed-43/test_predictions.csv" --seed 43 \
    --prediction "$BCSS_ROOT/dinov2_vitl14/${context}/seed-44/test_predictions.csv" --seed 44 \
    --output "$BCSS_ROOT/dinov2_vitl14/${context}/seed_mean_predictions.csv"
done
```

## 10. Locked inference

Run the two co-primary paired whole-case bootstraps at 10,000 replicates, then
apply the Holm correction:

```bash
INF="$MBC_PRIMARY_ROOT/results/paper-c/inference"
mkdir -p "$INF"

python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$IDC_ROOT/dinov2_vitl14/random-k1/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv" \
  --left-label random-k1 --right-label grouped-k1 \
  --replicates 10000 --output "$INF/delta_protocol.json"

python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$IDC_ROOT/dinov2_vitl14/grouped-mosaic9-complete-k9/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/dinov2_vitl14/grouped-upsample224-complete-k9/seed_mean_predictions.csv" \
  --left-label grouped-k9 --right-label grouped-k1 \
  --replicates 10000 --output "$INF/delta_context.json"

python papers/paper_c/scripts/holm_coprimary.py \
  --protocol-report "$INF/delta_protocol.json" \
  --context-report  "$INF/delta_context.json" \
  --output "$INF/coprimary_holm.json"
```

The bundled protocol regime is an ordered secondary and stays outside the Holm
family:

```bash
python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$IDC_ROOT/dinov2_vitl14/random-k1-bundled/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv" \
  --left-label random-k1-bundled --right-label grouped-k1 \
  --replicates 10000 --output "$INF/delta_protocol_bundled.json"
```

For BCSS the same script treats `case_id` as the TCGA patient identifier and
must resample within institutions:

```bash
python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$MBC_PRIMARY_ROOT/results/paper-c/bcss/dinov2_vitl14/k9/seed_mean_predictions.csv" \
  --right "$MBC_PRIMARY_ROOT/results/paper-c/bcss/dinov2_vitl14/k1/seed_mean_predictions.csv" \
  --left-label bcss-k9 --right-label bcss-k1 \
  --strata-column site --replicates 10000 \
  --output "$INF/bcss_delta_context.json"
```

Ordered-secondary reliability and rank-stability reports:

```bash
python papers/paper_c/scripts/secondary_reliability.py \
  --predictions "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv" \
  --output-dir "$INF/reliability/dinov2_vitl14-grouped-k1"

RANK_ARGS=()
for encoder in dinov2_vitl14 resnet50; do
  RANK_ARGS+=(--pair "$encoder" \
    "$IDC_ROOT/${encoder}/random-k1/seed-42/summary.json" \
    "$IDC_ROOT/${encoder}/grouped-k1/seed-42/summary.json")
done
python papers/paper_c/scripts/model_rank_stability.py \
  "${RANK_ARGS[@]}" \
  --left-label random-k1 --right-label grouped-k1 \
  --output "$INF/rank_stability.json"
```

Promote only curated, non-sensitive JSON summaries to
`papers/paper_c/results/`. Raw caches, predictions and logs remain under
`$MBC_PRIMARY_ROOT` or root `results/` and are never committed.
