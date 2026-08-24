# Paper C — RTX A4000 execution runbook

This runbook executes the preregistered classical foundation-model study on the
campus RTX A4000. Paper B images, caches and result values are not reused.

## Current gate

Do not run labelled probes until all five encoders pass model-loading,
dimension, memory and fp16-equivalence checks; the dependency lock hash is
hard-pinned; BCSS/IDC manifests are frozen; and
`preregistration/paper_c_protocol.md` is marked `LOCKED`.

The three original P0 faults are addressed in code:

- Virchow2 returns token-level output and pools `CLS || mean(patch)` after
  excluding CLS plus four register tokens;
- caches and manifests join only on unique `filepath`;
- the image builder refuses to run until
  `docker/requirements-paperc.lock` has a hard-coded qualified SHA.

## 1. Server preparation

```bash
cd "$HOME/projects/mbc"
git pull --ff-only
export MBC_PRIMARY_ROOT="$HOME/mbc-primary"
mkdir -p "$MBC_PRIMARY_ROOT"/{cache,embeddings,logs,provenance/paper-c,results/paper-c}
```

Create a Hugging Face token file outside the repository:

```bash
umask 077
printf '%s\n' 'hf_REPLACE_WITH_YOUR_TOKEN' > "$MBC_PRIMARY_ROOT/hf_token"
```

The Hugging Face account must have individually accepted the gated UNI2-h and
Virchow2 licences. The weights and embeddings cannot be redistributed.

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

The model registry pins full Hugging Face revisions:

```bash
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
  papers/paper_c/scripts/run_smoke_paperc_encoder.sh "$encoder"
done
```

Expected embedding dimensions:

- UNI2-h: 1536
- Virchow2: 2560
- Phikon-v2: 1024
- DINOv2 ViT-L/14: 1024
- ResNet-50: 2048

The report must show `PASS`, finite output and a batch peak below the 16 GB
A4000 limit. Lower the model's `batch_size_a4000` in the registry if necessary,
rerun every smoke test, and update the registry hash in the preregistration
before lock.

Qualify cache precision on the same 1,000 centres:

```bash
papers/paper_c/scripts/run_extract_embeddings.sh uni2_h upsample224 \
  --centre-limit 1000 --run-name precision-fp16
papers/paper_c/scripts/run_extract_embeddings.sh uni2_h upsample224 \
  --centre-limit 1000 --run-name precision-fp32 --fp32

docker run --rm \
  --mount type=bind,src="$MBC_PRIMARY_ROOT/embeddings",dst=/outputs,readonly \
  "$MBC_IMAGE" \
  python papers/paper_c/scripts/compare_embedding_precision.py \
    --fp16-cache /outputs/uni2_h_upsample224__precision-fp16 \
    --fp32-cache /outputs/uni2_h_upsample224__precision-fp32
```

Repeat for each encoder. Smoke caches have unique namespaces and never require
deleting a completed production cache.

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
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
  papers/paper_c/scripts/run_extract_embeddings.sh "$encoder" upsample224
done
```

Every full cache must contain 277,524 rows and report a complete provenance file.
The cache row order is not a split order.

## 7. IDC protocol experiment

For each encoder, run paired random and grouped out-of-fold probes:

```bash
MANIFEST="$MBC_PRIMARY_ROOT/provenance/paper-c/idc_patch_manifest.csv"
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
  CACHE="$MBC_PRIMARY_ROOT/embeddings/${encoder}_upsample224"
  for seed in 42 43 44; do
    python papers/paper_c/scripts/run_idc_probe_cv.py \
      --cache "$CACHE" --patch-manifest "$MANIFEST" --protocol random \
      --seed "$seed" \
      --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/idc/${encoder}/random-k1/seed-${seed}"
    python papers/paper_c/scripts/run_idc_probe_cv.py \
      --cache "$CACHE" --patch-manifest "$MANIFEST" --protocol grouped \
      --seed "$seed" \
      --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/idc/${encoder}/grouped-k1/seed-${seed}"
  done
done
```

Average each condition's seed-42/43/44 OOF files before any bootstrap or metric
comparison:

```bash
IDC_ROOT="$MBC_PRIMARY_ROOT/results/paper-c/idc"
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
  for condition in random-k1 grouped-k1; do
    python papers/paper_c/scripts/average_seed_predictions.py \
      --prediction "$IDC_ROOT/${encoder}/${condition}/seed-42/oof_predictions.csv" --seed 42 \
      --prediction "$IDC_ROOT/${encoder}/${condition}/seed-43/oof_predictions.csv" --seed 43 \
      --prediction "$IDC_ROOT/${encoder}/${condition}/seed-44/oof_predictions.csv" --seed 44 \
      --output "$IDC_ROOT/${encoder}/${condition}/seed_mean_predictions.csv"
  done
done
```

## 8. IDC context extraction and probes

The confirmatory reference contrast requires UNI2-h `K=9`; the full panel is an
ordered secondary analysis:

```bash
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
  for context in mosaic3 mosaic5 mosaic9; do
    papers/paper_c/scripts/run_extract_embeddings.sh "$encoder" "$context"
  done
done
```

Run grouped probes on the identical complete-`K=9` population:

```bash
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
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

python papers/paper_c/scripts/build_bcss_centres.py \
  --images "$MBC_PRIMARY_ROOT/datasets/bcss/images" \
  --masks "$MBC_PRIMARY_ROOT/datasets/bcss/masks" \
  --metadata "$MBC_PRIMARY_ROOT/datasets/bcss/mirror_rows.json" \
  --patient-splits "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_patient_splits.csv" \
  --output "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv"
```

Extract the same BCSS centres at both fields of view:

```bash
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
  papers/paper_c/scripts/run_extract_bcss_embeddings.sh "$encoder" k1
  papers/paper_c/scripts/run_extract_bcss_embeddings.sh "$encoder" k9
done
```

Then fit probes:

```bash
for seed in 42 43 44; do
  python papers/paper_c/scripts/run_bcss_probe.py \
    --cache "$MBC_PRIMARY_ROOT/embeddings/bcss_uni2_h_k1" \
    --centre-manifest "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv" \
    --seed "$seed" \
    --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/bcss/uni2_h/k1/seed-${seed}"

  python papers/paper_c/scripts/run_bcss_probe.py \
    --cache "$MBC_PRIMARY_ROOT/embeddings/bcss_uni2_h_k9" \
    --centre-manifest "$MBC_PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv" \
    --seed "$seed" \
    --output-dir "$MBC_PRIMARY_ROOT/results/paper-c/bcss/uni2_h/k9/seed-${seed}"
done
```

Average the BCSS seeds as well, so inference consumes one bundle per condition:

```bash
BCSS_ROOT="$MBC_PRIMARY_ROOT/results/paper-c/bcss"
for context in k1 k9; do
  python papers/paper_c/scripts/average_seed_predictions.py \
    --prediction "$BCSS_ROOT/uni2_h/${context}/seed-42/test_predictions.csv" --seed 42 \
    --prediction "$BCSS_ROOT/uni2_h/${context}/seed-43/test_predictions.csv" --seed 43 \
    --prediction "$BCSS_ROOT/uni2_h/${context}/seed-44/test_predictions.csv" --seed 44 \
    --output "$BCSS_ROOT/uni2_h/${context}/seed_mean_predictions.csv"
done
```

## 10. Locked inference

Run the two co-primary paired whole-case bootstraps at 10,000 replicates, then
apply the Holm correction:

```bash
INF="$MBC_PRIMARY_ROOT/results/paper-c/inference"
mkdir -p "$INF"

python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$IDC_ROOT/uni2_h/random-k1/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/uni2_h/grouped-k1/seed_mean_predictions.csv" \
  --left-label random-k1 --right-label grouped-k1 \
  --replicates 10000 --output "$INF/delta_protocol.json"

python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$IDC_ROOT/uni2_h/grouped-mosaic9-complete-k9/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/uni2_h/grouped-upsample224-complete-k9/seed_mean_predictions.csv" \
  --left-label grouped-k9 --right-label grouped-k1 \
  --replicates 10000 --output "$INF/delta_context.json"

python papers/paper_c/scripts/holm_coprimary.py \
  --protocol-report "$INF/delta_protocol.json" \
  --context-report  "$INF/delta_context.json" \
  --output "$INF/coprimary_holm.json"
```

For BCSS the same script treats `case_id` as the TCGA patient identifier and
must resample within institutions:

```bash
python papers/paper_c/scripts/paired_case_bootstrap.py \
  --left  "$MBC_PRIMARY_ROOT/results/paper-c/bcss/uni2_h/k9/seed_mean_predictions.csv" \
  --right "$MBC_PRIMARY_ROOT/results/paper-c/bcss/uni2_h/k1/seed_mean_predictions.csv" \
  --left-label bcss-k9 --right-label bcss-k1 \
  --strata-column site --replicates 10000 \
  --output "$INF/bcss_delta_context.json"
```

Ordered-secondary reliability and rank-stability reports:

```bash
python papers/paper_c/scripts/secondary_reliability.py \
  --predictions "$IDC_ROOT/uni2_h/grouped-k1/seed_mean_predictions.csv" \
  --output-dir "$INF/reliability/uni2_h-grouped-k1"

RANK_ARGS=()
for encoder in uni2_h virchow2 phikon_v2 dinov2_vitl14 resnet50; do
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
