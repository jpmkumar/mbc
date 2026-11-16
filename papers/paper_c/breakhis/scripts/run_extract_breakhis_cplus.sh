#!/usr/bin/env bash
# Extract one Paper C+ BreakHis encoder cache in the Paper C image.
# Usage: run_extract_breakhis_cplus.sh ENCODER
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ENCODER" >&2
  exit 2
fi
ENCODER="$1"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IMAGE_ENV="$ROOT/results/server_setup/current_paperc_image.env"
# shellcheck disable=SC1090
source "$IMAGE_ENV"

EXTRACTED="${MBC_BREAKHIS_ROOT:-$ROOT/papers/paper_d/data/BreaKHis_v1}"
MANIFEST="$ROOT/data/splits/paper_c_plus/breakhis/image_manifest.csv"
ARCHIVE_SHA_FILE="$PRIMARY_ROOT/provenance/breakhis_archive_sha256.txt"
if [[ -f "$ARCHIVE_SHA_FILE" ]]; then
  DATASET_SHA="$(awk 'NR==1{print $1}' "$ARCHIVE_SHA_FILE")"
else
  DATASET_SHA="d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640"
fi
CACHE_DIR="$PRIMARY_ROOT/cache"
EMB_DIR="$PRIMARY_ROOT/embeddings"
LOG_DIR="$PRIMARY_ROOT/logs"
mkdir -p "$CACHE_DIR" "$EMB_DIR" "$LOG_DIR" "$CACHE_DIR/torch" "$CACHE_DIR/torchhub"

RUN_LOG="$LOG_DIR/paperc_plus_embed_${ENCODER}_$(date -u +%Y%m%dT%H%M%SZ).log"
docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env USER=mbc --env LOGNAME=mbc --env HOME=/tmp \
  --env XDG_CACHE_HOME=/cache \
  --env TORCH_HOME=/cache/torch \
  --env HF_HOME=/cache/huggingface \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  --env "MBC_GIT_COMMIT=$MBC_GIT_COMMIT" \
  --env "MBC_IMAGE_ID=$MBC_IMAGE_ID" \
  --env "MBC_ENV_LOCK_SHA=$MBC_ENV_LOCK_SHA" \
  --env "MBC_DATASET_SHA=$DATASET_SHA" \
  --shm-size=16g \
  --mount type=bind,src="$EXTRACTED",dst=/data/breakhis,readonly \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  --mount type=bind,src="$EMB_DIR",dst=/outputs \
  --mount type=bind,src="$ROOT",dst=/opt/mbc,readonly \
  -w /opt/mbc \
  "$MBC_IMAGE" \
  python papers/paper_c/breakhis/scripts/extract_breakhis_embeddings.py \
    --encoder "$ENCODER" \
    --extracted-root /data/breakhis \
    --manifest /opt/mbc/data/splits/paper_c_plus/breakhis/image_manifest.csv \
    --output-dir /outputs \
    --hf-cache /cache/huggingface \
  2>&1 | tee "$RUN_LOG"

echo "Log: $RUN_LOG"
echo "Cache: $EMB_DIR/breakhis_${ENCODER}_upsample224"
