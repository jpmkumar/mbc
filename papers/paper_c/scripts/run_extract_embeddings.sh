#!/usr/bin/env bash
# Launch one embedding-extraction pass in the Paper C image with full provenance.
# Usage: run_extract_embeddings.sh ENCODER TRANSFORM [extra args...]
#   e.g. run_extract_embeddings.sh uni upsample224
#        run_extract_embeddings.sh uni mosaic3 --limit 5000
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 ENCODER TRANSFORM [extra args...]" >&2
  exit 2
fi
ENCODER="$1"; TRANSFORM="$2"; shift 2

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IMAGE_ENV="$ROOT/results/server_setup/current_paperc_image.env"
if [[ ! -f "$IMAGE_ENV" ]]; then
  echo "Run papers/paper_c/scripts/build_paperc_image.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$IMAGE_ENV"

DATASET_DIR="${MBC_DATASET_DIR:-$PRIMARY_ROOT/datasets/breast-histopathology-images}"
DATASET_SHA_FILE="$PRIMARY_ROOT/provenance/dataset_archive_sha256.txt"
CACHE_DIR="$PRIMARY_ROOT/cache"
EMB_DIR="$PRIMARY_ROOT/embeddings"
LOG_DIR="$PRIMARY_ROOT/logs"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset not found: $DATASET_DIR" >&2
  exit 1
fi
mkdir -p "$CACHE_DIR" "$EMB_DIR" "$LOG_DIR"

DATASET_SHA=""
[[ -f "$DATASET_SHA_FILE" ]] && DATASET_SHA="$(awk 'NR == 1 {print $1}' "$DATASET_SHA_FILE")"

# HF_TOKEN is required for the gated encoders. Keep it in $PRIMARY_ROOT/.env,
# which is outside the repository. Never commit it.
if [[ -f "$PRIMARY_ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$PRIMARY_ROOT/.env"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. UNI and Virchow are gated and will fail to download." >&2
  echo "Put HF_TOKEN=... in $PRIMARY_ROOT/.env" >&2
  exit 1
fi

RUN_LOG="$LOG_DIR/paperc_embed_${ENCODER}_${TRANSFORM}.log"

docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env USER=mbc \
  --env LOGNAME=mbc \
  --env HOME=/tmp \
  --env XDG_CACHE_HOME=/cache \
  --env TORCH_HOME=/cache/torch \
  --env HF_HOME=/cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "MBC_GIT_COMMIT=$MBC_GIT_COMMIT" \
  --env "MBC_IMAGE_ID=$MBC_IMAGE_ID" \
  --env "MBC_ENV_LOCK_SHA=$MBC_ENV_LOCK_SHA" \
  --env "MBC_DATASET_SHA=$DATASET_SHA" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  --mount type=bind,src="$EMB_DIR",dst=/outputs \
  "$MBC_IMAGE" \
  python papers/paper_c/scripts/extract_embeddings.py \
    --encoder "$ENCODER" \
    --transform "$TRANSFORM" \
    --archive-path /datasets/histopath \
    --output-dir /outputs \
    "$@" \
  2>&1 | tee "$RUN_LOG"

echo "Log: $RUN_LOG"
echo "Cache: $EMB_DIR/${ENCODER}_${TRANSFORM}"
