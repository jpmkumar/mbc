#!/usr/bin/env bash
# Launch one embedding-extraction pass in the Paper C image with full provenance.
# Usage: run_extract_embeddings.sh ENCODER TRANSFORM [extra args...]
#   e.g. run_extract_embeddings.sh uni2_h upsample224
#        run_extract_embeddings.sh uni2_h mosaic3 --centre-limit 5000 --run-name smoke
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

for path in "$DATASET_DIR" "$DATASET_SHA_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required artifact not found: $path" >&2
    exit 1
  fi
done
mkdir -p "$CACHE_DIR" "$EMB_DIR" "$LOG_DIR"

DATASET_SHA="$(awk 'NR == 1 {print $1}' "$DATASET_SHA_FILE")"

# Mount the token as a read-only secret rather than embedding it in Docker's
# container configuration. The file contains only the token, with no KEY= prefix.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PRIMARY_ROOT/hf_token}"
if [[ ! -f "$HF_TOKEN_FILE" ]]; then
  echo "Hugging Face token file not found: $HF_TOKEN_FILE" >&2
  echo "Write the token only (hf_...) and chmod 600 the file." >&2
  exit 1
fi

RUN_LOG="$LOG_DIR/paperc_embed_${ENCODER}_${TRANSFORM}_$(date -u +%Y%m%dT%H%M%SZ).log"

docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env USER=mbc \
  --env LOGNAME=mbc \
  --env HOME=/tmp \
  --env XDG_CACHE_HOME=/cache \
  --env TORCH_HOME=/cache/torch \
  --env HF_HOME=/cache/huggingface \
  --env "MBC_GIT_COMMIT=$MBC_GIT_COMMIT" \
  --env "MBC_IMAGE_ID=$MBC_IMAGE_ID" \
  --env "MBC_ENV_LOCK_SHA=$MBC_ENV_LOCK_SHA" \
  --env "MBC_DATASET_SHA=$DATASET_SHA" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$HF_TOKEN_FILE",dst=/run/secrets/hf_token,readonly \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  --mount type=bind,src="$EMB_DIR",dst=/outputs \
  "$MBC_IMAGE" \
  bash -lc 'export HF_TOKEN="$(< /run/secrets/hf_token)"; exec python papers/paper_c/scripts/extract_embeddings.py "$@"' -- \
    --encoder "$ENCODER" \
    --transform "$TRANSFORM" \
    --archive-path /datasets/histopath \
    --output-dir /outputs \
    "$@" \
  2>&1 | tee "$RUN_LOG"

echo "Log: $RUN_LOG"
echo "Cache: $EMB_DIR/${ENCODER}_${TRANSFORM}"
