#!/usr/bin/env bash
# Launch one embedding-extraction pass in the Paper C image with full provenance.
# Usage: run_extract_embeddings.sh ENCODER TRANSFORM [extra args...]
#   e.g. run_extract_embeddings.sh dinov2_vitl14 upsample224
#        run_extract_embeddings.sh resnet50 mosaic3 --centre-limit 5000 --run-name smoke
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

# Mount the token as a read-only secret when present. Gated encoders require it;
# ungated extraction (ResNet-50 / DINOv2) may proceed without a token file.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PRIMARY_ROOT/hf_token}"
mkdir -p "$CACHE_DIR/torch" "$CACHE_DIR/torchinductor" "$CACHE_DIR/triton" "$CACHE_DIR/torchhub"
DOCKER_MOUNTS=(
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly
  --mount type=bind,src="$CACHE_DIR",dst=/cache
  --mount type=bind,src="$EMB_DIR",dst=/outputs
  --mount type=bind,src="$ROOT/papers/paper_c",dst=/opt/mbc/papers/paper_c,readonly
)
INNER_PREFIX='exec'
if [[ -f "$HF_TOKEN_FILE" ]]; then
  DOCKER_MOUNTS+=(--mount type=bind,src="$HF_TOKEN_FILE",dst=/run/secrets/hf_token,readonly)
  INNER_PREFIX='export HF_TOKEN="$(< /run/secrets/hf_token)"; exec'
else
  echo "No HF token file; confirmatory panel does not require one ($ENCODER)" >&2
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
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  --env TRITON_CACHE_DIR=/cache/triton \
  --env "MBC_GIT_COMMIT=$MBC_GIT_COMMIT" \
  --env "MBC_IMAGE_ID=$MBC_IMAGE_ID" \
  --env "MBC_ENV_LOCK_SHA=$MBC_ENV_LOCK_SHA" \
  --env "MBC_DATASET_SHA=$DATASET_SHA" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${DOCKER_MOUNTS[@]}" \
  "$MBC_IMAGE" \
  bash -lc "$INNER_PREFIX"' python papers/paper_c/scripts/extract_embeddings.py "$@"' -- \
    --encoder "$ENCODER" \
    --transform "$TRANSFORM" \
    --archive-path /datasets/histopath \
    --output-dir /outputs \
    "$@" \
  2>&1 | tee "$RUN_LOG"

echo "Log: $RUN_LOG"
# Resolve the actual output directory from the trailing --run-name if present.
RUN_NAME=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--run-name" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    RUN_NAME="${ARGS[$((i + 1))]}"
  fi
done
if [[ -n "$RUN_NAME" ]]; then
  echo "Cache: $EMB_DIR/${ENCODER}_${TRANSFORM}__${RUN_NAME}"
else
  echo "Cache: $EMB_DIR/${ENCODER}_${TRANSFORM}"
fi
