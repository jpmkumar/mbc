#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 ENCODER {k1|k9} [extra args...]" >&2
  exit 2
fi
ENCODER="$1"; CONTEXT="$2"; shift 2
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IMAGE_ENV="$ROOT/results/server_setup/current_paperc_image.env"
[[ -f "$IMAGE_ENV" ]] || { echo "Build the Paper C image first." >&2; exit 1; }
# shellcheck disable=SC1090
source "$IMAGE_ENV"

BCSS_ROOT="$PRIMARY_ROOT/datasets/bcss"
MANIFEST="$PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv"
TOKEN_FILE="${HF_TOKEN_FILE:-$PRIMARY_ROOT/hf_token}"
CACHE_DIR="$PRIMARY_ROOT/cache"
EMB_DIR="$PRIMARY_ROOT/embeddings"
LOG_DIR="$PRIMARY_ROOT/logs"
for path in "$BCSS_ROOT/images" "$BCSS_ROOT/provenance.json" "$MANIFEST" "$TOKEN_FILE"; do
  [[ -e "$path" ]] || { echo "Required artifact not found: $path" >&2; exit 1; }
done
mkdir -p "$CACHE_DIR" "$EMB_DIR" "$LOG_DIR"
DATASET_SHA="$(python3 - "$BCSS_ROOT/provenance.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["materialized_tree_sha256"])
PY
)"
RUN_LOG="$LOG_DIR/paperc_bcss_${ENCODER}_${CONTEXT}_$(date -u +%Y%m%dT%H%M%SZ).log"

docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env HF_HOME=/cache/huggingface \
  --env "MBC_GIT_COMMIT=$MBC_GIT_COMMIT" \
  --env "MBC_IMAGE_ID=$MBC_IMAGE_ID" \
  --env "MBC_ENV_LOCK_SHA=$MBC_ENV_LOCK_SHA" \
  --env "MBC_DATASET_SHA=$DATASET_SHA" \
  --shm-size=16g \
  --mount type=bind,src="$BCSS_ROOT/images",dst=/datasets/bcss/images,readonly \
  --mount type=bind,src="$MANIFEST",dst=/datasets/bcss/centres.csv,readonly \
  --mount type=bind,src="$TOKEN_FILE",dst=/run/secrets/hf_token,readonly \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  --mount type=bind,src="$EMB_DIR",dst=/outputs \
  "$MBC_IMAGE" \
  bash -lc 'export HF_TOKEN="$(< /run/secrets/hf_token)"; exec python papers/paper_c/scripts/extract_bcss_embeddings.py "$@"' -- \
    --encoder "$ENCODER" \
    --context "$CONTEXT" \
    --images /datasets/bcss/images \
    --manifest /datasets/bcss/centres.csv \
    --output-dir /outputs \
    "$@" \
  2>&1 | tee "$RUN_LOG"

echo "Log: $RUN_LOG"
