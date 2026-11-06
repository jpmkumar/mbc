#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 ENCODER" >&2
  exit 2
fi
ENCODER="$1"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IMAGE_ENV="$ROOT/results/server_setup/current_paperc_image.env"
[[ -f "$IMAGE_ENV" ]] || { echo "Build the Paper C image first." >&2; exit 1; }
# shellcheck disable=SC1090
source "$IMAGE_ENV"

DATASET_DIR="${MBC_DATASET_DIR:-$PRIMARY_ROOT/datasets/breast-histopathology-images}"
CACHE_DIR="$PRIMARY_ROOT/cache"
TOKEN_FILE="${HF_TOKEN_FILE:-$PRIMARY_ROOT/hf_token}"
PROVENANCE_DIR="$PRIMARY_ROOT/provenance/paper-c"
[[ -e "$DATASET_DIR" ]] || { echo "Required artifact not found: $DATASET_DIR" >&2; exit 1; }

# Overlay host Paper C sources onto the image tree so uncommitted redesigns
# (ungated DINOv2 / ResNet loaders) are used without rebuilding the image.
DOCKER_MOUNTS=(
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly
  --mount type=bind,src="$CACHE_DIR",dst=/cache
  --mount type=bind,src="$ROOT/papers/paper_c",dst=/opt/mbc/papers/paper_c,readonly
)
INNER_PREFIX='exec'
if [[ -f "$TOKEN_FILE" ]]; then
  DOCKER_MOUNTS+=(--mount type=bind,src="$TOKEN_FILE",dst=/run/secrets/hf_token,readonly)
  INNER_PREFIX='export HF_TOKEN="$(< /run/secrets/hf_token)"; exec'
else
  echo "No HF token file; confirmatory panel does not require one ($ENCODER)" >&2
fi
mkdir -p "$CACHE_DIR" "$PROVENANCE_DIR" \
  "$CACHE_DIR/torch" "$CACHE_DIR/torchinductor" "$CACHE_DIR/triton" "$CACHE_DIR/torchhub"
REPORT="$PROVENANCE_DIR/encoder_smoke_${ENCODER}.json"

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
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${DOCKER_MOUNTS[@]}" \
  "$MBC_IMAGE" \
  bash -lc "$INNER_PREFIX"' python papers/paper_c/scripts/smoke_paperc_encoder.py "$@"' -- \
    --encoder "$ENCODER" \
    --archive-path /datasets/histopath \
  | tee "$REPORT"

python3 - "$REPORT" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
start = text.find("{")
report = json.loads(text[start:])
if report.get("status") != "PASS":
    raise SystemExit("Qualification report did not PASS.")
PY
echo "PASS report: $REPORT"
