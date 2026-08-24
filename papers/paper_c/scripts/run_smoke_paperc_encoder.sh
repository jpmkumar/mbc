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
for path in "$DATASET_DIR" "$TOKEN_FILE"; do
  [[ -e "$path" ]] || { echo "Required artifact not found: $path" >&2; exit 1; }
done
mkdir -p "$CACHE_DIR" "$PROVENANCE_DIR"
REPORT="$PROVENANCE_DIR/encoder_smoke_${ENCODER}.json"

docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env HF_HOME=/cache/huggingface \
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$TOKEN_FILE",dst=/run/secrets/hf_token,readonly \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  "$MBC_IMAGE" \
  bash -lc 'export HF_TOKEN="$(< /run/secrets/hf_token)"; exec python papers/paper_c/scripts/smoke_paperc_encoder.py "$@"' -- \
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
