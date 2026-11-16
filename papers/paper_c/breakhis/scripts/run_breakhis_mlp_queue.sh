#!/usr/bin/env bash
# Confirmatory F-MLP probes after locked BreakHis embeddings.
# Usage: run_breakhis_mlp_queue.sh [ENCODER ...]
# Default: dinov2_vitl14 then resnet50; protocols grouped+random; seeds 42/43/44.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IMAGE_ENV="$ROOT/results/server_setup/current_paperc_image.env"
# shellcheck disable=SC1090
source "$IMAGE_ENV"
MANIFEST="$ROOT/data/splits/paper_c_plus/breakhis/image_manifest.csv"
INNER="$ROOT/data/splits/paper_c_plus/breakhis"
RESULTS="$PRIMARY_ROOT/results/paper-c-plus"
mkdir -p "$RESULTS" "$PRIMARY_ROOT/logs"

ENCODERS=("$@")
if [[ ${#ENCODERS[@]} -eq 0 ]]; then
  ENCODERS=(dinov2_vitl14 resnet50)
fi

NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
notify() {
  local title="$1" body="$2" priority="${3:-default}"
  [[ -s "$NTFY_TOPIC_FILE" ]] || return 0
  local topic
  topic="$(tr -d '[:space:]' < "$NTFY_TOPIC_FILE")"
  [[ -n "$topic" ]] || return 0
  curl -fsS -H "Title: $title" -H "Priority: $priority" -H "Tags: paper-c-plus" \
    -d "$body" "https://ntfy.sh/${topic}" >/dev/null 2>&1 || true
}

run_one() {
  local encoder="$1" protocol="$2" seed="$3"
  local cache="$PRIMARY_ROOT/embeddings/breakhis_${encoder}_upsample224"
  local out="$RESULTS/${encoder}/fmlp-${protocol}/seed-${seed}"
  if [[ -f "$out/summary.json" ]]; then
    echo "skip existing $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  notify "C+ F-MLP $encoder $protocol s$seed" "starting"
  docker run --rm \
    --gpus device=0 \
    --user "$(id -u):$(id -g)" \
    --env USER=mbc --env LOGNAME=mbc --env HOME=/tmp \
    --shm-size=8g \
    --mount type=bind,src="$cache",dst=/cache,readonly \
    --mount type=bind,src="$ROOT",dst=/opt/mbc,readonly \
    --mount type=bind,src="$(dirname "$out")",dst=/outputs \
    -w /opt/mbc \
    "$MBC_IMAGE" \
    python papers/paper_c/breakhis/scripts/run_breakhis_mlp_cv.py \
      --cache /cache \
      --manifest /opt/mbc/data/splits/paper_c_plus/breakhis/image_manifest.csv \
      --inner-case-splits /opt/mbc/data/splits/paper_c_plus/breakhis \
      --protocol "$protocol" \
      --seed "$seed" \
      --output-dir "/outputs/$(basename "$out")"
  notify "C+ F-MLP $encoder $protocol s$seed DONE" "$(basename "$out")"
}

notify "C+ F-MLP queue started" "${ENCODERS[*]} × grouped/random × 42/43/44" high
for encoder in "${ENCODERS[@]}"; do
  for protocol in grouped random; do
    for seed in 42 43 44; do
      run_one "$encoder" "$protocol" "$seed"
    done
  done
done
notify "C+ F-MLP queue COMPLETE" "results under $RESULTS" high
