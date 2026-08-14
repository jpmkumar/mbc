#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 FOLD N_QUBITS" >&2
  exit 2
fi
FOLD="$1"
N_QUBITS="$2"
if [[ ! "$FOLD" =~ ^[1-4]$ ]] || [[ ! "$N_QUBITS" =~ ^(4|8|12)$ ]]; then
  echo "Declared cells are folds 1-4 and widths 4, 8, 12." >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IMAGE_ENV="$ROOT/results/server_setup/current_server_image.env"
if [[ ! -f "$IMAGE_ENV" ]]; then
  echo "Run scripts/build_histopath_server_image.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$IMAGE_ENV"

DATASET_DIR="${MBC_DATASET_DIR:-$PRIMARY_ROOT/datasets/breast-histopathology-images}"
SPLITS_DIR="${MBC_SPLITS_DIR:-$PRIMARY_ROOT/splits/histopath}"
DATASET_SHA_FILE="$PRIMARY_ROOT/provenance/dataset_archive_sha256.txt"
SPLIT_SHA_FILE="$PRIMARY_ROOT/provenance/split_checksums.json"
CACHE_DIR="$PRIMARY_ROOT/cache"
BUNDLE_DIR="$PRIMARY_ROOT/bundles"
CELL_DIR="$PRIMARY_ROOT/results/server-width/q${N_QUBITS}/fold${FOLD}"

for path in "$DATASET_DIR" "$SPLITS_DIR" "$DATASET_SHA_FILE" "$SPLIT_SHA_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required server artifact is missing: $path" >&2
    exit 1
  fi
done
if [[ -e "$CELL_DIR/results/histopath/cv_summary.json" ]]; then
  echo "A completed result already exists for q${N_QUBITS}/fold${FOLD}." >&2
  echo "The declared protocol forbids silent reruns." >&2
  exit 1
fi

mkdir -p \
  "$CELL_DIR/results" \
  "$CACHE_DIR" \
  "$BUNDLE_DIR" \
  "$PRIMARY_ROOT/logs"

DATASET_SHA="$(awk 'NR == 1 {print $1}' "$DATASET_SHA_FILE")"
SPLIT_SHA="$(sha256sum "$SPLIT_SHA_FILE" | awk '{print $1}')"
RUN_LOG="$PRIMARY_ROOT/logs/server_width_q${N_QUBITS}_fold${FOLD}.log"

set -o pipefail
docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env USER=mbc \
  --env LOGNAME=mbc \
  --env HOME=/tmp \
  --env XDG_CACHE_HOME=/cache \
  --env TORCH_HOME=/cache/torch \
  --env "MBC_GIT_COMMIT=$MBC_GIT_COMMIT" \
  --env "MBC_IMAGE_ID=$MBC_IMAGE_ID" \
  --env "MBC_ENV_LOCK_SHA=$MBC_ENV_LOCK_SHA" \
  --env "MBC_DATASET_SHA=$DATASET_SHA" \
  --env "MBC_SPLIT_MANIFEST_SHA=$SPLIT_SHA" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$SPLITS_DIR",dst=/opt/mbc/data/splits/histopath \
  --mount type=bind,src="$CELL_DIR/results",dst=/opt/mbc/results \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  --mount type=bind,src="$BUNDLE_DIR",dst=/outputs \
  "$MBC_IMAGE" \
  python scripts/run_histopath_width_server.py \
    --fold "$FOLD" \
    --n-qubits "$N_QUBITS" \
    --archive-path /datasets/histopath \
    --splits-dir /opt/mbc/data/splits/histopath \
    --output-root /outputs \
  2>&1 | tee "$RUN_LOG"

echo "Completed q${N_QUBITS}/fold${FOLD}"
echo "Log: $RUN_LOG"
echo "Results: $CELL_DIR/results/histopath"

