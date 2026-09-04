#!/usr/bin/env bash
# Run one Path A cell in the qualified container, with pinned provenance.
#
#   run_patha_server.sh <fold 0-4> <control|fair>
#
# Mirrors scripts/run_histopath_width_server.sh so Path A cells carry the same
# audit trail as the width matrix: pinned image digest, dependency-lock digest,
# dataset digest, and the five-fold manifest digest.
#
# Requires scripts/build_histopath_server_image.sh to have been run first.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <fold 0-4> <control|fair>" >&2
  exit 2
fi
FOLD="$1"
ARM="$2"

if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
  echo "Declared folds are 0-4." >&2
  exit 2
fi
case "$ARM" in
  control) TAG="termwarm" ;;
  fair)    TAG="fairwarm" ;;
  *) echo "arm must be 'control' or 'fair'." >&2; exit 2 ;;
esac

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
DATASET_SHA_FILE="$PRIMARY_ROOT/provenance/dataset_archive_sha256.txt"
CACHE_DIR="$PRIMARY_ROOT/cache"
BUNDLE_DIR="$PRIMARY_ROOT/bundles/path-a"
CELL_DIR="$PRIMARY_ROOT/results/path-a/fold${FOLD}_${TAG}"

# The five-fold paper partition ships in the repository, so it is mounted from
# the checkout rather than from a server-local copy. Its digest is verified in
# the container against the value declared in run_patha_server.py.
SPLITS_DIR="$ROOT/data/splits/histopath_kaggle"

for path in "$DATASET_DIR" "$DATASET_SHA_FILE" "$SPLITS_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Required artifact is missing: $path" >&2
    exit 1
  fi
done

# Patch-level manifests are untracked by design: they are large and rebuildable
# from the committed case-ID lists. Check before starting the container, so a
# missing rebuild costs a second rather than a queue slot.
FOLD_DIR="$SPLITS_DIR/folds/fold_${FOLD}"
MISSING_MANIFESTS=()
for name in train.csv test.csv; do
  [[ -e "$FOLD_DIR/$name" ]] || MISSING_MANIFESTS+=("$name")
done
if [[ ${#MISSING_MANIFESTS[@]} -gt 0 ]]; then
  cat >&2 <<EOF
Patch-level manifests are missing in $FOLD_DIR:
  ${MISSING_MANIFESTS[*]}

Only the case-ID lists (train_patients.csv / test_patients.csv) ship in git.
Rebuild the patch manifests losslessly from them plus the archive, once:

  python3 data/download/split_histopath_archive.py \\
    --archive-path "$DATASET_DIR" \\
    --output-dir data/splits/histopath_kaggle \\
    --mode cv --from-patient-manifest

That reads the committed case-ID lists rather than recomputing the split, which
matters because StratifiedGroupKFold is not stable across scikit-learn versions.
Afterwards confirm 'git status --short data/splits/histopath_kaggle' is clean:
any change to patient_stats.csv or split_stats.json means the archive is not the
published cohort.
EOF
  exit 1
fi

if find "$CELL_DIR" -name cv_summary.json -print -quit 2>/dev/null | grep -q .; then
  echo "A completed result already exists for fold${FOLD}/${ARM}." >&2
  echo "The declaration forbids silent reruns; move or delete it first." >&2
  exit 1
fi

mkdir -p "$CELL_DIR/results" "$CACHE_DIR" "$BUNDLE_DIR" "$PRIMARY_ROOT/logs"

DATASET_SHA="$(awk 'NR == 1 {print $1}' "$DATASET_SHA_FILE")"
RUN_LOG="$PRIMARY_ROOT/logs/path_a_fold${FOLD}_${TAG}.log"

echo "=== PaperB Path A (containerised) ==="
echo "  fold    : $FOLD"
echo "  arm     : $ARM ($TAG)"
echo "  image   : $MBC_IMAGE"
echo "  splits  : $SPLITS_DIR"
echo "  results : $CELL_DIR/results"
echo "  log     : $RUN_LOG"
echo

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
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$SPLITS_DIR",dst=/opt/mbc/data/splits/histopath_kaggle \
  --mount type=bind,src="$CELL_DIR/results",dst=/opt/mbc/results \
  --mount type=bind,src="$CACHE_DIR",dst=/cache \
  --mount type=bind,src="$BUNDLE_DIR",dst=/outputs \
  "$MBC_IMAGE" \
  python PaperB_PathA/scripts/run_patha_server.py \
    --fold "$FOLD" \
    --arm "$ARM" \
    --archive-path /datasets/histopath \
    --splits-dir /opt/mbc/data/splits/histopath_kaggle \
    --output-root /outputs \
  2>&1 | tee "$RUN_LOG"

echo
echo "Completed fold${FOLD}/${ARM}"
echo "Log:        $RUN_LOG"
echo "Results:    $CELL_DIR/results"
echo "Provenance: $BUNDLE_DIR/provenance_fold${FOLD}_${TAG}.json"
