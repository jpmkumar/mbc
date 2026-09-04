#!/usr/bin/env bash
# Rebuild the patch-level fold manifests for the five-fold paper partition.
#
#   rebuild_patch_manifests.sh
#
# Only the case-ID lists (train_patients.csv / test_patients.csv) ship in git.
# The patch-level manifests (train.csv / test.csv, ~13 MB each) are untracked
# because they are large and rebuildable losslessly from the case-ID lists plus
# the archive. A fresh checkout therefore needs this once before training.
#
# Runs inside the qualified container: the rebuild imports src.data, which pulls
# in torchvision, so it cannot run against a bare host interpreter.
#
# --from-patient-manifest reads the committed case-ID lists instead of
# recomputing the split. That distinction is essential: StratifiedGroupKFold is
# not stable across scikit-learn versions, so recomputing would silently produce
# a different partition from the published one.
set -euo pipefail

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
SPLITS_DIR="$ROOT/data/splits/histopath_kaggle"

for path in "$DATASET_DIR" "$SPLITS_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Required artifact is missing: $path" >&2
    exit 1
  fi
done

echo "=== Rebuilding patch-level manifests ==="
echo "  image   : $MBC_IMAGE"
echo "  archive : $DATASET_DIR"
echo "  splits  : $SPLITS_DIR"
echo

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env USER=mbc \
  --env LOGNAME=mbc \
  --env HOME=/tmp \
  --env XDG_CACHE_HOME=/cache \
  --mount type=bind,src="$DATASET_DIR",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$SPLITS_DIR",dst=/opt/mbc/data/splits/histopath_kaggle \
  --mount type=bind,src="$PRIMARY_ROOT/cache",dst=/cache \
  "$MBC_IMAGE" \
  python data/download/split_histopath_archive.py \
    --archive-path /datasets/histopath \
    --output-dir data/splits/histopath_kaggle \
    --mode cv --from-patient-manifest

echo
echo "=== Verifying the partition was not altered ==="
# The rebuild also rewrites patient_stats.csv and split_stats.json. Any change
# to those tracked files means the archive is not the published cohort, so the
# comparison against the published results would be invalid.
if [[ -n "$(git status --porcelain data/splits/histopath_kaggle)" ]]; then
  echo "REFUSING TO CONTINUE: tracked split files changed." >&2
  git status --short data/splits/histopath_kaggle >&2
  echo >&2
  echo "The archive is not the published cohort. Do not run Path A against it." >&2
  exit 1
fi
echo "  tracked split files unchanged (archive matches the published cohort)"

DIGEST="$(python3 PaperB_PathA/scripts/run_patha_server.py --print-manifest-sha)"
echo "  manifest digest: $DIGEST"

echo
echo "Rebuilt. Patch manifests present for:"
for fold in 0 1 2 3 4; do
  dir="$SPLITS_DIR/folds/fold_${fold}"
  train_ok=$([[ -e "$dir/train.csv" ]] && echo yes || echo MISSING)
  test_ok=$([[ -e "$dir/test.csv" ]] && echo yes || echo MISSING)
  printf '  fold %s  train.csv=%s  test.csv=%s\n' "$fold" "$train_ok" "$test_ok"
done
