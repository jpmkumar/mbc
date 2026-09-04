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
# The rebuild rewrites patient_stats.csv and split_stats.json as well as the
# patch manifests. patient_stats.csv is the per-case fingerprint and must be
# byte-identical. split_stats.json is allowed to differ in archive_path alone,
# which records wherever the archive was mounted and is not cohort-defining.
python3 - <<'PY' || exit 1
import json
import subprocess
import sys

PREFIX = "data/splits/histopath_kaggle"
IGNORED = {"archive_path"}


def committed(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"])


changed = subprocess.check_output(
    ["git", "status", "--porcelain", PREFIX], text=True
).splitlines()
tracked_changed = [line[3:].strip() for line in changed if line[:2].strip() in {"M", "MM"}]

if not tracked_changed:
    print("  tracked split files unchanged")
    sys.exit(0)

fatal = []
for rel in tracked_changed:
    if rel.endswith("split_stats.json"):
        before = json.loads(committed(rel))
        after = json.loads(open(rel, "rb").read())
        diff = {
            key
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        } - IGNORED
        if diff:
            fatal.append(f"{rel}: cohort fields differ {sorted(diff)}")
        else:
            print(f"  {rel}: only archive_path differs (expected, not cohort-defining)")
            print(f"      committed {before.get('archive_path')!r} -> now {after.get('archive_path')!r}")
    else:
        fatal.append(f"{rel}: tracked partition file changed")

if fatal:
    print("REFUSING TO CONTINUE:", file=sys.stderr)
    for item in fatal:
        print(f"  {item}", file=sys.stderr)
    print(
        "\nThe archive is not the published cohort. Do not run Path A against it.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

DIGEST="$(python3 PaperB_PathA/scripts/run_patha_server.py --print-manifest-sha)"
EXPECTED="091fe005155a88a5397afb5d6d381d397cf3c4da00d6f7efc5a0a487fab1963e"
if [[ "$DIGEST" != "$EXPECTED" ]]; then
  echo "REFUSING TO CONTINUE: partition digest changed." >&2
  echo "  observed $DIGEST" >&2
  echo "  expected $EXPECTED" >&2
  exit 1
fi
echo "  partition digest matches: $DIGEST"

echo
echo "To restore the committed archive_path in split_stats.json (optional, keeps"
echo "the tracked tree clean for the image builder):"
echo "  git checkout -- data/splits/histopath_kaggle/split_stats.json"

echo
echo "Rebuilt. Patch manifests present for:"
for fold in 0 1 2 3 4; do
  dir="$SPLITS_DIR/folds/fold_${fold}"
  train_ok=$([[ -e "$dir/train.csv" ]] && echo yes || echo MISSING)
  test_ok=$([[ -e "$dir/test.csv" ]] && echo yes || echo MISSING)
  printf '  fold %s  train.csv=%s  test.csv=%s\n' "$fold" "$train_ok" "$test_ok"
done
