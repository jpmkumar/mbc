#!/usr/bin/env bash
# Run one Path A cell: a single fold under one arm.
#
#   run_patha_fold.sh <fold 0-4> <control|fair>
#
# control -> --no-stage-init-from-best  (published schedule, tagged termwarm)
# fair    -> --stage-init-from-best     (fair warmup,        tagged fairwarm)
#
# Everything else is held at the published five-fold values; see
# PaperB_PathA/README.md and preregistration/staged_hybrid_fair_warmup_protocol.md.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <fold 0-4> <control|fair>" >&2
  exit 2
fi

FOLD="$1"
ARM="$2"

case "$FOLD" in
  0|1|2|3|4) ;;
  *) echo "fold must be 0-4, got '$FOLD'" >&2; exit 2 ;;
esac

case "$ARM" in
  control) STAGE_FLAG="--no-stage-init-from-best"; TAG="termwarm" ;;
  fair)    STAGE_FLAG="--stage-init-from-best";    TAG="fairwarm" ;;
  *) echo "arm must be 'control' or 'fair', got '$ARM'" >&2; exit 2 ;;
esac

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PATHA_DIR="$ROOT/PaperB_PathA"
RESULTS_DIR="$PATHA_DIR/results/fold${FOLD}_${TAG}"
LOG_DIR="$PATHA_DIR/results/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# The five-fold paper partition, not the server width partition.
SPLITS_DIR="${MBC_PATHA_SPLITS:-$ROOT/data/splits/histopath_kaggle}"
ARCHIVE_PATH="${MBC_PATHA_ARCHIVE:-}"

if [[ ! -d "$SPLITS_DIR" ]]; then
  echo "Splits directory not found: $SPLITS_DIR" >&2
  echo "Set MBC_PATHA_SPLITS if the manifest lives elsewhere." >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$LOG_DIR/fold${FOLD}_${TAG}_${STAMP}.log"

CMD=(python scripts/train_histopath_cv.py
  --fold "$FOLD"
  --experiment E3
  --seed 42
  --n-qubits 8
  --splits-dir "$SPLITS_DIR"
  "$STAGE_FLAG")

if [[ -n "$ARCHIVE_PATH" ]]; then
  CMD+=(--archive-path "$ARCHIVE_PATH")
fi

echo "=== PaperB Path A ==="
echo "  fold      : $FOLD"
echo "  arm       : $ARM ($TAG)"
echo "  splits    : $SPLITS_DIR"
echo "  results   : $RESULTS_DIR"
echo "  log       : $RUN_LOG"
echo "  command   : ${CMD[*]}"
echo

if [[ "${MBC_PATHA_DRY_RUN:-0}" == "1" ]]; then
  echo "(dry run; nothing launched)"
  exit 0
fi

# Record provenance next to the artifacts so the arm is auditable later.
{
  echo "utc=$STAMP"
  echo "fold=$FOLD"
  echo "arm=$ARM"
  echo "tag=$TAG"
  echo "stage_flag=$STAGE_FLAG"
  echo "splits_dir=$SPLITS_DIR"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "command=${CMD[*]}"
} > "$RESULTS_DIR/run_provenance.txt"

"${CMD[@]}" 2>&1 | tee "$RUN_LOG"

echo
echo "Completed fold ${FOLD} / ${ARM}."
echo "Artifacts: $RESULTS_DIR"
