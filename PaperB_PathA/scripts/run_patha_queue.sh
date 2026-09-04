#!/usr/bin/env bash
# Run all ten Path A cells one at a time: five folds x two arms.
#
#   run_patha_queue.sh [--dry-run] [--no-container]
#
# Runs in the qualified container by default, so each cell records the pinned
# image, dependency-lock, dataset and split-manifest digests. --no-container
# falls back to bare Python, which records only the git commit and is not
# equivalent provenance; use it for debugging, not for reported runs.
#
# Arms are interleaved fold by fold so that calendar or thermal drift cannot be
# confounded with the arm. Stops on the first failure so a partial queue is
# never silently reported as complete.
set -euo pipefail

DRY_RUN=0
USE_CONTAINER=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-container) USE_CONTAINER=0 ;;
    -h|--help) echo "Usage: $0 [--dry-run] [--no-container]"; exit 0 ;;
    *) echo "Usage: $0 [--dry-run] [--no-container]" >&2; exit 2 ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PATHA_DIR="$ROOT/PaperB_PathA"
if [[ "$USE_CONTAINER" == "1" ]]; then
  LAUNCHER="$PATHA_DIR/scripts/run_patha_server.sh"
  MODE="container (pinned provenance)"
  PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
  RESULTS_ROOT="$PRIMARY_ROOT/results/path-a"
else
  LAUNCHER="$PATHA_DIR/scripts/run_patha_fold.sh"
  MODE="bare python (git commit only; NOT reportable provenance)"
  RESULTS_ROOT="$PATHA_DIR/results"
fi
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
QUEUE_LOG="$PRIMARY_ROOT/logs/path_a_queue.log"
LOCK_FILE="$PRIMARY_ROOT/logs/path_a_queue.lock"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
mkdir -p "$PRIMARY_ROOT/logs" "$PATHA_DIR/results/logs"

# Same ntfy contract as the width queue: silently no-op when no topic is
# configured, never fail the run because a notification failed.
notify() {
  local title="$1" body="$2" priority="${3:-default}"
  [[ -s "$NTFY_TOPIC_FILE" ]] || return 0
  local topic
  topic="$(tr -d '[:space:]' < "$NTFY_TOPIC_FILE")"
  [[ -n "$topic" ]] || return 0
  curl -fsS \
    -H "Title: $title" \
    -H "Priority: $priority" \
    -d "$body" \
    "https://ntfy.sh/${topic}" >/dev/null 2>&1 || true
}

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$QUEUE_LOG"
}

# Control before fair within each fold, so the paired baseline always exists
# first if the queue is interrupted.
CELLS=(
  "0 control" "0 fair"
  "1 control" "1 fair"
  "2 control" "2 fair"
  "3 control" "3 fair"
  "4 control" "4 fair"
)

tag_for_arm() {
  case "$1" in
    control) echo "termwarm" ;;
    fair)    echo "fairwarm" ;;
  esac
}

is_complete() {
  local fold="$1" tag="$2" dir="$RESULTS_ROOT/fold${fold}_${tag}"
  # A cell is complete when its run directory holds a cv_summary.json at any
  # depth. find keeps this portable to bash 3.2, which has no globstar.
  [[ -d "$dir" ]] || return 1
  [[ -n "$(find "$dir" -name cv_summary.json -print -quit 2>/dev/null)" ]]
}

PENDING=()
DONE=()
for cell in "${CELLS[@]}"; do
  read -r fold arm <<< "$cell"
  tag="$(tag_for_arm "$arm")"
  if is_complete "$fold" "$tag"; then
    DONE+=("fold${fold}/${arm}")
  else
    PENDING+=("$cell")
  fi
done

echo "=== PaperB Path A queue ==="
echo "  mode     : $MODE"
echo "  results  : $RESULTS_ROOT"
echo "  complete : ${#DONE[@]}  ${DONE[*]:-(none)}"
echo "  pending  : ${#PENDING[@]}"
for cell in "${PENDING[@]}"; do
  read -r fold arm <<< "$cell"
  echo "             fold${fold} / ${arm}"
done
echo "  estimate : ~1.15 h per cell on RTX A4000 (measured from server q8 cells)"
awk -v n="${#PENDING[@]}" 'BEGIN { printf "  total    : ~%.1f h remaining\n", n * 1.15 }'
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run; nothing launched)"
  exit 0
fi

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "Nothing to do. Analyse with:"
  echo "  python3 PaperB_PathA/scripts/compare_patha.py"
  exit 0
fi

# One queue at a time. A second launch would contend for the single GPU.
if [[ -e "$LOCK_FILE" ]] && kill -0 "$(cat "$LOCK_FILE" 2>/dev/null)" 2>/dev/null; then
  echo "A Path A queue is already running (pid $(cat "$LOCK_FILE"))." >&2
  echo "Remove $LOCK_FILE only if that process is gone." >&2
  exit 1
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

TOTAL="${#PENDING[@]}"
STARTED_AT="$(date -Iseconds)"
log "queue start: $TOTAL cells pending, mode=$MODE"
notify "Path A started" "$TOTAL cells pending on $(hostname -s). Estimated $(awk -v n="$TOTAL" 'BEGIN{printf "%.1f", n*1.15}') h."

INDEX=0
for cell in "${PENDING[@]}"; do
  read -r fold arm <<< "$cell"
  INDEX=$((INDEX + 1))
  CELL_START=$(date +%s)
  log "[$INDEX/$TOTAL] starting fold${fold}/${arm}"

  if ! "$LAUNCHER" "$fold" "$arm"; then
    log "[$INDEX/$TOTAL] FAILED fold${fold}/${arm} — queue stopped"
    notify "Path A FAILED" \
      "fold${fold}/${arm} failed after $((INDEX - 1))/$TOTAL completed. Queue stopped. Log: $QUEUE_LOG" \
      "high"
    exit 1
  fi

  ELAPSED=$(( ($(date +%s) - CELL_START) / 60 ))
  log "[$INDEX/$TOTAL] finished fold${fold}/${arm} in ${ELAPSED} min"
  notify "Path A $INDEX/$TOTAL" "fold${fold}/${arm} done in ${ELAPSED} min."
done

log "queue complete: $TOTAL cells since $STARTED_AT"

# Put the actual verdict in the final notification, not just "done".
VERDICT="$(python3 "$PATHA_DIR/scripts/compare_patha.py" --summary 2>/dev/null || echo "run compare_patha.py for the verdict")"
log "verdict: $VERDICT"
notify "Path A complete" "$VERDICT" "high"

echo
echo "Queue complete. Verdict: $VERDICT"
echo "Full analysis:"
echo "  python3 PaperB_PathA/scripts/compare_patha.py"
