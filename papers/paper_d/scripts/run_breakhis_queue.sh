#!/usr/bin/env bash
# Queue Paper D BreakHis cells: prepare manifests, then F/T × MLP/VQC.
# ntfy: same contract as Path A / width queue — topic from
# ~/.config/mbc/ntfy_topic (or $MBC_NTFY_TOPIC_FILE); silent no-op if absent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
if ! "$PYTHON" -c "import sys" 2>/dev/null; then
  echo "No usable Python: tried .venv/bin/python and python3" >&2
  exit 1
fi

FOLDS="${FOLDS:-0 1 2 3 4}"
ARMS="${ARMS:-F-MLP F-VQC T-MLP T-VQC}"
QUICK=0
DRY=0

PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
QUEUE_LOG="${MBC_PAPERD_QUEUE_LOG:-$PRIMARY_ROOT/logs/paper_d_queue.log}"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
mkdir -p "$(dirname "$QUEUE_LOG")" "$ROOT/results/breakhis_paper_d"

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

usage() {
  echo "Usage: $0 [--quick] [--dry-run] [--folds \"0 1\"] [--arms \"F-MLP F-VQC\"]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

EXTRACTED="$ROOT/papers/paper_d/data/BreaKHis_v1"
if [[ ! -d "$EXTRACTED/histology_slides" ]]; then
  echo "Extract BreakHis first:" >&2
  echo "  tar -xzf papers/paper_d/data/BreaKHis_v1.tar.gz -C papers/paper_d/data" >&2
  exit 1
fi

# Refuse labelled runs against a broken lock.
"$PYTHON" papers/paper_d/scripts/verify_paperd_lock.py

PENDING=()
DONE=()
for fold in $FOLDS; do
  for arm in $ARMS; do
    out="$ROOT/results/breakhis_paper_d/fold${fold}_${arm}/summary.json"
    if [[ -f "$out" && "$QUICK" -eq 0 ]]; then
      DONE+=("fold${fold}/${arm}")
    else
      PENDING+=("${fold} ${arm}")
    fi
  done
done

echo "=== Paper D BreakHis queue ==="
echo "  complete : ${#DONE[@]}  ${DONE[*]:-(none)}"
echo "  pending  : ${#PENDING[@]}"
for cell in "${PENDING[@]:-}"; do
  echo "             $cell"
done
echo "  ntfy     : $([ -s "$NTFY_TOPIC_FILE" ] && echo "on ($NTFY_TOPIC_FILE)" || echo "off (no topic file)")"
echo "  log      : $QUEUE_LOG"
echo

if [[ "$DRY" -eq 1 ]]; then
  for fold in $FOLDS; do
    echo "$PYTHON papers/paper_d/scripts/prepare_breakhis_manifests.py --fold $fold"
  done
  for cell in "${PENDING[@]:-}"; do
    read -r fold arm <<< "$cell"
    echo "$PYTHON papers/paper_d/scripts/train_breakhis_cv.py --fold $fold --arm $arm${QUICK:+ --quick}"
  done
  echo "(dry run; nothing launched)"
  exit 0
fi

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "Nothing to do — all cells have summary.json"
  notify "Paper D complete" "All cells already finished on $(hostname -s)." "default"
  exit 0
fi

TOTAL="${#PENDING[@]}"
STARTED_AT="$(date -Iseconds)"
HOST="$(hostname -s 2>/dev/null || hostname)"
log "queue start: $TOTAL cells pending"
notify "Paper D started" \
  "$TOTAL cells pending on ${HOST}. Folds=[$FOLDS] arms=[$ARMS]."

# Prepare manifests for every fold we will touch.
PREPARED=""
for cell in "${PENDING[@]}"; do
  read -r fold arm <<< "$cell"
  case " $PREPARED " in
    *" $fold "*) ;;
    *)
      log "prepare fold $fold"
      "$PYTHON" papers/paper_d/scripts/prepare_breakhis_manifests.py --fold "$fold"
      PREPARED="$PREPARED $fold"
      ;;
  esac
done

INDEX=0
for cell in "${PENDING[@]}"; do
  read -r fold arm <<< "$cell"
  INDEX=$((INDEX + 1))
  CELL_START=$(date +%s)
  log "[$INDEX/$TOTAL] starting fold${fold}/${arm}"
  cmd=("$PYTHON" papers/paper_d/scripts/train_breakhis_cv.py --fold "$fold" --arm "$arm")
  if [[ "$QUICK" -eq 1 ]]; then
    cmd+=(--quick)
  fi
  if ! "${cmd[@]}"; then
    log "[$INDEX/$TOTAL] FAILED fold${fold}/${arm} — queue stopped"
    notify "Paper D FAILED" \
      "fold${fold}/${arm} failed after $((INDEX - 1))/$TOTAL completed on ${HOST}. Log: $QUEUE_LOG" \
      "high"
    exit 1
  fi
  ELAPSED=$(( ($(date +%s) - CELL_START) / 60 ))
  log "[$INDEX/$TOTAL] finished fold${fold}/${arm} in ${ELAPSED} min"
  notify "Paper D $INDEX/$TOTAL" "fold${fold}/${arm} done in ${ELAPSED} min on ${HOST}."
done

log "queue complete: $TOTAL cells since $STARTED_AT"
notify "Paper D complete" \
  "$TOTAL cells finished on ${HOST} since $STARTED_AT. See results/breakhis_paper_d/" \
  "high"

echo "queue finished"
