#!/usr/bin/env bash
# Run remaining server-primary width cells one at a time.
#
# Declared order from SERVER_HISTOPATH.md / histopath_vqc_width_server_protocol.md:
#   fold 1: q8, q4, q12
#   fold 2: q12, q8, q4
#   fold 3: q4, q12, q8
#   fold 4: q8, q12, q4
#
# Skips any cell that already has cv_summary.json. Stops on the first failure.
# Does not start a second GPU job: refuse if the GPU is busy, or --wait until idle.
#
# Do not launch this while a cell is already training unless you pass --wait.
set -euo pipefail

USAGE="Usage: $0 [--dry-run] [--wait]
  --dry-run  print completed / pending cells and exit
  --wait     if the GPU is busy, poll until idle then continue
"

DRY_RUN=0
WAIT_GPU=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --wait) WAIT_GPU=1 ;;
    -h|--help)
      printf '%s' "$USAGE"
      exit 0
      ;;
    *)
      printf '%s' "$USAGE" >&2
      exit 2
      ;;
  esac
done

# Interleaved order: calendar/thermal drift is not confounded with width.
CELLS=(
  "1 8"
  "1 4"
  "1 12"
  "2 12"
  "2 8"
  "2 4"
  "3 4"
  "3 12"
  "3 8"
  "4 8"
  "4 12"
  "4 4"
)

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
LAUNCHER="$ROOT/scripts/run_histopath_width_server.sh"
QUEUE_LOG="$PRIMARY_ROOT/logs/server_width_queue.log"
LOCK_FILE="$PRIMARY_ROOT/logs/server_width_queue.lock"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"

cell_dir() {
  local fold="$1" n_qubits="$2"
  printf '%s' "$PRIMARY_ROOT/results/server-width/q${n_qubits}/fold${fold}"
}

cell_done() {
  local fold="$1" n_qubits="$2"
  [[ -e "$(cell_dir "$fold" "$n_qubits")/results/histopath/cv_summary.json" ]]
}

gpu_busy() {
  if docker ps --format '{{.Image}} {{.Names}}' 2>/dev/null | grep -Eq 'mbc-gpu|pytorch'; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
      | grep -Eiq 'python|pt_main_thread'; then
      return 0
    fi
  fi
  return 1
}

notify() {
  local title="$1" body="$2"
  if [[ ! -s "$NTFY_TOPIC_FILE" ]]; then
    return 0
  fi
  local topic
  topic="$(tr -d '[:space:]' < "$NTFY_TOPIC_FILE")"
  [[ -n "$topic" ]] || return 0
  curl -fsS \
    -H "Title: $title" \
    -H "Priority: default" \
    -d "$body" \
    "https://ntfy.sh/${topic}" >/dev/null 2>&1 || true
}

log() {
  mkdir -p "$PRIMARY_ROOT/logs"
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$QUEUE_LOG"
}

if [[ ! -x "$LAUNCHER" ]]; then
  echo "Missing executable launcher: $LAUNCHER" >&2
  exit 1
fi

echo "Server-primary width queue"
echo "Primary root: $PRIMARY_ROOT"
echo

pending=()
for pair in "${CELLS[@]}"; do
  read -r fold n_qubits <<<"$pair"
  if cell_done "$fold" "$n_qubits"; then
    echo "  done     fold $fold  q$n_qubits"
  else
    echo "  pending  fold $fold  q$n_qubits"
    pending+=("$pair")
  fi
done
echo

if [[ ${#pending[@]} -eq 0 ]]; then
  echo "All twelve server cells are complete."
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: ${#pending[@]} cell(s) would run next, one at a time."
  exit 0
fi

if gpu_busy; then
  if [[ "$WAIT_GPU" -eq 1 ]]; then
    log "GPU busy; waiting for the current cell to finish."
    while gpu_busy; do
      sleep 60
    done
    log "GPU idle; starting remaining cells."
  else
    echo "GPU is busy. Do not start a second server cell." >&2
    echo "Leave the current job running, or rerun with --wait after it is safe." >&2
    exit 1
  fi
fi

mkdir -p "$PRIMARY_ROOT/logs"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another width queue is already running ($LOCK_FILE)." >&2
  exit 1
fi

# Recompute pending after a possible wait: the running cell may have finished.
pending=()
for pair in "${CELLS[@]}"; do
  read -r fold n_qubits <<<"$pair"
  if ! cell_done "$fold" "$n_qubits"; then
    pending+=("$pair")
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  log "All twelve server cells are complete."
  exit 0
fi

log "Queue starting: ${#pending[@]} remaining cell(s)."
notify "MBC server queue started" "${#pending[@]} remaining width cells."

for pair in "${pending[@]}"; do
  read -r fold n_qubits <<<"$pair"
  if cell_done "$fold" "$n_qubits"; then
    log "Skipping fold $fold q$n_qubits (already complete)."
    continue
  fi
  if gpu_busy; then
    log "Refusing fold $fold q$n_qubits: GPU became busy."
    notify "MBC server queue stopped" "GPU busy before fold $fold q$n_qubits."
    exit 1
  fi
  log "Starting fold $fold q$n_qubits"
  if ! "$LAUNCHER" "$fold" "$n_qubits"; then
    log "FAILED fold $fold q$n_qubits — queue stopped."
    notify "MBC server queue FAILED" "Stopped at fold $fold q$n_qubits."
    exit 1
  fi
  log "Finished fold $fold q$n_qubits"
  notify "MBC q${n_qubits}/Fold-${fold} completed" "Server width cell finished."
done

log "Queue complete: all pending server width cells finished."
notify "MBC server queue complete" "All remaining primary width cells finished."
