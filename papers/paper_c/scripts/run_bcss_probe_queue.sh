#!/usr/bin/env bash
# Paper C BCSS FOV probe queue (k1 vs k9; confirmatory encoders).
# ntfy: ~/.config/mbc/ntfy_topic (or $MBC_NTFY_TOPIC_FILE); silent no-op if absent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
MANIFEST="${MBC_BCSS_MANIFEST:-$PRIMARY_ROOT/provenance/paper-c/bcss_centres.csv}"
OUT_ROOT="${MBC_PAPERC_BCSS_RESULTS:-$PRIMARY_ROOT/results/paper-c/bcss}"
LOG_DIR="$PRIMARY_ROOT/logs"
QUEUE_LOG="${MBC_PAPERC_BCSS_LOG:-$LOG_DIR/paperc_bcss_probe_queue.log}"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
PYTHON="${PYTHON:-python3}"

ENCODERS="${ENCODERS:-dinov2_vitl14 resnet50}"
CONTEXTS="${CONTEXTS:-k1 k9}"
SEEDS="${SEEDS:-42 43 44}"
EXPECT_N="${EXPECT_N_BCSS:-68784}"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

notify() {
  local title="$1" body="$2" priority="${3:-default}"
  [[ -s "$NTFY_TOPIC_FILE" ]] || return 0
  local topic
  topic="$(tr -d '[:space:]' < "$NTFY_TOPIC_FILE")"
  [[ -n "$topic" ]] || return 0
  curl -fsS \
    -H "Title: $title" \
    -H "Priority: $priority" \
    -H "Tags: paper-c" \
    -d "$body" \
    "https://ntfy.sh/${topic}" >/dev/null 2>&1 || true
}

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$QUEUE_LOG"
}

fail_notify_exit() {
  local rc="${1:-1}"
  local detail="${2:-queue error}"
  trap - ERR
  log "FAIL $detail rc=$rc"
  notify "Paper C queue FAIL" "$detail (rc=$rc)" urgent
  exit "$rc"
}

trap 'rc=$?; fail_notify_exit "$rc" "line=${LINENO} cmd=${BASH_COMMAND}"' ERR

read_auprc() {
  local summary_path="$1"
  "$PYTHON" - "$summary_path" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"{summary['test_metrics']['case_balanced_auprc']:.4f}")
PY
}

require() {
  [[ -e "$1" ]] || fail_notify_exit 1 "missing required path: $1"
}

require "$MANIFEST"
require "papers/paper_c/scripts/run_bcss_probe.py"
require "papers/paper_c/scripts/average_seed_predictions.py"
"$PYTHON" papers/paper_c/scripts/verify_paperc_lock.py

jobs=()
for encoder in $ENCODERS; do
  for context in $CONTEXTS; do
    cache="$PRIMARY_ROOT/embeddings/bcss_${encoder}_${context}"
    require "$cache/provenance.json"
    "$PYTHON" papers/paper_c/scripts/verify_embedding_cache.py "$cache" --expect-n "$EXPECT_N"
    for seed in $SEEDS; do
      jobs+=("$encoder|$context|$seed")
    done
  done
done

log "=== Paper C BCSS probe queue ==="
log "manifest=$MANIFEST"
log "out=$OUT_ROOT"
log "expect_n=$EXPECT_N"
log "jobs=${#jobs[@]}"
log "ntfy=$([ -s "$NTFY_TOPIC_FILE" ] && echo on || echo off)"
notify "Paper C BCSS probes start" "BCSS k1/k9 queue: ${#jobs[@]} cells" default

for spec in "${jobs[@]}"; do
  IFS='|' read -r encoder context seed <<<"$spec"
  cache="$PRIMARY_ROOT/embeddings/bcss_${encoder}_${context}"
  out="$OUT_ROOT/${encoder}/${context}/seed-${seed}"
  summary="$out/summary.json"
  if [[ -f "$summary" ]]; then
    log "SKIP bcss $encoder $context seed=$seed (summary exists)"
    continue
  fi
  log "START bcss $encoder $context seed=$seed"
  notify "Paper C probe start" "bcss $encoder $context seed=$seed" low
  mkdir -p "$out"
  cell_log="$LOG_DIR/paperc_bcss_probe_${encoder}_${context}_s${seed}_$(date -u +%Y%m%dT%H%M%SZ).log"
  set +e
  "$PYTHON" papers/paper_c/scripts/run_bcss_probe.py \
    --cache "$cache" \
    --centre-manifest "$MANIFEST" \
    --seed "$seed" \
    --output-dir "$out" \
    >"$cell_log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 || ! -f "$summary" ]]; then
    trap - ERR
    fail_notify_exit "$rc" "bcss $encoder $context seed=$seed probe rc=$rc log=$cell_log"
  fi
  auprc="$(read_auprc "$summary")"
  log "DONE bcss $encoder $context seed=$seed auprc=$auprc"
  notify "Paper C probe done" "bcss $encoder $context seed=$seed auprc=$auprc" default
done

log "Averaging BCSS seed predictions..."
for encoder in $ENCODERS; do
  for context in $CONTEXTS; do
    out_mean="$OUT_ROOT/${encoder}/${context}/seed_mean_predictions.csv"
    if [[ -f "$out_mean" ]]; then
      log "SKIP average bcss $encoder $context"
      continue
    fi
    "$PYTHON" papers/paper_c/scripts/average_seed_predictions.py \
      --prediction "$OUT_ROOT/${encoder}/${context}/seed-42/test_predictions.csv" --seed 42 \
      --prediction "$OUT_ROOT/${encoder}/${context}/seed-43/test_predictions.csv" --seed 43 \
      --prediction "$OUT_ROOT/${encoder}/${context}/seed-44/test_predictions.csv" --seed 44 \
      --output "$out_mean"
    log "AVERAGED bcss $encoder $context -> $out_mean"
  done
done

trap - ERR
log "BCSS_QUEUE_COMPLETE"
notify "Paper C BCSS complete" "BCSS k1/k9 × 2 encoders × 3 seeds" high
