#!/usr/bin/env bash
# Paper C IDC K=1 protocol probe queue (co-primary Δ_protocol + bundled secondary).
# ntfy: ~/.config/mbc/ntfy_topic (or $MBC_NTFY_TOPIC_FILE); silent no-op if absent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
MANIFEST="${MBC_IDC_MANIFEST:-$PRIMARY_ROOT/provenance/paper-c/idc_patch_manifest.csv}"
OUT_ROOT="${MBC_PAPERC_IDC_RESULTS:-$PRIMARY_ROOT/results/paper-c/idc}"
LOG_DIR="$PRIMARY_ROOT/logs"
QUEUE_LOG="${MBC_PAPERC_PROTOCOL_LOG:-$LOG_DIR/paperc_idc_protocol_queue.log}"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
PYTHON="${PYTHON:-python3}"

ENCODERS="${ENCODERS:-dinov2_vitl14 resnet50}"
SEEDS="${SEEDS:-42 43 44}"

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

# Catch set -e failures (parse bugs, missing files, average step, etc.).
trap 'rc=$?; fail_notify_exit "$rc" "line=${LINENO} cmd=${BASH_COMMAND}"' ERR

read_auprc() {
  local summary_path="$1"
  "$PYTHON" - "$summary_path" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
print(f"{summary['metrics']['case_balanced_auprc']:.4f}")
PY
}

require() {
  [[ -e "$1" ]] || fail_notify_exit 1 "missing required path: $1"
}

require "$MANIFEST"
require "data/splits/paper_c/idc/inner_split_summary.json"
require "papers/paper_c/scripts/run_idc_probe_cv.py"
require "papers/paper_c/scripts/average_seed_predictions.py"
"$PYTHON" papers/paper_c/scripts/verify_paperc_lock.py

jobs=()
for encoder in $ENCODERS; do
  cache="$PRIMARY_ROOT/embeddings/${encoder}_upsample224"
  require "$cache/provenance.json"
  "$PYTHON" papers/paper_c/scripts/verify_embedding_cache.py "$cache" --expect-n 277524
  for seed in $SEEDS; do
    jobs+=("$encoder|random|case-balanced|random-k1|$seed")
    jobs+=("$encoder|grouped|case-balanced|grouped-k1|$seed")
    jobs+=("$encoder|random|protocol-native|random-k1-bundled|$seed")
  done
done

log "=== Paper C IDC protocol queue ==="
log "manifest=$MANIFEST"
log "out=$OUT_ROOT"
log "jobs=${#jobs[@]}"
log "ntfy=$([ -s "$NTFY_TOPIC_FILE" ] && echo on || echo off)"
notify "Paper C probes start" "IDC protocol queue: ${#jobs[@]} cells (resume-safe)" default

for spec in "${jobs[@]}"; do
  IFS='|' read -r encoder protocol weighting condition seed <<<"$spec"
  cache="$PRIMARY_ROOT/embeddings/${encoder}_upsample224"
  out="$OUT_ROOT/${encoder}/${condition}/seed-${seed}"
  summary="$out/summary.json"
  if [[ -f "$summary" ]]; then
    log "SKIP $encoder $condition seed=$seed (summary exists)"
    continue
  fi
  log "START $encoder $condition seed=$seed weighting=$weighting"
  notify "Paper C probe start" "$encoder $condition seed=$seed" low
  mkdir -p "$out"
  cell_log="$LOG_DIR/paperc_probe_${encoder}_${condition}_s${seed}_$(date -u +%Y%m%dT%H%M%SZ).log"
  set +e
  "$PYTHON" papers/paper_c/scripts/run_idc_probe_cv.py \
    --cache "$cache" \
    --patch-manifest "$MANIFEST" \
    --protocol "$protocol" \
    --weighting "$weighting" \
    --seed "$seed" \
    --output-dir "$out" \
    >"$cell_log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 || ! -f "$summary" ]]; then
    # Disable ERR trap so we control the message, then exit.
    trap - ERR
    fail_notify_exit "$rc" "$encoder $condition seed=$seed probe rc=$rc log=$cell_log"
  fi
  auprc="$(read_auprc "$summary")"
  log "DONE $encoder $condition seed=$seed auprc=$auprc"
  notify "Paper C probe done" "$encoder $condition seed=$seed auprc=$auprc" default
done

log "Averaging seed predictions..."
for encoder in $ENCODERS; do
  for condition in random-k1 grouped-k1 random-k1-bundled; do
    out_mean="$OUT_ROOT/${encoder}/${condition}/seed_mean_predictions.csv"
    if [[ -f "$out_mean" ]]; then
      log "SKIP average $encoder $condition"
      continue
    fi
    "$PYTHON" papers/paper_c/scripts/average_seed_predictions.py \
      --prediction "$OUT_ROOT/${encoder}/${condition}/seed-42/oof_predictions.csv" --seed 42 \
      --prediction "$OUT_ROOT/${encoder}/${condition}/seed-43/oof_predictions.csv" --seed 43 \
      --prediction "$OUT_ROOT/${encoder}/${condition}/seed-44/oof_predictions.csv" --seed 44 \
      --output "$out_mean"
    log "AVERAGED $encoder $condition -> $out_mean"
  done
done

trap - ERR
log "PROTOCOL_QUEUE_COMPLETE"
notify "Paper C protocol complete" "IDC K=1 random/grouped/bundled × 2 encoders × 3 seeds" high
