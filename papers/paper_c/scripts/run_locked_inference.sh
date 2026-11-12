#!/usr/bin/env bash
# Paper C locked inference: co-primary bootstraps + Holm + ordered secondaries.
# ntfy: ~/.config/mbc/ntfy_topic (or $MBC_NTFY_TOPIC_FILE); silent no-op if absent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
IDC_ROOT="${MBC_PAPERC_IDC_RESULTS:-$PRIMARY_ROOT/results/paper-c/idc}"
BCSS_ROOT="${MBC_PAPERC_BCSS_RESULTS:-$PRIMARY_ROOT/results/paper-c/bcss}"
INF="${MBC_PAPERC_INFERENCE:-$PRIMARY_ROOT/results/paper-c/inference}"
LOG_DIR="$PRIMARY_ROOT/logs"
QUEUE_LOG="${MBC_PAPERC_INFERENCE_LOG:-$LOG_DIR/paperc_locked_inference.log}"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
PYTHON="${PYTHON:-python3}"
REPLICATES="${REPLICATES:-10000}"

mkdir -p "$LOG_DIR" "$INF"

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

require() {
  [[ -e "$1" ]] || fail_notify_exit 1 "missing required path: $1"
}

require "$IDC_ROOT/dinov2_vitl14/random-k1/seed_mean_predictions.csv"
require "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv"
require "$IDC_ROOT/dinov2_vitl14/grouped-mosaic9-complete-k9/seed_mean_predictions.csv"
require "$IDC_ROOT/dinov2_vitl14/grouped-upsample224-complete-k9/seed_mean_predictions.csv"
require "$IDC_ROOT/dinov2_vitl14/random-k1-bundled/seed_mean_predictions.csv"
require "$BCSS_ROOT/dinov2_vitl14/k1/seed_mean_predictions.csv"
require "$BCSS_ROOT/dinov2_vitl14/k9/seed_mean_predictions.csv"
require "papers/paper_c/scripts/paired_case_bootstrap.py"
require "papers/paper_c/scripts/holm_coprimary.py"
require "papers/paper_c/scripts/secondary_reliability.py"
require "papers/paper_c/scripts/model_rank_stability.py"

log "=== Paper C locked inference ==="
log "inf=$INF replicates=$REPLICATES"
notify "Paper C inference start" "co-primary bootstraps + Holm + secondaries" default

run_boot() {
  local name="$1"; shift
  local out="$INF/$name.json"
  if [[ -f "$out" ]]; then
    log "SKIP $name (exists)"
    return 0
  fi
  log "START $name"
  notify "Paper C inference" "START $name" low
  "$PYTHON" papers/paper_c/scripts/paired_case_bootstrap.py "$@" --replicates "$REPLICATES" --output "$out"
  log "DONE $name -> $out"
  notify "Paper C inference" "DONE $name" default
}

# Co-primary Δ_protocol (DINOv2 K=1 random vs grouped, case-balanced).
run_boot delta_protocol \
  --left  "$IDC_ROOT/dinov2_vitl14/random-k1/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv" \
  --left-label random-k1 --right-label grouped-k1

# Co-primary Δ_context (DINOv2 mosaic9 vs upsample224 on complete-K9 pop).
run_boot delta_context \
  --left  "$IDC_ROOT/dinov2_vitl14/grouped-mosaic9-complete-k9/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/dinov2_vitl14/grouped-upsample224-complete-k9/seed_mean_predictions.csv" \
  --left-label grouped-k9 --right-label grouped-k1

if [[ ! -f "$INF/coprimary_holm.json" ]]; then
  log "START holm_coprimary"
  "$PYTHON" papers/paper_c/scripts/holm_coprimary.py \
    --protocol-report "$INF/delta_protocol.json" \
    --context-report "$INF/delta_context.json" \
    --output "$INF/coprimary_holm.json"
  log "DONE holm_coprimary"
  notify "Paper C Holm done" "coprimary_holm.json written" high
else
  log "SKIP holm_coprimary"
fi

# Ordered secondary: bundled random regime vs grouped.
run_boot delta_protocol_bundled \
  --left  "$IDC_ROOT/dinov2_vitl14/random-k1-bundled/seed_mean_predictions.csv" \
  --right "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv" \
  --left-label random-k1-bundled --right-label grouped-k1

# BCSS FOV replication (site-stratified).
run_boot bcss_delta_context \
  --left  "$BCSS_ROOT/dinov2_vitl14/k9/seed_mean_predictions.csv" \
  --right "$BCSS_ROOT/dinov2_vitl14/k1/seed_mean_predictions.csv" \
  --left-label bcss-k9 --right-label bcss-k1 \
  --strata-column site

# ResNet BCSS FOV (ordered secondary replication).
run_boot bcss_delta_context_resnet50 \
  --left  "$BCSS_ROOT/resnet50/k9/seed_mean_predictions.csv" \
  --right "$BCSS_ROOT/resnet50/k1/seed_mean_predictions.csv" \
  --left-label bcss-k9 --right-label bcss-k1 \
  --strata-column site

# Reliability on grouped K=1 DINOv2.
if [[ ! -f "$INF/reliability/dinov2_vitl14-grouped-k1/reliability_summary.json" ]]; then
  log "START secondary_reliability"
  "$PYTHON" papers/paper_c/scripts/secondary_reliability.py \
    --predictions "$IDC_ROOT/dinov2_vitl14/grouped-k1/seed_mean_predictions.csv" \
    --output-dir "$INF/reliability/dinov2_vitl14-grouped-k1"
  log "DONE secondary_reliability"
else
  log "SKIP secondary_reliability"
fi

# Rank stability random vs grouped (seed-42 summaries as runbook).
if [[ ! -f "$INF/rank_stability.json" ]]; then
  log "START model_rank_stability"
  RANK_ARGS=()
  for encoder in dinov2_vitl14 resnet50; do
    RANK_ARGS+=(--pair "$encoder"
      "$IDC_ROOT/${encoder}/random-k1/seed-42/summary.json"
      "$IDC_ROOT/${encoder}/grouped-k1/seed-42/summary.json")
  done
  "$PYTHON" papers/paper_c/scripts/model_rank_stability.py \
    "${RANK_ARGS[@]}" \
    --left-label random-k1 --right-label grouped-k1 \
    --output "$INF/rank_stability.json"
  log "DONE model_rank_stability"
else
  log "SKIP model_rank_stability"
fi

trap - ERR
log "INFERENCE_COMPLETE"
notify "Paper C inference complete" "co-primary + Holm + BCSS + secondaries under $INF" high
