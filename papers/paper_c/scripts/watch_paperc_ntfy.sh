#!/usr/bin/env bash
# Tail Paper C embedding logs and push ntfy events.
# Topic: ~/.config/mbc/ntfy_topic (or $MBC_NTFY_TOPIC_FILE); silent no-op if absent.
# Same contract as Path A / width queue / Paper D.
set -euo pipefail

PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
LOG_DIR="${PRIMARY_ROOT}/logs"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
STATE_DIR="${PRIMARY_ROOT}/provenance/paper-c/ntfy_watch"
mkdir -p "$STATE_DIR" "$LOG_DIR"

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

seen_file="$STATE_DIR/seen_lines.txt"
touch "$seen_file"

already_seen() {
  local key="$1"
  grep -Fxq -- "$key" "$seen_file" 2>/dev/null
}

mark_seen() {
  local key="$1"
  printf '%s\n' "$key" >> "$seen_file"
}

handle_line() {
  local src="$1" line="$2"
  local key="${src}|${line}"
  already_seen "$key" && return 0

  case "$line" in
    START\ *)
      mark_seen "$key"
      notify "Paper C start" "$line" default
      ;;
    DONE\ *)
      mark_seen "$key"
      notify "Paper C done" "$line" default
      ;;
    K1_DONE\ *)
      mark_seen "$key"
      notify "Paper C K=1 complete" "$line" high
      ;;
    PHASE2_COMPLETE\ *)
      mark_seen "$key"
      notify "Paper C embeddings complete" "$line" high
      ;;
    CACHE\ VERIFY\ OK)
      mark_seen "$key"
      notify "Paper C cache OK" "from ${src}: ${line}" default
      ;;
    CACHE\ VERIFY\ FAILED*|LOCK\ VERIFY\ FAILED*)
      mark_seen "$key"
      notify "Paper C FAILED" "from ${src}: ${line}" urgent
      ;;
    *"shard "*" complete "*)
      # Notify every shard (14 for full IDC K=1).
      mark_seen "$key"
      notify "Paper C shard" "$line" low
      ;;
    FAIL\ *|*"FAIL "*)
      mark_seen "$key"
      notify "Paper C FAIL" "from ${src}: ${line}" urgent
      ;;
    *"Traceback (most recent call last)"*|*"SyntaxError:"*|*"Error:"*|*"RuntimeError"*|*"Permission denied"*)
      mark_seen "$key"
      notify "Paper C error" "from ${src}: ${line}" urgent
      ;;
  esac
}

scan_file() {
  local path="$1"
  [[ -f "$path" ]] || return 0
  local base
  base="$(basename "$path")"
  # Read whole file; seen_file dedupes across polls.
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] || continue
    handle_line "$base" "$line"
  done < "$path"
}

echo "Paper C ntfy watcher"
echo "  topic file : $([ -s "$NTFY_TOPIC_FILE" ] && echo on || echo off)"
echo "  log dir    : $LOG_DIR"
echo "  state      : $STATE_DIR"
notify "Paper C watcher on" "Monitoring embedding logs under ${LOG_DIR}" low

while true; do
  # Prefer active chain/phase logs; also catch per-encoder tees.
  shopt -s nullglob
  for path in \
    "$LOG_DIR"/paperc_full_k1_chain_*.log \
    "$LOG_DIR"/paperc_phase2_fov_bcss_*.log \
    "$LOG_DIR"/paperc_idc_protocol_queue*.log \
    "$LOG_DIR"/paperc_idc_fov_queue*.log \
    "$LOG_DIR"/paperc_bcss_probe_queue*.log \
    "$LOG_DIR"/paperc_probe_*.log \
    "$LOG_DIR"/paperc_bcss_probe_*.log \
    "$LOG_DIR"/paperc_embed_*.log \
    "$LOG_DIR"/paperc_bcss_*.log
  do
    scan_file "$path"
  done

  shopt -u nullglob
  sleep 20
done
