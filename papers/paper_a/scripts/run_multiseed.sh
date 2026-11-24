#!/usr/bin/env bash
# Multi-seed revision runs for Paper A (reviewer: seed variance, boost fold rows).
#
# Folds stay fixed (CV seed 42); only the initialisation seed changes.
# Init seed 42 reproduces the deposited confirmatory / boost numbers.
#
# ntfy: ~/.config/mbc/ntfy_topic (or $MBC_NTFY_TOPIC_FILE); silent no-op if absent.
#
# Usage (from the mbc repo root, venv active):
#   bash papers/paper_a/scripts/run_multiseed.sh            # 5 seeds
#   SEEDS="42 1 2" bash papers/paper_a/scripts/run_multiseed.sh
#   QUICK=1 bash papers/paper_a/scripts/run_multiseed.sh    # smoke test
set -euo pipefail
PY=${PY:-python}
SEEDS=${SEEDS:-"42 101 202 303 404"}
OUT=${OUT:-papers/paper_a/results/multiseed}
Q=""; [[ "${QUICK:-0}" == "1" ]] && Q="--quick"
NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
mkdir -p "$OUT"

ntfy_send() {
  local title="$1" body="$2"
  [[ -s "$NTFY_TOPIC_FILE" ]] || return 0
  local topic
  topic="$(tr -d '[:space:]' <"$NTFY_TOPIC_FILE")"
  [[ -n "$topic" ]] || return 0
  curl -sS -H "Title: $title" -d "$body" "https://ntfy.sh/${topic}" >/dev/null 2>&1 || true
}

ntfy_send "Paper A multiseed" "started seeds=[$SEEDS] out=$OUT quick=${QUICK:-0}"
echo "ntfy=$([ -s "$NTFY_TOPIC_FILE" ] && echo on || echo off)"

status=0
for s in $SEEDS; do
  echo "=== init seed $s : confirmatory (MSE/30) ==="
  if ! $PY papers/paper_a/scripts/train_wbcd_cv.py --seed 42 --init-seed "$s" $Q \
      --out-dir "$OUT/confirmatory/seed_$s"; then
    status=1
    ntfy_send "Paper A multiseed FAILED" "confirmatory seed=$s"
    break
  fi
  echo "=== init seed $s : VQC CE/80 ==="
  if ! $PY papers/paper_a/scripts/train_wbcd_cv.py --seed 42 --init-seed "$s" $Q \
      --vqc-loss ce --vqc-epochs 80 --no-qsvm --out-dir "$OUT/vqc_ce80/seed_$s"; then
    status=1
    ntfy_send "Paper A multiseed FAILED" "vqc_ce80 seed=$s"
    break
  fi
  echo "=== init seed $s : readout / boost arms ==="
  if ! $PY papers/paper_a/scripts/train_wbcd_vqc_boost.py --seed 42 --init-seed "$s" $Q \
      --out-dir "$OUT/vqc_boost/seed_$s"; then
    status=1
    ntfy_send "Paper A multiseed FAILED" "vqc_boost seed=$s"
    break
  fi
  ntfy_send "Paper A multiseed" "finished seed=$s"
done

if [[ "$status" -eq 0 ]]; then
  $PY papers/paper_a/scripts/analysis_multiseed.py --root "$OUT"
  ntfy_send "Paper A multiseed complete" "summary=$OUT/multiseed_summary.md"
else
  exit "$status"
fi
