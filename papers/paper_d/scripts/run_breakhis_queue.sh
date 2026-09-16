#!/usr/bin/env bash
# Queue Paper D BreakHis cells: prepare manifests, then F/T × MLP/VQC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

FOLDS="${FOLDS:-0 1 2 3 4}"
ARMS="${ARMS:-F-MLP F-VQC T-MLP T-VQC}"
QUICK=0
DRY=0

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

for fold in $FOLDS; do
  echo "=== prepare fold $fold ==="
  if [[ "$DRY" -eq 1 ]]; then
    echo "$PYTHON papers/paper_d/scripts/prepare_breakhis_manifests.py --fold $fold"
  else
    "$PYTHON" papers/paper_d/scripts/prepare_breakhis_manifests.py --fold "$fold"
  fi
  for arm in $ARMS; do
    out="results/breakhis_paper_d/fold${fold}_${arm}/summary.json"
    if [[ -f "$out" && "$QUICK" -eq 0 ]]; then
      echo "skip complete cell $out"
      continue
    fi
    echo "=== train fold=$fold arm=$arm ==="
    cmd=("$PYTHON" papers/paper_d/scripts/train_breakhis_cv.py --fold "$fold" --arm "$arm")
    if [[ "$QUICK" -eq 1 ]]; then
      cmd+=(--quick)
    fi
    if [[ "$DRY" -eq 1 ]]; then
      echo "${cmd[*]}"
    else
      "${cmd[@]}"
    fi
  done
done

echo "queue finished"
