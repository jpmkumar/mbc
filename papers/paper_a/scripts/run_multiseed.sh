#!/usr/bin/env bash
# Multi-seed revision runs for Paper A (reviewer: seed variance, boost fold rows).
#
# Folds stay fixed (CV seed 42); only the initialisation seed changes.
# Init seed 42 reproduces the deposited confirmatory / boost numbers.
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
mkdir -p "$OUT"

for s in $SEEDS; do
  echo "=== init seed $s : confirmatory (MSE/30) ==="
  $PY papers/paper_a/scripts/train_wbcd_cv.py --seed 42 --init-seed "$s" $Q \
      --out-dir "$OUT/confirmatory/seed_$s"
  echo "=== init seed $s : VQC CE/80 ==="
  $PY papers/paper_a/scripts/train_wbcd_cv.py --seed 42 --init-seed "$s" $Q \
      --vqc-loss ce --vqc-epochs 80 --no-qsvm --out-dir "$OUT/vqc_ce80/seed_$s"
  echo "=== init seed $s : readout / boost arms ==="
  $PY papers/paper_a/scripts/train_wbcd_vqc_boost.py --seed 42 --init-seed "$s" $Q \
      --out-dir "$OUT/vqc_boost/seed_$s"
done

$PY papers/paper_a/scripts/analysis_multiseed.py --root "$OUT"
