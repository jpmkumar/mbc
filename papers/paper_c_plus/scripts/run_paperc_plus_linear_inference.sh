#!/usr/bin/env bash
# Seed-mean + paired bootstrap for Paper C+ linear-SGD bridge secondaries.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
RES="$PRIMARY_ROOT/results/paper-c-plus"
INF="$RES/inference"
PYTHON="${PYTHON:-python3}"
REPLICATES="${REPLICATES:-10000}"
mkdir -p "$INF"

average_arm() {
  local encoder="$1" protocol="$2"
  local out="$RES/${encoder}/linear-${protocol}/seed_mean_predictions.csv"
  if [[ -f "$out" ]]; then
    echo "skip average $encoder linear-$protocol"
    return 0
  fi
  "$PYTHON" papers/paper_c/scripts/average_seed_predictions.py \
    --prediction "$RES/${encoder}/linear-${protocol}/seed-42/oof_predictions.csv" --seed 42 \
    --prediction "$RES/${encoder}/linear-${protocol}/seed-43/oof_predictions.csv" --seed 43 \
    --prediction "$RES/${encoder}/linear-${protocol}/seed-44/oof_predictions.csv" --seed 44 \
    --output "$out"
  "$PYTHON" - <<PY
import pandas as pd
from pathlib import Path
mean = Path("$out")
seed = Path("$RES/${encoder}/linear-${protocol}/seed-42/oof_predictions.csv")
m = pd.read_csv(mean)
s = pd.read_csv(seed, usecols=["filepath", "magnification"])
out = m.merge(s, on="filepath", how="left", validate="one_to_one")
assert out["magnification"].notna().all()
out.to_csv(mean, index=False)
print(f"augmented {mean}")
PY
}

for enc in dinov2_vitl14 resnet50; do
  for proto in grouped random; do
    average_arm "$enc" "$proto"
  done
done

boot() {
  local name="$1" left="$2" right="$3" seed="$4"
  local out="$INF/$name.json"
  if [[ -f "$out" ]]; then
    echo "skip $name"
    return 0
  fi
  "$PYTHON" papers/paper_c/scripts/paired_case_bootstrap.py \
    --left "$left" --right "$right" \
    --left-label random-linear --right-label grouped-linear \
    --metric case_balanced_auprc \
    --replicates "$REPLICATES" --seed "$seed" \
    --output "$out"
}

boot delta_protocol_dinov2_linear \
  "$RES/dinov2_vitl14/linear-random/seed_mean_predictions.csv" \
  "$RES/dinov2_vitl14/linear-grouped/seed_mean_predictions.csv" \
  52028

boot delta_protocol_resnet50_linear \
  "$RES/resnet50/linear-random/seed_mean_predictions.csv" \
  "$RES/resnet50/linear-grouped/seed_mean_predictions.csv" \
  52029

"$PYTHON" papers/paper_c_plus/scripts/summarize_breakhis_per_mag.py \
  --encoder dinov2_vitl14 --head linear \
  --output "$INF/per_mag_dinov2_linear.json" || true

echo "linear inference written under $INF"
