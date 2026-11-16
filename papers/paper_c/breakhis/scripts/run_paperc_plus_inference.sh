#!/usr/bin/env bash
# Paper C+ locked inference: seed-mean OOF → paired case bootstrap → Holm (family=1).
# Magnification is multi-valued within cases, so co-primary bootstrap is whole-case
# (same as Paper C IDC Δ_protocol). Per-mag tables remain ordered secondaries.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PRIMARY_ROOT="${MBC_PRIMARY_ROOT:-$HOME/mbc-primary}"
RES="$PRIMARY_ROOT/results/paper-c-plus"
INF="$RES/inference"
LOG_DIR="$PRIMARY_ROOT/logs"
PYTHON="${PYTHON:-python3}"
REPLICATES="${REPLICATES:-10000}"
mkdir -p "$INF" "$LOG_DIR"

NTFY_TOPIC_FILE="${MBC_NTFY_TOPIC_FILE:-$HOME/.config/mbc/ntfy_topic}"
notify() {
  local title="$1" body="$2" priority="${3:-default}"
  [[ -s "$NTFY_TOPIC_FILE" ]] || return 0
  local topic; topic="$(tr -d '[:space:]' < "$NTFY_TOPIC_FILE")"
  [[ -n "$topic" ]] || return 0
  curl -fsS -H "Title: $title" -H "Priority: $priority" -H "Tags: paper-c-plus" \
    -d "$body" "https://ntfy.sh/${topic}" >/dev/null 2>&1 || true
}

average_arm() {
  local encoder="$1" protocol="$2"
  local out="$RES/${encoder}/fmlp-${protocol}/seed_mean_predictions.csv"
  if [[ -f "$out" ]]; then
    echo "skip average $encoder $protocol"
    return 0
  fi
  "$PYTHON" papers/paper_c/scripts/average_seed_predictions.py \
    --prediction "$RES/${encoder}/fmlp-${protocol}/seed-42/oof_predictions.csv" --seed 42 \
    --prediction "$RES/${encoder}/fmlp-${protocol}/seed-43/oof_predictions.csv" --seed 43 \
    --prediction "$RES/${encoder}/fmlp-${protocol}/seed-44/oof_predictions.csv" --seed 44 \
    --output "$out"
  # Carry magnification from seed-42 for secondary tables (not co-primary strata).
  "$PYTHON" - <<PY
import pandas as pd
from pathlib import Path
mean = Path("$out")
seed = Path("$RES/${encoder}/fmlp-${protocol}/seed-42/oof_predictions.csv")
m = pd.read_csv(mean)
s = pd.read_csv(seed, usecols=["filepath", "magnification"])
out = m.merge(s, on="filepath", how="left", validate="one_to_one")
assert out["magnification"].notna().all()
out.to_csv(mean, index=False)
print(f"augmented {mean} with magnification")
PY
}

notify "C+ inference start" "seed-mean + bootstrap + Holm family=1" high
for enc in dinov2_vitl14 resnet50; do
  for proto in grouped random; do
    average_arm "$enc" "$proto"
  done
done

# Co-primary: DINOv2 F-MLP random − grouped
OUT_PROTO="$INF/delta_protocol_dinov2_fmlp.json"
if [[ ! -f "$OUT_PROTO" ]]; then
  "$PYTHON" papers/paper_c/scripts/paired_case_bootstrap.py \
    --left  "$RES/dinov2_vitl14/fmlp-random/seed_mean_predictions.csv" \
    --right "$RES/dinov2_vitl14/fmlp-grouped/seed_mean_predictions.csv" \
    --left-label random-fmlp --right-label grouped-fmlp \
    --metric case_balanced_auprc \
    --replicates "$REPLICATES" \
    --seed 52026 \
    --output "$OUT_PROTO"
fi

# Ordered secondary: ResNet-50 F-MLP protocol gap
OUT_RN="$INF/delta_protocol_resnet50_fmlp.json"
if [[ ! -f "$OUT_RN" ]]; then
  "$PYTHON" papers/paper_c/scripts/paired_case_bootstrap.py \
    --left  "$RES/resnet50/fmlp-random/seed_mean_predictions.csv" \
    --right "$RES/resnet50/fmlp-grouped/seed_mean_predictions.csv" \
    --left-label random-fmlp --right-label grouped-fmlp \
    --metric case_balanced_auprc \
    --replicates "$REPLICATES" \
    --seed 52027 \
    --output "$OUT_RN"
fi

# Holm family size 1 (FOV omitted)
"$PYTHON" - <<'PY'
import json
from pathlib import Path
inf = Path.home() / "mbc-primary/results/paper-c-plus/inference"
proto = json.loads((inf / "delta_protocol_dinov2_fmlp.json").read_text())
raw = float(proto["two_sided_bootstrap_p"])
holm = min(1.0, 1 * raw)  # family size 1
out = {
    "method": "Holm step-down, family size 1 (FOV not co-primary on BreakHis)",
    "alpha": 0.05,
    "ordered_hypotheses": ["protocol_optimism"],
    "results": {
        "protocol_optimism": {
            "raw_p": raw,
            "holm_adjusted_p": holm,
            "reject_at_0.05": holm <= 0.05,
            "difference": proto["difference"],
            "ci_95_percentile": proto["ci_95_percentile"],
            "left_estimate": proto["left_estimate"],
            "right_estimate": proto["right_estimate"],
        }
    },
    "secondary_resnet50_fmlp_delta": json.loads(
        (inf / "delta_protocol_resnet50_fmlp.json").read_text()
    ),
}
path = inf / "coprimary_holm.json"
if path.exists():
    raise SystemExit(f"exists: {path}")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY

notify "C+ inference COMPLETE" "see $INF/coprimary_holm.json" high
echo "INF=$INF"
