#!/usr/bin/env bash
# Run on SJT516SCOPE052 after repo + venv exist.
set -euo pipefail

REPO="${MBC_REPO:-$HOME/projects/mbc}"
if [[ ! -d "$REPO/.git" && -d "$HOME/mbc/.git" ]]; then
  REPO="$HOME/mbc"
fi
cd "$REPO"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel -q
python -m pip install -q torch scikit-learn pandas scipy pennylane pennylane-lightning

SCRIPT="papers/paper_a/scripts/train_wbcd_gpu_investigation.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing $SCRIPT — scp from laptop or git pull after push." >&2
  exit 1
fi

QUICK="${1:-}"
OUT="${MBC_GPU_OUT:-papers/paper_a/results/gpu_investigation}"
if [[ "$QUICK" == "--quick" ]]; then
  OUT="papers/paper_a/results/gpu_investigation_smoke"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L 2>/dev/null || true

if [[ "$QUICK" == "--quick" ]]; then
  exec python "$SCRIPT" --quick --out-dir "$OUT"
else
  exec python "$SCRIPT" --out-dir "$OUT"
fi