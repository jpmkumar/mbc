# Paper A — GPU server investigation (why quantum loses)

Campus runs use the same **seed-42, 5-fold splits** as `train_wbcd_cv.py`. Malignant
= label 0. Results gitignored locally unless you copy JSON back.

## Install on the server (once)

Typical VIT scope nodes: **no `python` alias**, **PEP 668** blocks
`pip install` on the system Python — you must create `.venv` first.

```bash
# Correct repo (jpmkumar, not jpnkumar)
git clone https://github.com/jpmkumar/mbc.git
cd mbc

# Python 3.12+ on scope — always python3, not python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# CPU-only fallback (works everywhere):
pip install torch scikit-learn pandas scipy pennylane pennylane-lightning

# If nvidia-smi works and you want PennyLane GPU backend:
# pip install pennylane-lightning-gpu
# python -c "import pennylane as qml; qml.device('lightning.gpu', wires=4)"

python -c "import torch; print('cuda', torch.cuda.is_available())"
```

**Copy the investigation script from your laptop** if the server clone predates it
(the file is not on GitHub until you push):

```bash
# On laptop:
scp papers/paper_a/scripts/train_wbcd_gpu_investigation.py \
  papers/paper_a/GPU_INVESTIGATION.md \
  kumar_24phd0656@<server>:~/mbc/papers/paper_a/scripts/
```

PennyLane uses `lightning.qubit` on CPU or `lightning.gpu` when CUDA is available
(the script picks automatically).

## Run the investigation battery

```bash
cd ~/mbc
source .venv/bin/activate

# Quick smoke (~5–15 min CPU)
python papers/paper_a/scripts/train_wbcd_gpu_investigation.py --quick \
  --out-dir papers/paper_a/results/gpu_investigation_smoke

# Full battery (~1–3 h on GPU; several hours CPU)
python papers/paper_a/scripts/train_wbcd_gpu_investigation.py \
  --out-dir papers/paper_a/results/gpu_investigation
```

Copy back to laptop:

- `gpu_investigation/summary.json`
- `gpu_investigation/*.csv`

## What each arm tests (link to “quantum advantage”)

| Arm | Classical–quantum question |
|-----|----------------------------|
| **Sample efficiency** | With less train data, does a VQC catch up to MLP/SVM? (data advantage) |
| **Random labels** | If both fit shuffled labels, expressivity is enough — loss is the issue |
| **No-entangle VQC** | Product-state unitaries only — is entanglement helping or hurting? |
| **LR grid (equal budget)** | After matched hyperparameter search, does VQC still lose? |
| **State overlap** | Are encoded quantum states almost classical (high fidelity to product states)? |

Superposition, interference and gates are **present in simulation**; they do not
guarantee a better decision boundary on a **classical tabular** dataset that
RBF-SVM already nearly solves.