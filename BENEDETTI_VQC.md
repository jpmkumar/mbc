# Benedetti-aligned VQC (Stage B retry)

Implements the PQC pattern from Benedetti et al. (2019, *Quantum Sci. Technol.* 043001):

1. **Classical pre-encoding** — LayerNorm on 8-d compressed features  
2. **Angle encoding** — sigmoid → \([0, \pi]\) rotations  
3. **Shallow hardware-efficient ansatz** — RY/RZ + linear CNOT  
4. **Full readout** — PauliZ on all 8 qubits  
5. **Classical post-processing** — `Linear(8, 2)`

Config: `configs/benedetti_vqc.yaml`

## Colab: Stage B only (reuse Stage A backbone)

After mounting Drive and pulling latest code:

```bash
# Copy Stage A checkpoint from Drive
mkdir -p results/checkpoints
cp /content/drive/MyDrive/mbc_colab/checkpoints/E3_hybrid_seed42.pt results/checkpoints/

python experiments/run_training.py \
  --config configs/benedetti_vqc.yaml \
  --experiment hybrid \
  --modality mammo \
  --stage b \
  --reset-stages \
  --no-auto-resume \
  --resume results/checkpoints/E3_hybrid_seed42.pt
```

Outputs:

- `results/checkpoints/E3_hybrid_benedetti_seed42.pt` (best)
- `results/checkpoints/E3_hybrid_benedetti_seed42_latest.pt` (resume)
- `results/E3_hybrid_benedetti_seed42_metrics.json`

Copy checkpoints back to Drive when done.

## Evaluate

```bash
python scripts/analyze_results.py \
  --checkpoint results/checkpoints/E3_hybrid_benedetti_seed42.pt \
  --eval-stage b \
  --modality mammo
```

Compare with Stage A:

```bash
python scripts/analyze_results.py \
  --checkpoint results/checkpoints/E3_hybrid_seed42.pt \
  --eval-stage a \
  --modality mammo
```

## Key training settings

| Setting | Value |
|---------|-------|
| `freeze_backbone_in_stage_b` | true |
| `backbone_eval_in_stage_b` | true |
| `cache_frozen_backbone_features` | true (Stage B runs on cached 8-d vectors) |
| `classical_device` | auto (GPU on Colab) |
| `lr_quantum` | 5e-5 |
| `quantum_weight_decay` | 1e-3 |
| `stage_b_epochs` | 20 |
| `stage_c_epochs` | 0 (skip joint fine-tune initially) |

See `PERFORMANCE.md` for why training was slow and expected runtimes after the refactor.
