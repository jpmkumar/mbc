# Paper A confirmatory and sensitivity outputs

| Directory | What |
|-----------|------|
| `.` (this folder) | Locked confirmatory 1×5, MSE, 30 VQC epochs, seed 42 |
| `repeated_2x5/` | Same protocol, `--n-repeats 2` (10 scores) |
| `vqc_ce80/` | `--vqc-loss ce --vqc-epochs 80` (equalised with matched MLP) |
| `oof_scores.json` | Out-of-fold malignant scores for Fig 3 (does not replace confirmatory JSON) |
| `quantum_sweep.json` | Side ansatz / kernel sweep |
| `hybrid_sweep.json` | Side end-to-end / fusion hybrid sweep |

Do not overwrite the locked `confirmatory_*.json` in this folder.
