# Pre-registered Stage-B settings

`stage_b_locked_selection_fold0.json` is the verbatim selection produced by the
Fold-0 exploratory pilot (`scripts/run_vqc_stage_b_pilot.py`), before the Fold-0
held-out test split was ever scored. It records `held_out_test_evaluated: false`
and selects each head by validation AUPRC only:

| head | features | learning rate |
| --- | --- | --- |
| MLP | raw | 1e-3 |
| VQC | raw | 1e-2 |

Confirmatory folds must reuse this file rather than re-tuning. Pass it to
`scripts/evaluate_vqc_stage_b_locked.py --locked-selection`, which refuses any
selection file that admits prior test access. Training budget is held fixed
across folds and heads: 2000 optimizer steps, batch 64, 4096 train and 1024
validation patches per class, 8 qubits, 2 layers, angle-Y encoding, linear
entanglement.
