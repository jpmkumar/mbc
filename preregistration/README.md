# Pre-registration index

| Protocol | Paper / track | Status |
|----------|---------------|--------|
| [`paper_d_breakhis_protocol.md`](paper_d_breakhis_protocol.md) | **Paper D** BreakHis transfer | **LOCKED** follow-on (2026-09-16) |
| [`paper_c_protocol.md`](paper_c_protocol.md) | Paper C grouping / FOV | DRAFT |
| [`staged_hybrid_fair_warmup_protocol.md`](staged_hybrid_fair_warmup_protocol.md) | Paper B Path A | declared |
| [`histopath_vqc_width_protocol.md`](histopath_vqc_width_protocol.md) | Paper B width ablation | declared |
| [`histopath_vqc_width_server_protocol.md`](histopath_vqc_width_server_protocol.md) | Paper B server width | declared |
| [`stage_b_protocol_v3_nested_secondary.md`](stage_b_protocol_v3_nested_secondary.md) | Paper B Stage-B | declared |

---

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
