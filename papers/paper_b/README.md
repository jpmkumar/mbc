# Paper B — Histopathology IDC hybrid QML (null / equivalence)

**Data:** Kaggle Breast Histopathology Images — **277,524 patches** under
**279 public case identifiers** (image patches, not tabular). The public
derivative does not establish that these identifiers are 279 independent
patients, so grouping is described by case ID throughout.

**Claim:** Under case-ID-grouped CV, focal loss, TTA, and capacity-controlled
classical baselines, the quantum-active stages of the E3 hybrid schedule never
survive validation selection: every deployed E3 predictor uses the Stage-A linear
route. Stage-B frozen-head evaluation and end-to-end width runs (q4/q8/q12) are
reported separately and must not be conflated with direct VQC performance in the
stage-selected E3 aggregates.

**Active experiments:** Kaggle width cells (e.g. `FOLD=3`, `N_QUBITS=4`), server-primary matrix when campus access returns.

## Directory layout

| Path | Git | Purpose |
|------|-----|---------|
| [`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md) | yes | Paths to cite in the paper |
| [`scripts/export_curated_metrics.py`](scripts/export_curated_metrics.py) | yes | Histopath metrics export |
| [`results/`](results/) | **no** | Curated JSON for tables |

## Quick commands

```bash
python papers/paper_b/scripts/export_curated_metrics.py
```

See [`../README.md`](../README.md) for Paper A vs Paper B split.
