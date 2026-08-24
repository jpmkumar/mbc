# Paper B — Histopathology IDC hybrid QML (null / equivalence)

**Data:** Kaggle Breast Histopathology Images — **277,524 patches**, **279 patients** (image patches, not tabular).

**Claim:** Under patient-level CV, focal loss, TTA, and matched classical controls, a
**bolt-on VQC head does not beat** a parameter-matched MLP; Stage-B frozen-head
equivalence and end-to-end width ablations (q4/q8/q12) extend the evaluation.

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
