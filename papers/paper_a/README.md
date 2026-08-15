# Paper A — Tabular WBCD hybrid QML

**Data:** Wisconsin Breast Cancer **Diagnostic** (and/or Original) — **569 samples**,
**30 numeric features** (tabular, not images).

**Claim:** A hybrid classical–quantum classifier can be **competitive** when the
quantum circuit operates on the **native feature vector**, not on a CNN bottleneck.

**Contrast with Paper B:** Histopath patches route images through EfficientNet first;
quantum only sees an 8-D embedding. Paper A tests the regime where literature often
reports hybrid gains.

## Status

| Item | Status |
|------|--------|
| Pilot metrics (SVM / VQC / hybrid) | In `publication/publication_metrics.json` → `pilot_wbcd_tabular` |
| Rigorous patient/sample CV + matched controls | **To do** for journal submission |
| Dedicated LaTeX manuscript | **Not started** (use this workspace when drafting) |

## Directory layout

| Path | Git | Purpose |
|------|-----|---------|
| [`AGENTS.md`](AGENTS.md) | yes | Agent brief |
| [`SKILL.md`](SKILL.md) | yes | Portable Cursor skill |
| [`scripts/export_curated_metrics.py`](scripts/export_curated_metrics.py) | yes | Snapshot pilot + future runs |
| [`results/`](results/) | **no** | Curated metrics |
| [`assets/`](assets/) | **no** | Local notes, reference PDFs |

## Quick commands

```bash
python papers/paper_a/scripts/export_curated_metrics.py
```

## Pilot numbers (reference only — expand before submission)

From `publication/publication_metrics.json`:

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| SVM (RBF) | 0.974 | 0.979 | 0.996 |
| VQC standalone | 0.939 | 0.954 | 0.978 |
| Hybrid MLP+VQC | 0.877 | 0.908 | 0.938 |

These are **exploratory**; Paper A needs proper CV, matched capacity, and
imbalance-aware metrics before claiming Q1 readiness.
