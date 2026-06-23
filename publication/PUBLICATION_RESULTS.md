# Publication Results Summary

Auto-generated from `publication/publication_metrics.json`. Re-run: `python scripts/generate_publication.py`

## Primary result (recommended for paper)

**Stage A (enhanced)** on CBIS-DDSM test ($n=445$):

| Metric | Value |
|--------|-------|
| Balanced accuracy | **70.1%** |
| Malignant recall (sensitivity) | **84.7%** |
| Precision | 65.5% |
| F1 | 73.9% |
| AUC | 76.3% |

Confusion matrix: TN=124, FP=99, FN=34, TP=188

## Figures for manuscript

| File | Use in paper |
|------|----------------|
| `figures/fig_mammo_metrics_comparison.png` | Results section — metric bars |
| `figures/fig_confusion_matrices.png` | Results — Stage A vs B confusion |
| `figures/fig_dataset_splits.png` | Experimental setup |
| `figures/fig_pilot_vs_imaging.png` | Discussion — pilot bridge |
| `figures/fig_training_stages.png` | Method — training strategy |

## LaTeX

Include in `main.tex`:

```latex
\input{publication/tables/publication_tables.tex}
```

Or copy tables from `publication/tables/publication_tables.tex`.

## Key claims (copy-ready)

> On CBIS-DDSM mammography (2,966 ROI images; 445-image test split), enhanced classical Stage A achieved **70.1% balanced accuracy**, **84.7% malignant recall**, and **0.763 AUC**, outperforming the initial pipeline (61.8%) and Benedetti VQC Stage B (69.0% balanced accuracy, 59.9% recall).

> Hybrid VQC Stage B did not exceed the classical head on test metrics; **Stage A enhanced is the recommended deployment model**.
