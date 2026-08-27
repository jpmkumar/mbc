# Curated metrics snapshot

`curated_metrics.json` and `export_log.txt` are **released with the code**. They
are the experiment summaries referred to in the article's Data Availability
Statement, and they let a reader check every published number without the full
run directories (which stay local, under repo-root `results/`).

Regenerate with:

```bash
python papers/paper_b/scripts/export_curated_metrics.py
```

## Contents

`curated_metrics.json` carries an `exported_utc` stamp, the resolved source path
of every input, and these blocks under `metrics`:

| Block | Backs |
| --- | --- |
| `deployed_arms` | Table 3 and Figure 5 — per-fold test metrics for E2, E2b, E3 |
| `stage_b_v2_final`, `stage_b_v2_sensitivity` | Table 7 pre-registered locked-rate columns |
| `stage_b_v3_final`, `stage_b_v3_sensitivity` | Table 7 secondary per-fold-rate columns |
| `width_matrix_analysis` | Table 8, Figure 10, and the stage-selection counts |
| `server_width_matrix` | Raw server width cells behind `width_matrix_analysis` |

`deployed_arms` is resolved per fold from the run directories rather than from
`results/histopath/cv_summary.json`, which is a stale single-arm artifact that
cannot reproduce Table 3. The lookup mirrors `load_arm` in
`papers/paper_b/scripts/plot_paper_figures.py`; if one changes, change both.

## Checking the published numbers

Table 3 means and standard deviations are the per-fold values in
`deployed_arms` aggregated over the five folds. The Table 4 Friedman statistics
are `scipy.stats.friedmanchisquare` over the three arms' five per-fold values,
which reproduces chi-square 1.60 (F1), 2.80 (balanced accuracy) and 1.20
(AUPRC). Stage-selection counts come from `quantum_stage_evidence` in each width
arm: two server 12-qubit cells and two Kaggle 4-qubit cells select Stage B, so
Stage A wins fourteen of the eighteen end-to-end cells.
