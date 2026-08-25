# Pre-outcome precision simulation

**Run:** 21 August 2026, before any labelled model comparison.  
**Script:** `papers/paper_c/scripts/simulate_precision.py`.  
**Design:** 500 trials per cell using only the 279 observed case-identifier
sizes/class counts, capped at 100 simulated patches per class per identifier.

The simulation used paired case effects and patch noise, then varied assumed
optimization logit noise (`SD=0.02, 0.05, 0.10`), representation signal shift
(`0, 0.05, 0.10`) and averaged probe seeds (`1, 3, 5`). It did not use
embeddings or model predictions.

## Decision

Use **three deterministic probe seeds (42, 43, 44)** for each condition, average
their patch probabilities before case-level inference, and retain seed-wise
results only as an optimization-variability diagnostic.

Under moderate assumed optimization noise (`SD=0.05`), increasing from one to
three seeds reduced the null AP perturbation from about `−0.00026` to
`−0.00009` and its simulation SD from `0.00009` to `0.00005`. Increasing from
three to five seeds changed the mean by only about `0.00003`, negligible
relative to a `0.01` material-effect scale. Under high assumed noise
(`SD=0.10`), three seeds similarly reduced the null perturbation by about
two-thirds; five seeds offered a smaller additional reduction.

This is an engineering precision decision, not a power guarantee. Whole-case
bootstrap uncertainty remains the inferential interval, and seeds are not
treated as independent sample size.
