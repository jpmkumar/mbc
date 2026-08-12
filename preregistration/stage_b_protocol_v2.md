# Stage-B confirmatory protocol v2

Declared **before** any v2 result was computed. Recorded in git so the
declaration timestamp precedes the first v2 run.

## Why v2 exists

Protocol v1 used three initialization seeds per head per fold. Fold 2 showed
that this is too few: a single MLP run converged 0.0134 test AUPRC below its
two siblings, which moved the fold-level mean gap by +0.0044 and flipped its
sign. That shift is roughly 40 percent of the entire between-fold spread
observed so far, so seed noise is a first-order contributor to the cross-fold
estimate rather than a rounding detail. The patient-cluster interval cannot
absorb it, because that interval resamples patients while holding the trained
checkpoints fixed.

v2 therefore raises the seed budget for every fold, not only the folds where
v1 looked unstable. Applying the change uniformly is the point: a seed
increase applied only to folds scored after an inconvenient result would be an
outcome-triggered protocol change.

## What does not change

Head selection stays locked to `stage_b_locked_selection_fold0.json`: raw
features, MLP at learning rate 1e-3, VQC at learning rate 1e-2. The training
budget stays at 2000 optimizer steps, batch 64, 4096 train and 1024 validation
patches per class, 8 qubits, 2 layers, angle-Y encoding, linear entanglement.
Thresholds still come from each checkpoint's validation subset. The practical
margin stays at 0.01 AUPRC. No hyperparameter is re-tuned on any fold.

## What changes

**Seeds.** Ten per head per fold, 42 through 51. Seeds 42 through 44 already
exist for folds 0 through 2 and are reused rather than retrained, so this adds
seven runs per head per fold.

**Fold-level statistic.** Unchanged as the mean paired VQC minus MLP test
AUPRC across seeds. The seed median and the per-head seed spread are reported
beside it. With ten seeds a single bad run moves the mean by about a tenth of
its deviation instead of a third.

**Instability rate.** For each head, the fraction of seeds whose test AUPRC
falls more than 0.01 below that head's per-fold median, reported per fold and
pooled. This treats optimization fragility as a result rather than as noise to
average away.

**Cross-fold analysis.** The fold is the unit of analysis. The five fold-level
mean gaps are combined using the corrected resampled t statistic of Nadeau and
Bengio, which inflates the naive variance by `1/k + n_test/n_train` to account
for the training-set overlap that makes ordinary k-fold t-tests
anticonservative. Reporting the uncorrected interval alongside it is allowed;
the corrected interval is the one that governs any claim.

## Decision rules, fixed in advance

- **Practical equivalence** is claimed only if the corrected 90 percent
  interval for the mean cross-fold gap lies entirely inside ±0.01 AUPRC. This
  is the two one-sided tests procedure at α = 0.05.
- **A difference** is claimed only if the corrected 95 percent interval
  excludes zero *and* the mean gap exceeds 0.01 in absolute value. An interval
  that excludes zero while sitting inside the margin supports neither claim on
  its own and is reported as a measurable but clinically negligible gap.
- **Neither** outcome is reported as "inconclusive with the observed interval",
  not as a null result.
- Both the v1 three-seed and the v2 ten-seed analyses are reported. If they
  disagree, the disagreement is reported as a finding about protocol
  sensitivity; v2 governs the conclusion because it was declared in advance and
  is better powered.

## Known limitations, acknowledged in advance

Five folds give four degrees of freedom, so the corrected interval will be
wide. That is honest rather than fixable at this sample size. Folds share
training patients by construction, which the correction mitigates but does not
eliminate. Fold 0's existing cache predates patient-ID storage, so it supports
paired gaps but no patient-cluster interval; if it is regenerated with patient
IDs, the regenerated cache replaces it and the legacy result is reported as a
provenance check rather than dropped silently.
