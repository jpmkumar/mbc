# Stage-B locked protocol for a new patient fold

Runs the matched MLP versus VQC comparison on a frozen Stage-A cache using the
pre-registered Fold-0 settings, then scores the held-out test split exactly
once. Everything below runs **locally on CPU** in roughly eight minutes per
fold; no GPU and no Kaggle session are needed, because the heads see only the
8-dimensional cached features.

Settings are fixed by `preregistration/stage_b_locked_selection_fold0.json`
(raw features; MLP learning rate 1e-3, VQC learning rate 1e-2). Do not retune
them per fold.

## Step 0 — Place the downloaded cache

```bash
unzip -o ~/Downloads/stage_a_patient_cache_fold<N>_<stamp>.zip -d /tmp/fold<N>
cp /tmp/fold<N>/*/E3_histopath_fold<N>_*_features.pt \
   results/histopath/feature_cache/
```

Protocol v2 (`preregistration/stage_b_protocol_v2.md`) requires ten seeds per
head per fold. Steps 1 and 2 below produce the three-seed v1 result that is
still reported; steps 3 to 5 produce the v2 result that governs the
conclusion. Budget about 30 minutes of local CPU per fold for all of it.

## Step 1 — Train the matched heads (about 6 minutes)

```bash
.venv/bin/python scripts/run_vqc_stage_b_pilot.py \
  --feature-cache results/histopath/feature_cache/E3_histopath_fold<N>_histopath_seed42_pre_mag_mac_ulg_ulc_features.pt \
  --output-dir results/histopath/vqc_stage_b_pilot_fold<N> \
  --feature-transforms raw \
  --learning-rates 0.001 0.01 \
  --seeds 42 43 44
```

Both learning rates are trained so the pre-registered cell exists for each
head, and so the fold's own validation ranking can be reported afterwards as a
robustness check. That ranking must not change which checkpoint is scored.

## Step 2 — Score the held-out split once (about 2 minutes)

```bash
.venv/bin/python scripts/evaluate_vqc_stage_b_locked.py \
  --feature-cache results/histopath/feature_cache/E3_histopath_fold<N>_histopath_seed42_pre_mag_mac_ulg_ulc_features.pt \
  --pilot-dir results/histopath/vqc_stage_b_pilot_fold<N> \
  --locked-selection preregistration/stage_b_locked_selection_fold0.json \
  --output-dir results/histopath/vqc_stage_b_locked_test_fold<N> \
  --fold <N> --bootstrap-replicates 2000 --seeds 42 43 44
```

The evaluator refuses to overwrite an existing `locked_test_summary.json`, so
each fold can only be scored once without a deliberate deletion. It applies the
validation-derived threshold from each checkpoint, and reports a 95 percent
percentile interval on the paired AUPRC gap by resampling whole test patients.

## Step 3 — Seed the confirmatory directory (instant)

Reuse the trained seeds rather than repeating them, keeping only the two
pre-registered cells.

```bash
SRC=results/histopath/vqc_stage_b_pilot_fold<N>
DST=results/histopath/vqc_stage_b_confirmatory_fold<N>
mkdir -p $DST/raw_lr0p001 $DST/raw_lr0p01
cp $SRC/raw_lr0p001/mlp_seed4[234]_best_val_auprc.pt $DST/raw_lr0p001/
cp $SRC/raw_lr0p01/vqc_seed4[234]_best_val_auprc.pt  $DST/raw_lr0p01/
```

## Step 4 — Add seeds 45 to 51 (about 7 minutes)

```bash
CACHE=results/histopath/feature_cache/E3_histopath_fold<N>_histopath_seed42_pre_mag_mac_ulg_ulc_features.pt
DST=results/histopath/vqc_stage_b_confirmatory_fold<N>

.venv/bin/python scripts/run_vqc_stage_b_pilot.py --feature-cache $CACHE \
  --output-dir $DST --feature-transforms raw --learning-rates 0.001 \
  --models mlp --seeds 45 46 47 48 49 50 51 --summary-name mlp_seed_extension

.venv/bin/python scripts/run_vqc_stage_b_pilot.py --feature-cache $CACHE \
  --output-dir $DST --feature-transforms raw --learning-rates 0.01 \
  --models vqc --seeds 45 46 47 48 49 50 51 --summary-name vqc_seed_extension
```

## Step 5 — Score the ten-seed result once (about 6 minutes)

```bash
.venv/bin/python scripts/evaluate_vqc_stage_b_locked.py \
  --feature-cache $CACHE \
  --pilot-dir results/histopath/vqc_stage_b_confirmatory_fold<N> \
  --locked-selection preregistration/stage_b_locked_selection_fold0.json \
  --output-dir results/histopath/vqc_stage_b_confirmatory_test_fold<N> \
  --fold <N> --bootstrap-replicates 2000 \
  --seeds 42 43 44 45 46 47 48 49 50 51
```

## Step 6 — Aggregate across folds, once all five exist

```bash
.venv/bin/python scripts/aggregate_vqc_stage_b_folds.py \
  --summaries results/histopath/vqc_stage_b_confirmatory_test_fold*/locked_test_summary.json \
  --output results/histopath/vqc_stage_b_crossfold.json
```

This applies the pre-declared decision rules: the Nadeau-Bengio corrected 90%
interval must sit inside ±0.01 AUPRC for an equivalence claim, and the
corrected 95% interval must exclude zero with a mean beyond the margin for a
difference claim. Anything else is reported as inconclusive.

## Results so far

v1, three seeds:

| fold | mean VQC − MLP | seed median | patient-cluster 95% CI |
| --- | --- | --- | --- |
| 0, legacy cache | −0.00633 | −0.00621 | not available, cache predates patient IDs |
| 0, regenerated | −0.00253 | −0.00257 | [−0.00833, −0.00020] |
| 1 | −0.00026 | −0.00025 | [−0.00051, −0.00009] |
| 2 | +0.00441 | −0.00005 | [+0.00257, +0.00659] |

v2, ten seeds:

| fold | mean VQC − MLP | seed median | MLP seed spread | VQC seed spread |
| --- | --- | --- | --- | --- |
| 0 | −0.00241 | −0.00257 | 0.00075 | 0.00331 |
| 1 | +0.00107 | −0.0000005 | 0.01178 | 0.00058 |
| 2 | +0.00063 | −0.00005 | 0.01342 | 0.00564 |

Fold 0's regenerated split is identical to the legacy one down to the class
counts, so the fold generator is deterministic across code versions. Its
encoder is not: early stopping fired at 7 epochs instead of 16, giving weaker
Stage-A features, and the fold-level gap moved from −0.00633 to −0.00241 as a
result. Same patients, same split, different representation. Report the
regenerated value and cite the legacy one as evidence that the gap depends on
Stage-A training luck rather than on the head.

Ten seeds move both folds toward zero and reveal the real pattern: when both
heads converge, their test AUPRC agrees to about five decimal places, and the
fold-level mean is set almost entirely by how often a head fails to converge.
The MLP produced one badly converged seed in each fold, roughly 0.012 to 0.013
AUPRC below its own median; the VQC produced none beyond the flag threshold.
Report the mean as the primary statistic, with the median and the seed spread
beside it.

Folds 3 and 4 are still required before the cross-fold decision in Step 6 can
be treated as final.
