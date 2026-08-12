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

## Results so far

| fold | MLP test AUPRC | VQC test AUPRC | VQC − MLP | patient-cluster 95% CI |
| --- | --- | --- | --- | --- |
| 0 | 0.89197 | 0.88564 | −0.00633 | not available, cache predates patient IDs |
| 1 | 0.88071 | 0.88045 | −0.00026 | [−0.00051, −0.00009] |

Both folds fall inside the pre-specified ±0.01 practical margin. Once folds 2
through 4 are scored, aggregate the five fold-level paired differences before
making any cross-fold or equivalence statement.
