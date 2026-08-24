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

## Steps 4b to 5b — Secondary nested-rate analysis (about 20 minutes per fold)

Declared in `preregistration/stage_b_protocol_v3_nested_secondary.md`. Run this
**after** the fold's v2 result exists, never instead of it. Both heads are
trained at both candidate rates, each head's rate is then chosen from that
fold's validation split alone, and the test split is scored a second time.

Training is bit-reproducible: retraining fold 3's MLP at 1e-3 with seed 42
returned 0.9471811078241599 validation AUPRC, identical to the v2 run to all
sixteen digits. So all four cells are retrained here rather than part-copied
from the v2 directories. That keeps the procedure identical on every fold,
including fold 0 whose pilot directory no longer exists locally, and it makes
the self-check below exact rather than approximate.

```bash
N=3
CACHE=results/histopath/feature_cache/E3_histopath_fold${N}_histopath_seed42_pre_mag_mac_ulg_ulc_features.pt
DST=results/histopath/vqc_stage_b_nested_fold${N}

.venv/bin/python scripts/run_vqc_stage_b_pilot.py --feature-cache $CACHE \
  --output-dir $DST --feature-transforms raw --learning-rates 0.001 0.01 \
  --seeds 42 43 44 45 46 47 48 49 50 51 --summary-name nested_grid

.venv/bin/python scripts/select_vqc_stage_b_nested.py --pilot-dir $DST \
  --fold $N --output results/histopath/vqc_stage_b_nested_selection_fold${N}.json

.venv/bin/python scripts/evaluate_vqc_stage_b_locked.py --feature-cache $CACHE \
  --pilot-dir $DST \
  --locked-selection results/histopath/vqc_stage_b_nested_selection_fold${N}.json \
  --output-dir results/histopath/vqc_stage_b_nested_test_fold${N} \
  --fold $N --bootstrap-replicates 2000 \
  --seeds 42 43 44 45 46 47 48 49 50 51
```

Budget about 20 minutes of training and 6 minutes of scoring per fold. When the
selector reports that it reproduced the locked rates, the fold's v3 numbers must
equal its v2 numbers exactly; a mismatch means the pipeline, not the protocol,
is at fault.

## Step 6 — Aggregate across folds, once all five exist

```bash
.venv/bin/python scripts/aggregate_vqc_stage_b_folds.py \
  --summaries results/histopath/vqc_stage_b_confirmatory_test_fold{0,1,2,3,4}/locked_test_summary.json \
  --output results/histopath/vqc_stage_b_crossfold_v2_final.json

.venv/bin/python scripts/aggregate_vqc_stage_b_folds.py \
  --summaries results/histopath/vqc_stage_b_nested_test_fold{0,1,2,3,4}/locked_test_summary.json \
  --output results/histopath/vqc_stage_b_crossfold_v3_final.json

# Pre-declared sensitivity analyses excluding the Fold-0 selection fold.
.venv/bin/python scripts/aggregate_vqc_stage_b_folds.py \
  --summaries results/histopath/vqc_stage_b_confirmatory_test_fold{1,2,3,4}/locked_test_summary.json \
  --expected-folds 4 \
  --output results/histopath/vqc_stage_b_crossfold_v2_sensitivity_folds1_4.json

.venv/bin/python scripts/aggregate_vqc_stage_b_folds.py \
  --summaries results/histopath/vqc_stage_b_nested_test_fold{1,2,3,4}/locked_test_summary.json \
  --expected-folds 4 \
  --output results/histopath/vqc_stage_b_crossfold_v3_sensitivity_folds1_4.json
```

This applies the pre-declared decision rules: the Nadeau-Bengio corrected 90%
interval must sit inside ±0.01 AUPRC for an equivalence claim, and the
corrected 95% interval must exclude zero with a mean beyond the margin for a
difference claim. Anything else is reported as inconclusive.

## Final results

v1, three seeds:

| fold | mean VQC − MLP | seed median | patient-cluster 95% CI |
| --- | --- | --- | --- |
| 0, legacy cache | −0.00633 | −0.00621 | not available, cache predates patient IDs |
| 0, regenerated | −0.00253 | −0.00257 | [−0.00833, −0.00020] |
| 1 | −0.00026 | −0.00025 | [−0.00051, −0.00009] |
| 2 | +0.00441 | −0.00005 | [+0.00257, +0.00659] |
| 3 | +0.01592 | +0.00948 | [+0.00859, +0.02490] |

v2, ten seeds:

| fold | mean VQC − MLP | seed median | MLP seed spread | VQC seed spread |
| --- | --- | --- | --- | --- |
| 0 | −0.00241 | −0.00257 | 0.00075 | 0.00331 |
| 1 | +0.00107 | −0.0000005 | 0.01178 | 0.00058 |
| 2 | +0.00063 | −0.00005 | 0.01342 | 0.00564 |
| 3 | +0.01553 | +0.01365 | 0.03928 | 0.01123 |
| 4 | +0.00748 | +0.00334 | 0.02499 | 0.01874 |

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

Fold 3 breaks that pattern and needs its own paragraph. Its ten-seed median
stays at +0.01365 instead of collapsing toward zero, so it is not one stray
MLP run. The two heads still share a ceiling — the MLP's best seed reaches
0.88927 test AUPRC against the VQC's 0.89008, a gap of 0.00081 — but the MLP
lands within 0.005 of that ceiling in only 3 of 10 seeds while the VQC manages
9 of 10.

The cause is the locked learning rate, not the head. The fold-3 pilot also
trained the non-registered cells, and at 1e-2 the MLP scores 0.94894,
0.94894, 0.94894 validation AUPRC across three seeds: perfectly stable and
slightly *above* the VQC's own locked 1e-2 numbers of 0.94882, 0.94696,
0.94587. So the pre-registered 1e-3 rate, chosen on fold 0, simply fails to
transfer to fold 3, and the primary comparison there pits a badly tuned MLP
against a well tuned VQC. This diagnostic is post-hoc and uses validation
data only; it explains the fold, it does not license editing it.

Note the direction of the bias. It inflates the VQC and therefore pushes the
cross-fold result away from the equivalence conclusion, so it is not
self-serving. With four folds the corrected 90% interval is
[−0.00994, +0.01735] around a mean of +0.00370, and the pre-declared rules
return **inconclusive**.

The transfer problem is handled by the secondary analysis in steps 4b to 5b:
each head's rate is chosen inside its own fold from validation data only, so
the comparison is between two fairly tuned heads. Protocol v2 stays primary
because v3 was added after seeing a v2 result, and that ordering is disclosed
in the declaration rather than smoothed over. Both results are reported side by
side, and no fold is scored a third time.

Fold 4 independently reproduces the transfer problem. Under v2 the locked
1e-3 MLP rate yields +0.00748 mean VQC-minus-MLP AUPRC, but both heads are
unstable and the seed median is +0.00334. Validation selection moves the MLP
to 1e-2; under v3 the gap reverses to −0.00416 mean and −0.00207 median, both
inside the practical margin.

v3, ten seeds with each head's rate selected independently from that fold's
validation split:

| fold | selected MLP LR | selected VQC LR | mean VQC − MLP | seed median | MLP seed spread | VQC seed spread |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1e-2 | 1e-2 | −0.00254 | −0.00267 | 0.00010 | 0.00331 |
| 1 | 1e-2 | 1e-2 | −0.00013 | −0.000003 | 0.000006 | 0.00058 |
| 2 | 1e-2 | 1e-2 | −0.00083 | −0.00017 | 0.00011 | 0.00564 |
| 3 | 1e-2 | 1e-2 | −0.00143 | −0.00038 | 0.00005 | 0.01123 |
| 4 | 1e-2 | 1e-2 | −0.00416 | −0.00207 | 0.00114 | 0.01874 |

### Final cross-fold decisions

| analysis | folds | mean gap | corrected 90% CI | TOST p | decision |
| --- | --- | --- | --- | --- | --- |
| v2 primary, locked Fold-0 rates | 0–4 | +0.00446 | [−0.00613, +0.01504] | 0.16345 | inconclusive |
| v2 sensitivity, selection fold excluded | 1–4 | +0.00618 | [−0.00583, +0.01818] | 0.25394 | inconclusive |
| v3 secondary, nested validation selection | 0–4 | −0.00182 | [−0.00415, +0.00051] | 0.00085 | practical equivalence |
| v3 sensitivity, selection fold excluded | 1–4 | −0.00164 | [−0.00467, +0.00139] | 0.00371 | practical equivalence |

The primary pre-registered conclusion is **inconclusive at the observed
interval width**, not a null result. The outcome-triggered but pre-declared
secondary analysis shows **practical equivalence within ±0.01 AUPRC** when
both heads receive the same fold-specific validation selection. Excluding the
Fold-0 selection fold leaves both decisions unchanged. The manuscript must
report both conclusions and the ordering of the protocols: v3 explains the v2
failure but does not replace it.

The scientific finding is therefore narrower than either “quantum advantage”
or “VQC failure.” The VQC and matched MLP heads have practically equivalent
discrimination after fair tuning. The apparent VQC gains under locked rates
come from poor transfer of the MLP learning rate and seed-level convergence
failures. VQC seeds remain less stable on folds 3 and 4, so equivalence in
AUPRC does not imply equivalent optimization reliability.
