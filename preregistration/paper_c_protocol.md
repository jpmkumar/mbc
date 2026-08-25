# Paper C protocol — grouping and field-of-view effects in pathology foundation models

**Status:** DRAFT — not yet outcome-locked.  
**Created:** 2026-08-21.  
**Lock condition:** change status to `LOCKED`, record the source commit, model
snapshot revisions, split/index hashes and protocol SHA-256 before any labelled
comparative probe is run.

Engineering tests that load models, inspect dimensions, assemble synthetic
mosaics or extract a small unlabeled cache are permitted while this document is
draft. Comparative performance inspection is not.

## 1. Research questions

1. How much does random patch cross-validation overestimate case-balanced AUPRC
   relative to case-identifier-grouped cross-validation on the IDC archive?
2. Does approximately pretraining-matched physical field of view (`K=9`) improve
   grouped performance relative to an upsampled isolated patch (`K=1`)?
3. Does the field-of-view effect replicate on institution-held-out BCSS breast
   histology?
4. Do model rankings change across grouping protocols or fields of view?

## 2. Cohorts and units

### IDC

- 277,524 RGB patches; 78,786 IDC-positive and 198,738 non-IDC.
- 279 public case identifiers.
- Grouping unit: archive case identifier. It is not asserted to be a verified
  patient identifier.
- Frozen grouped outer folds:
  `data/splits/histopath_kaggle/folds/fold_*/`.

### BCSS

- 151 breast-cancer slides/ROIs with pixel-level masks.
- Revision-pinned CC0 mirror: `MedOtter/BCSS` at
  `502d4a3fbc77dbaca6f4664c19e2379ff077d418`.
- Grouping unit: TCGA patient identifier.
- External test institutions: `OL`, `LL`, `E2`, `EW`, `GM`, `S3`, following the
  published institution-held-out split used by the EVA benchmark.
- Remaining institutions form the development pool. Patients are assigned by
  `SHA256("32026|" + patient_id) mod 100`: values 0--69 train, 70--84
  validation and 85--99 calibration.

BCSS is an external replication of the field-of-view mechanism, not an
identical-target validation cohort.

## 3. Models

Reference encoder for both co-primary contrasts: **UNI2-h**.

Predeclared panel:

1. UNI2-h;
2. Virchow2;
3. Phikon-v2;
4. DINOv2 ViT-L/14;
5. ImageNet ResNet-50.

Primary adaptation is frozen embeddings plus a deterministic linear probe.
Exact Hugging Face or torchvision revisions, vendor transforms, output pooling,
licences and artifact SHA-256 values must be entered in
`papers/paper_c/config/model_registry.json` before protocol lock. A model whose
revision cannot be frozen or whose licence prevents the intended analysis is
reported as unavailable; it is not silently replaced after outcomes are seen.

## 4. Partitions and probe tuning

### IDC grouped

Use the existing five outer case-ID folds. Within each outer-training set,
stratify case identifiers by IDC-positive patch-ratio bin and assign
approximately 75%/12.5%/12.5% to inner train/validation/calibration, using
effective seed `1042 + outer_fold`. Freeze identifier lists before extraction.

### IDC random

Assign every patch to one of five label-stratified outer folds using seed 42.
Inside each outer-training pool, create label-stratified
75%/12.5%/12.5% train/validation/calibration partitions with seed
`2042 + outer_fold`. Patches sharing a case may cross partitions by design.

### Probe

- Standardise features using inner-train statistics only.
- Logistic-loss linear probe fitted by deterministic stochastic gradient
  descent with L2 regularisation (`SGDClassifier`, tolerance `1e-4`, maximum
  200 epochs).
- Candidate L2 `alpha` values:
  `{1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`.
- Fit with case-balanced sample weights in grouped analyses and ordinary
  class-balanced weights in random-patch analyses.
- Select regularisation by validation case-balanced AUPRC in grouped analyses
  and validation AUPRC in random analyses.
- Fit temperature scaling on calibration data only.
- Select the binary operating threshold on validation data only by maximum MCC;
  ties choose the higher threshold.

The search grid and fitting tolerance are identical across models and
conditions.

## 5. Context construction

IDC crop coordinates are interpreted on a 50-pixel grid after an audit confirms
spacing. Context levels:

- `K=1`: isolated 50×50 centre patch;
- `K=3`: 150×150 field;
- `K=5`: 250×250 field;
- `K=9`: 450×450 field.

Every assembled image is transformed to the encoder's documented input using
its vendor preprocessing. The centre patch supplies the label.

Primary context population: centres with every required `K=9` neighbour.
Identical centres are used for all `K` levels. Padded mosaics are excluded from
the primary analysis and reported only as a sensitivity analysis with missing
fraction recorded.

The pre-outcome coordinate audit found 49,798 complete `K=9` centres (21,706
non-IDC; 28,092 IDC-positive) from 216/279 case identifiers. Eligibility-index
SHA-256:
`e970ae0b03b4c1f9fbbd8bebcbe6082b309c610a0844f4c55cfd45a4df20f1b7`.
This population shift is handled by pairing the same centres across `K` and by
case balancing; it remains a stated limitation.

Mosaics are forbidden in the IDC random-patch protocol.

For BCSS, sample a fixed centre grid from native 0.25 µm/pixel images. Ignore
mask classes `outside_roi`, `exclude` and `undetermined`. Class `tumor` is
positive; all other valid tissue classes, including DCIS, are negative. Require
at least 80% valid pixels and 80% class purity in the centre 50×50 region.

## 6. Co-primary estimands

1. `Δ_protocol`: IDC random-patch minus case-ID-grouped case-balanced AUPRC for
   UNI2-h at `K=1`, using paired out-of-fold predictions for all patches.
2. `Δ_context`: IDC grouped `K=9` minus `K=1` case-balanced AUPRC for UNI2-h on
   complete-neighbourhood centres.

Holm's procedure controls family-wise error at 0.05 across these two tests.
Two-sided 95% intervals are reported. No minimum effect is assumed; estimates
and intervals, not a significance label alone, determine interpretation.

## 7. Ordered secondary analyses

1. Protocol and context contrasts for the remaining encoders.
2. Model-rank Kendall correlation and rank changes across protocols/contexts.
3. Unweighted AUPRC and case-balanced/clustered AUROC.
4. MCC, sensitivity and specificity at validation-locked thresholds.
5. Calibration intercept/slope, Brier score, log loss and reliability plots.
6. BCSS `K=9 − K=1` context replication for UNI2-h, followed by other encoders.
7. Empirical risk–coverage curves.
8. BCSS ROI-level tumour-area agreement.
9. A separately declared last-block fine-tuning sensitivity analysis, only if
   the full frozen-probe primary matrix is complete.

No decision-curve, clinical workload, formal conformal, fairness, carbon or
distillation analysis is primary or confirmatory.

## 8. Statistical inference

- Produce one out-of-fold prediction per eligible observation and condition.
- Run deterministic probe seeds 42, 43 and 44 for every condition, then average
  their patch probabilities before inferential resampling. This count was fixed
  by the pre-outcome simulation in
  `papers/paper_c/results/precision_simulation.md`.
- Use paired whole-case bootstrap for IDC and institution-stratified patient
  bootstrap for BCSS.
- Development intervals use at least 2,000 replicates; locked manuscript
  intervals use 10,000.
- Recompute both sides of every contrast within each bootstrap replicate.
- Use percentile intervals. For co-primary two-sided p-values, centre the paired
  bootstrap distribution at the observed contrast and apply the finite-replicate
  `(extreme + 1)/(B + 1)` correction before Holm adjustment.
- Treat patches, folds and seeds as correlated—not independent sample size.
- Do not use ordinary patch-level DeLong tests or fold-level `n=5` t-tests.

Before lock, run a simulation-based precision analysis using only group sizes,
label prevalences and assumed effect grids, without model predictions.

## 9. Falsification rules

- If `Δ_protocol` is small with an interval excluding a material optimism
  effect, do not claim that IDC literature is substantially inflated.
- If `Δ_context` is small with an interval excluding a material gain, reject the
  context-starvation mechanism for UNI2-h.
- If BCSS does not replicate the context direction, restrict conclusions to the
  IDC derivative.
- If ranks remain stable, reject the rank-instability hypothesis.

No result authorises changing the co-primary metrics, reference encoder,
eligible-centre population or correction family.

## 10. Reproducibility and stopping

Every run records:

- source commit and container image ID;
- dependency-lock SHA-256;
- dataset, split, embedding-index and eligible-centre hashes;
- model repository revision plus weight/config SHA-256;
- transform configuration, precision, GPU and CUDA versions;
- probe hyperparameter grid and selected value.

Completed prediction bundles are immutable. Operational failures may resume from
verified checkpoints; completed finite runs are not repeated because their
results are inconvenient.

## 11. Protocol-lock record

Complete before changing status to `LOCKED`:

- Source commit: `TBD`
- Container image ID: `TBD`
- Dependency lock SHA-256: `TBD`
- IDC dataset SHA-256: `TBD`
- IDC outer split manifest SHA-256:
  `ac9d06510ca3555e6d481f1f870ab92fc69411ee3b9fa53da9aa7a60ce9bd013`
- IDC inner split summary SHA-256:
  `f2fe95231edd3fe036b1617421609f05590fa1662782bf91de2b12e31b037b33`
- IDC coordinate/eligibility index SHA-256:
  `e970ae0b03b4c1f9fbbd8bebcbe6082b309c610a0844f4c55cfd45a4df20f1b7`
- IDC filepath-keyed protocol manifest SHA-256:
  `b28e4acc2c3482256c17971cd422e39713d5ca4df860bc2929a31b3104caa266`
- BCSS dataset SHA-256: `TBD`
- BCSS split manifest SHA-256: `TBD`
- Model registry SHA-256:
  `28da488906a2bb4e3725cc279624e6e3b0ede0ad67a7c5f8ebd98aaa1d4cfdb7`
- Protocol SHA-256: `TBD`
