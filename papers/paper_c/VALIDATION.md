# Paper C — validation and inference protocol

This study is about evaluation validity. The grouping unit, calibration unit,
estimand and uncertainty calculation must therefore be explicit before labelled
comparisons begin.

## 1. Units and terminology

### IDC

The archive exposes 279 internally consistent case identifiers, not a verified
patient field. All patches sharing an identifier remain in one partition.
Manuscript terminology is **case-ID grouped**. The archive cannot exclude shared
patients across identifiers, so the grouped result is the strongest independence
the public derivative permits—not proof of patient independence.

### BCSS

The external cohort exposes TCGA patient identifiers and source institutions.
Patients are disjoint across partitions; institutions selected for testing are
not used for probe fitting or calibration.

## 2. Partitions

### IDC case-ID-grouped arm

The existing frozen outer five-fold manifests under
`data/splits/histopath_kaggle/` define test case identifiers. Never use
`data/splits/histopath/`, which belongs to Paper B's separate server width
matrix.

Inside each outer-training set, freeze three disjoint identifier sets:

- `inner-train`: linear-probe fitting;
- `inner-val`: hyperparameter and operating-threshold selection;
- `inner-cal`: temperature scaling only;
- `outer-test`: scored once.

The split generator balances identifier-level IDC-positive patch ratio without
using outer-test outcomes for any decision.

### IDC random-patch arm

Construct five patch folds so every patch receives exactly one out-of-fold
prediction. Each fold receives its own inner train/validation/calibration patch
partitions. Use the same probe family, search space and computational budget as
the grouped arm.

Random-patch evaluation intentionally permits different patches from the same
case identifier across partitions. It is an estimate of protocol contamination,
not a deployment-valid model. Mosaic inputs are forbidden in this arm because a
neighbouring tile can create an additional direct leakage channel.

### BCSS

Use a frozen institution-held-out test set. Within the remaining institutions,
partition by patient into train, validation and calibration. The exact institution
lists, patient counts and class composition are fixed in preregistration.

## 3. Comparable out-of-fold predictions

For IDC, both protocols ultimately produce one prediction for each of the same
277,524 patches. This gives paired predictions despite different fold
assignments. All cache/manifest joins use the unique relative `filepath`; row
order is never assumed.

For context comparisons, use a common centre population:

- primary: centres with a complete `K=9` neighbourhood and valid centre label;
- sensitivity: all centres, with neighbour completeness and padding fraction
  recorded.

Every `K` condition predicts the same centres. This prevents context effects from
being confounded by sample inclusion.

## 4. Probe fitting and fairness between conditions

Primary adaptation is a frozen encoder plus a deterministic linear probe. Each
condition receives:

- the same feature standardisation policy;
- the same regularisation grid;
- the same class/case weighting rule;
- the same inner-validation objective;
- the same number of fitting attempts.

For the co-primary contrast both arms additionally share one weighting regime:
each case identifier carries equal total weight when fitting the probe, scoring
the validation objective, fitting temperature and selecting the threshold. The
arms then differ only in how patches reach folds and inner partitions, so the
contrast is attributable to grouping.

A second, ordered-secondary contrast restores the conventional random-patch
recipe (class-balanced fitting weights, unweighted validation AUPRC, patch-equal
calibration). That version estimates the gap against random-patch practice as
published, and must not be described as isolating grouping.

Model-specific vendor transforms are retained because they are part of the
pretrained representation; no downstream statistics are re-estimated from
outer-test data.

Last-block fine-tuning is secondary and uses a separately preregistered,
architecture-aware budget. It cannot replace or redefine the frozen-probe
primary analysis.

## 5. Primary estimands

Two co-primary contrasts are fixed:

1. **Protocol optimism:** random-patch minus case-ID-grouped case-balanced AUPRC
   for the reference encoder at `K=1`, both arms under the harmonised weighting
   regime.
2. **Context gain:** `K=9` minus `K=1` case-ID-grouped case-balanced AUPRC for
   the same encoder on complete-neighbourhood centres.

Each case identifier receives equal total weight in case-balanced AUPRC,
irrespective of its number of patches. Holm adjustment controls the family-wise
error of the two co-primary tests.

Ordered secondary outcomes:

1. encoder-specific protocol and context effects;
2. model-rank stability;
3. unweighted AUPRC and clustered AUROC;
4. MCC, sensitivity and specificity at an inner-validation-locked threshold;
5. Brier score, log loss and calibration intercept/slope;
6. BCSS context replication;
7. empirical risk–coverage and ROI tumour-area agreement.

## 6. Inference

- Average predictions across stochastic seeds before sampling-based inference.
- Resample whole IDC case identifiers for paired confidence intervals.
- For BCSS, resample patients within institutions; report institution-specific
  results descriptively.
- Use at least 2,000 bootstrap replicates for development checks and 10,000 for
  locked manuscript intervals.
- Compute paired differences inside every replicate.
- Use percentile intervals. Compute co-primary two-sided p-values from the
  bootstrap distribution centred at the observed contrast, with the
  `(extreme + 1)/(B + 1)` finite-replicate correction.
- Do not use patch-level confidence intervals, ordinary DeLong tests, folds as
  `n=5`, or seeds as independent observations.
- Report point estimate, 95% interval and exact bootstrap procedure for every
  primary and ordered-secondary contrast.

A simulation-based precision analysis using observed identifier sizes and class
ratios determines whether extra optimization seeds materially narrow uncertainty.

## 7. Calibration and selective prediction

Temperature scaling is fit on `inner-cal` with each case receiving equal total
weight. It never touches validation or test data. Report:

- calibration intercept and slope;
- case-balanced and unweighted Brier score/log loss;
- reliability plots with case-cluster uncertainty;
- ECE only as a bin-dependent descriptive statistic.

The primary manuscript does not claim conformal coverage. Pooled patches are not
exchangeable deployment units, while merely grouping calibration patches does
not define a patient-level conformal target. Empirical risk–coverage curves are
allowed and are labelled descriptive. Group/hierarchical conformal methods
require a separate methods review and amendment before use.

## 8. BCSS label construction

BCSS masks contain tumour, stroma and other tissue classes plus outside-ROI and
exclude labels. Before extraction, preregister:

- physical resolution and centre-grid spacing;
- tumour-versus-other mapping;
- minimum valid-mask fraction;
- centre-label purity threshold;
- handling of DCIS, outside-ROI and excluded pixels;
- patient and institution partition lists.

IDC and BCSS targets are related but not identical. BCSS evaluates replication
of the field-of-view mechanism, not transport calibration of the IDC classifier.

## 9. Locked-test discipline

- The preregistration is committed before labelled comparative runs.
- Engineering smoke tests may verify loading, dimensions, transforms and cache
  integrity but may not report comparative label-based performance.
- Evaluators refuse to overwrite completed out-of-fold prediction bundles.
- Every bundle records source commit, container ID, dependency lock, model
  revision, weight/config hashes, dataset hash, split hash and index hash.
- Analyses added after primary results are visible are explicitly labelled
  exploratory and cannot replace the co-primary contrasts.

## 10. Claims the design cannot support

Do not infer:

- verified patient-level independence in IDC;
- patient diagnosis, prognosis or treatment benefit;
- clinical net benefit or laboratory workload reduction;
- demographic fairness;
- equivalent-task external validation;
- formal patient-level conformal coverage.

The study supports claims about evaluation protocol, representation transfer,
physical field of view, case-grouped internal performance and external
mechanistic replication.
