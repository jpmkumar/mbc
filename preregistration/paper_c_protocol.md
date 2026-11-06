# Paper C protocol — grouping and field-of-view effects in pathology foundation models

**Status:** LOCKED — outcome-locked; labelled comparative probes may proceed
only after Stage-2/3 lock verify.  
**Created:** 2026-08-21.  
**Amended:** 2026-09-01; 2026-11-06 (ungated panel + BCSS stride 150).  
**Locked:** 2026-11-06.  
**Lock condition:** status is `LOCKED`; complete Stage 2–3 in section 11 before
any labelled comparative probe is run. Label-blind embedding extraction is
permitted once Stage 1 is committed.

Engineering tests that load models, inspect dimensions, assemble synthetic
mosaics or extract a small unlabeled cache are permitted while this document is
draft. Comparative performance inspection is not.

The 2026-09-01 amendment, made before any labelled comparison, harmonises the
weighting regime across both arms of the co-primary protocol contrast so that it
isolates grouping, demotes the conventional bundled regime to an ordered
secondary, declares a smallest effect size of interest, and replaces the
self-referential lock record with an external lock manifest.

The 2026-11-06 amendment, also before any labelled comparison, retires gated
Hugging Face pathology encoders that the executing institution cannot lawfully
obtain, and retargets both co-primary contrasts onto the openly downloadable
DINOv2 ViT-L/14 reference encoder.

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

Reference encoder for both co-primary contrasts: **DINOv2 ViT-L/14**.

### Confirmatory panel (executable without gated Hugging Face access)

1. DINOv2 ViT-L/14 — reference encoder (`torch.hub` / Facebook Research release);
2. ImageNet ResNet-50 — conventional control (`torchvision` weights CDN).

### Declared unavailable before labelled comparisons

1. UNI2-h — gated CC-BY-NC-ND Hugging Face weights; institutional account
   approval cannot be completed at the executing site;
2. Virchow2 — same institutional gated-access barrier;
3. Phikon-v2 — Hugging Face / Owkin access not obtainable under the same site
   policy.

These three remain listed in `papers/paper_c/config/model_registry.json` with
`status: unavailable` and an explicit reason. They are **not** replaced by other
gated models after outcomes are seen. The manuscript must state that the study
quantifies grouping and field-of-view effects for **openly redistributable**
frozen encoders, not for the gated pathology-foundation-model panel used in
some prior IDC benchmarks.

Primary adaptation remains frozen embeddings plus a deterministic linear probe.
Exact revisions, vendor transforms, output pooling, licences and artifact
SHA-256 values must be recorded in the registry before protocol lock.

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
- Fit temperature scaling on calibration data only.
- Select the binary operating threshold on validation data only by maximum MCC;
  ties choose the higher threshold.
- Where two `alpha` values tie on the validation objective, select the smaller
  value.

### Weighting regimes

The confirmatory contrast must isolate grouping, so the co-primary analysis runs
both arms under one identical weighting regime.

- **Harmonised (`--weighting case-balanced`, co-primary).** Both arms give each
  case identifier equal total weight when fitting the probe, when scoring the
  validation objective, when fitting temperature and when selecting the
  operating threshold. The only difference between the arms is how patches are
  assigned to folds and inner partitions.
- **Bundled (`--weighting protocol-native`, ordered secondary).** The grouped
  arm is unchanged; the random arm reverts to the conventional random-patch
  recipe of class-balanced fitting weights, unweighted validation AUPRC and
  patch-equal calibration. This estimates the difference a reader would observe
  between a conventionally executed random-patch study and a grouped one.

The harmonised contrast supports statements about grouping. The bundled contrast
supports statements about published random-patch practice as a whole, and must
never be described as isolating grouping.

The search grid, fitting tolerance and weighting regime are identical across
models within a contrast.

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

For BCSS, sample a fixed centre grid from native 0.25 µm/pixel images with
**stride 150 pixels** (centre size 50, context window 450). Stride 50 was
rejected before labelled comparisons because it produced highly redundant
centres (~6× denser than needed for overnight A4000 extraction). Ignore mask
classes `outside_roi`, `exclude` and `undetermined`. Class `tumor` is
positive; all other valid tissue classes, including DCIS, are negative. Require
at least 80% valid pixels and 80% class purity in the centre 50×50 region. The
frozen centre-manifest SHA-256 is recorded in the external lock after rebuild.

## 6. Co-primary estimands

1. `Δ_protocol`: IDC random-patch minus case-ID-grouped case-balanced AUPRC for
   DINOv2 ViT-L/14 at `K=1` under the **harmonised** weighting regime, using
   paired out-of-fold predictions for all patches.
2. `Δ_context`: IDC grouped `K=9` minus `K=1` case-balanced AUPRC for DINOv2
   ViT-L/14 on complete-neighbourhood centres.

Holm's procedure controls family-wise error at 0.05 across these two tests.
Two-sided 95% intervals are reported.

### Smallest effect size of interest

A difference of **0.01 case-balanced AUPRC** is the preregistered smallest
effect of interest for both co-primary contrasts. It is the scale against which
the pre-outcome simulation in `papers/paper_c/results/precision_simulation.md`
judged seed-averaging noise to be negligible, so effects below it cannot be
separated from optimisation noise under this design.

Interpretation is driven by the estimate and interval, not by a significance
label alone. A confidence interval lying entirely inside ±0.01 supports the
statement that no effect of interest was detected. An interval that merely
includes zero while extending beyond ±0.01 is inconclusive and must be reported
as such.

## 7. Ordered secondary analyses

1. The bundled-regime protocol contrast for DINOv2 ViT-L/14, reported alongside
   the harmonised co-primary so the contribution of conventional random-patch
   tuning practice is visible.
2. Protocol and context contrasts for ResNet-50.
3. Model-rank changes across protocols and contexts for the two confirmatory
   encoders, with rank uncertainty computed inside synchronised whole-case
   bootstrap replicates.
4. Unweighted AUPRC and case-balanced/clustered AUROC.
5. MCC, sensitivity and specificity at validation-locked thresholds.
6. Calibration intercept/slope, Brier score, log loss and reliability plots.
7. BCSS `K=9 − K=1` context replication for DINOv2 ViT-L/14, then ResNet-50.
8. Empirical risk–coverage curves.
9. BCSS ROI-level tumour-area agreement.
10. A separately declared last-block fine-tuning sensitivity analysis, only if
    the full frozen-probe primary matrix is complete.

Secondary analyses are not corrected as a family; they are ordered and reported
descriptively. An analysis in this list that has no committed, tested
implementation at the moment of lock is demoted to exploratory and is labelled
as such in the manuscript.

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
  context-starvation mechanism for DINOv2 ViT-L/14 under this design.
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

## 11. Protocol-lock procedure

A document cannot contain its own digest, the commit that introduces it, or the
digest of an image built from that commit. The lock is therefore recorded in a
separate artifact created after this file is frozen.

**Stage 1 — freeze the protocol.** Set the status line to `LOCKED`, commit this
file together with the code and split artifacts it references, and change
nothing afterwards.

**Stage 2 — write the external lock manifest.** From that commit, build the
container image, then write `preregistration/paper_c_lock.json` recording:

- the Stage-1 source commit;
- the SHA-256 of this protocol file at that commit;
- the container image digest and the dependency-lock SHA-256;
- the IDC dataset archive SHA-256;
- the SHA-256 of every split, eligibility and manifest artifact actually read by
  the run, each identified by repository-relative path;
- the BCSS mirror tree hash, patient-split and centre-manifest SHA-256;
- the model-registry SHA-256 and each encoder's resolved revision and weight
  digests.

Every digest is taken over the exact bytes as committed. Artifacts generated on
the server rather than committed are recorded by absolute path and digest in the
same manifest, and each run's own provenance file must reproduce those digests
or abort.

**Stage 3 — verify.** Recompute every digest in the manifest from a clean
checkout of the Stage-1 commit before the first labelled run. Any mismatch
voids the lock and requires a new Stage 1.

Artifacts known at drafting time, to be re-verified rather than trusted:

| Artifact | Path | SHA-256 at drafting |
|---|---|---|
| IDC inner split summary | `data/splits/paper_c/idc/inner_split_summary.json` | `46e8f4494422ac201a53ee5a9c8438fcd11c9a128ba87ebe12c6aa721e14de00` |
| Model registry | `papers/paper_c/config/model_registry.json` | `f704a084813fea90a07730c58708578f3e5e6e13d3011a3877e4f2f660b0aaa7` |
| IDC coordinate/eligibility index | server-generated | `e970ae0b03b4c1f9fbbd8bebcbe6082b309c610a0844f4c55cfd45a4df20f1b7` |
| IDC filepath-keyed protocol manifest | server-generated | `b28e4acc2c3482256c17971cd422e39713d5ca4df860bc2929a31b3104caa266` |
| BCSS patient splits | server-generated | `8d884bd3c8b9c2f3aa6bb28a944ac282682ec57e2f958689232f42b88ea854ca` |
| BCSS centres (stride 150) | server-generated | `2e47a2cc874966703228c3b5bed2c5621b785ef7e2f9efb22b0295cda9ec25a8` |

The IDC outer split manifest digest is inherited from the shared histopathology
release and must be re-stated in the lock manifest against an explicit path,
because the previously recorded value did not identify which file it covered.
