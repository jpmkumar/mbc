# Paper C — locked redesign and execution plan

**Working title:** *How grouping protocol and field of view shape pathology
foundation-model evaluation in breast histopathology*

**Thesis:** evaluation grouping and physical field of view can alter both
measured performance and pathology-foundation-model rankings. These effects must
be isolated before a small-patch benchmark is interpreted as evidence of model
transfer.

**Quantum:** none. Paper C is entirely classical.

**Primary cohorts:**

- IDC: 277,524 50×50 patches under 279 public case identifiers;
- BCSS: 151 breast-cancer slides/ROIs with patient/site identifiers and
  pixel-level tissue masks.

**Compute:** the qualified campus RTX A4000 server. Foundation-model embeddings
are extracted once from revision-pinned weights and cached; probes and
case-cluster analyses operate on the caches.

---

## Phase 0 — protocol freeze

1. **Literature refresh — complete.** The June 2026 FM benchmark on this exact
   archive and the 2025 grouped cross-site IDC study are now included. “First FM
   evaluation” and “first grouped IDC benchmark” are retired.
2. **Archive audit — complete.** Counts and filename structure are internally
   consistent, but patient identity is not recoverable. Use “case identifier”
   and “case-ID grouped,” never unqualified “279 patients.”
3. Define and freeze:
   - IDC random-patch and case-ID-grouped five-fold manifests;
   - inner train/validation/calibration partitions;
   - BCSS patient/site-held-out train/validation/calibration/test assignment;
   - two co-primary contrasts and Holm correction;
   - exact model revisions, transforms and probe search spaces.
4. Run coordinate/context eligibility audits before selecting the common centre
   set for `K=1,3,5,9`.
5. File the preregistration before any labelled comparative run.

**Gate:** no full embedding or labelled probe run until the preregistration and
runtime qualification checks pass. Small model-loading and synthetic engineering
tests are permitted.

## Phase 1 — qualified frozen embeddings

Primary panel:

1. UNI2-h;
2. Virchow2;
3. Phikon-v2;
4. DINOv2 ViT-L/14 generic self-supervised control;
5. ImageNet ResNet-50 conventional control.

Requirements:

- Hugging Face snapshot revisions and local weight/config hashes are immutable;
- vendor transforms and pooling rules are verified against model-card examples;
- Virchow2 register tokens are excluded from patch-token pooling;
- per-model A4000 batch sizes pass a real-image smoke test;
- float16 caches agree with float32 references within a declared tolerance;
- `filepath` is the only join key between caches, labels and split manifests.

Extract IDC `K=1` first. Context caches follow only after coordinate completeness
is audited. BCSS extraction follows the same revision-pinned pipeline.

## Phase 2 — primary A: protocol optimism

For each encoder at `K=1`:

- produce one out-of-fold prediction per IDC patch under random-patch five-fold
  cross-validation;
- produce one out-of-fold prediction per IDC patch under the frozen
  case-ID-grouped five-fold cross-validation;
- use the same deterministic linear-probe class, preprocessing and declared
  nested tuning budget in both protocols.

**Primary estimand:** random-minus-grouped case-balanced AUPRC for the
preregistered reference encoder.

**Ordered secondaries:** encoder-specific gaps, unweighted AUPRC, case-balanced
AUROC, MCC, sensitivity/specificity at a validation-locked threshold, Brier
score, and bootstrap uncertainty in model-rank stability.

## Phase 3 — primary B: physical field of view

Under case-ID-grouped evaluation:

- primary contrast: `K=9 − K=1` case-balanced AUPRC for the reference encoder;
- intermediate descriptive levels: `K=3` and `K=5`;
- primary population: centres with complete `K=9` neighbourhoods, identical
  across every context condition;
- sensitivity population: all eligible centres with padding metadata included.

The `K=9` window approximately matches the physical field of a 224-pixel 20×
foundation-model tile when the nominal 40× IDC scale is interpreted as
0.25 µm/pixel. This physical assumption is stated and tested on BCSS, where
resolution metadata are available.

## Phase 4 — BCSS external replication

- Derive tumour-versus-other centre labels from native masks using a
  preregistered purity rule; ignore outside-ROI/exclude pixels.
- Keep patients and institutions disjoint across train, calibration and test.
- Repeat the `K=9 − K=1` representation experiment.
- Treat the result as external replication of the field-of-view mechanism, not
  identical-label validation of an IDC classifier.

Optional secondary: ROI-level predicted tumour-area agreement against masks.
This is not called patient burden or clinical utility.

## Phase 5 — reliability and sensitivity analyses

On grouped predictions only:

- case-balanced temperature scaling on the calibration partition;
- calibration intercept/slope, Brier score, log loss and reliability diagrams;
- empirical risk–coverage curves without formal conformal guarantees;
- limited last-block fine-tuning only for predeclared encoders if the primary
  frozen-probe matrix is complete;
- descriptive latency, peak memory and cache size.

No decision-curve analysis, clinical workload claim, demographic fairness
comparison, carbon/distillation contribution, or broad architecture zoo enters
the primary paper.

## Phase 6 — inference and manuscript

- Average stochastic-seed predictions before statistical inference.
- Pool paired out-of-fold predictions and bootstrap whole case identifiers.
- Use clustered AUROC and paired case-cluster intervals; do not use ordinary
  patch-level DeLong tests or treat folds/seeds as independent sample size.
- Apply Holm correction to the two co-primary tests.
- Complete TRIPOD+AI and CLAIM 2024 checklists.
- Disclose the shared IDC cohort and split artifacts with Paper B; reuse no
  Paper B result values and include no QML discussion.

**Venue path:** *Medical Image Analysis* if the protocol effect and BCSS context
replication are both clear; otherwise *IEEE JBHI*, *Artificial Intelligence in
Medicine*, or *Computers in Biology and Medicine*.

---

## Falsification and interpretation

- A small protocol gap rejects a strong “published performance is inflated”
  narrative; it remains a valid estimate and the paper centres on field of view.
- No `K=9` gain rejects the context-starvation mechanism for the tested
  encoders; report the null without adding post-hoc context variants.
- Failure to replicate the context effect on BCSS limits the result to the IDC
  derivative and rules out a general mechanism claim.
- Rank stability across protocols rejects the rank-reversal hypothesis while
  retaining the absolute optimism estimate.

No outcome changes the prespecified model panel, centre population, primary
metrics or correction family.
