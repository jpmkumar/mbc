# Paper D protocol — BreakHis external histopath generalisation

Status: **LOCKED as immediate follow-on after Paper B submit** — declared
2026-09-16, before any BreakHis training run, split generation, or labelled
comparative probe.

Engineering that downloads the archive, audits case identifiers, or builds
unlabelled caches is permitted. Comparative performance inspection is not.

Governing workspace (Mac + GitHub): `papers/paper_d/`.
Manuscript authoring stays local under `papers/paper_d/manuscript/` (gitignored).
Research code, config, lock scripts, and this protocol are tracked on
https://github.com/jpmkumar/mbc. Code under that workspace must not alter the
estimands, arms, or decision rules below without a dated amendment.

## Why this protocol exists

Paper B asks whether a bolt-on hybrid VQC head helps on one public IDC patch
cohort under case-ID-grouped CV. The answer is a null: validation does not
select the quantum stage, and the VQC does not beat its capacity-matched
classical control. The Scientific Reports manuscript's first load-bearing
limitation is that this is a **single cohort**. Paper D closes that limitation
without reopening the Kaggle IDC bake-off.

The question is domain shift, not within-cohort capacity:

> Does a Paper-B–style hybrid VQC head transfer across magnification and
> staining on BreakHis under case-level splits, or does the null stay?

"No gain under shift" is a **pre-allowed outcome**. A negative result strengthens
the Paper B null across cohorts; it is not a failed experiment.

## Prior outcome knowledge

This declaration is not outcome-blind about Paper B. Known at drafting time:

- E2 / E2b / E3 five-fold: VQC not selected; matched MLP (E2b) competitive or
  better on the same compressed embedding.
- Width ablation (4/8/12): no material AUPRC gain from circuit width.
- Path A fair-warmup: declared separately; its outcome does not change Paper D's
  estimand (BreakHis transfer), only whether B's schedule caveat is lifted.

No BreakHis scores, folds, or head comparisons have been inspected for this
paper. Any informal BreakHis browsing before this lock voids confirmatory
status for the affected arm and must be disclosed as exploratory.

## Hypothesis

Primary contrast — on BreakHis case-ID folds, within each adaptation regime:

- **H1:** the VQC head exceeds the E2b-matched MLP on held-out case-balanced
  AUPRC by more than the smallest effect of interest (0.01).
- **H0 (allowed):** the VQC does not exceed the matched MLP by ≥0.01 AUPRC;
  intervals support practical equivalence or favour the classical head.

H0 is the expected continuation of Paper B. Rejecting H0 under shift would be
the surprising result and must survive the pre-declared inference below.

## Cohort and units

- **Dataset:** BreakHis (Spanhol et al.), all four magnifications
  (40×, 100×, 200×, 400×) retained as strata, not pooled without a stratum
  label.
- **Task:** binary benign vs malignant at the image/patch unit used after the
  pre-outcome audit; subtype labels are secondary only.
- **Grouping unit:** patient / case identifier. No image from a case may appear
  in more than one outer split. Image-level random CV is forbidden for the
  primary analysis.
- **Optional source arm:** Paper B's frozen IDC folds remain a *source*
  domain only. Their test labels are never used to tune BreakHis heads.
  Cross-domain evaluation (train IDC → test BreakHis) is an ordered secondary,
  not co-primary.

Split manifests must be committed under `data/splits/breakhis/` with a
content digest recorded in the lock manifest before the first labelled run.

## Arms (what changes, what must not)

Two classical adaptation regimes × two heads. All four cells are required.

| ID | Backbone | Head |
|----|----------|------|
| F-MLP | Frozen EfficientNet-B0 + Paper-B compression to 8-D | E2b-matched MLP |
| F-VQC | Same frozen backbone and compression | Paper-B VQC (8q, 2 layers, angle-Y, linear entanglement, full readout) |
| T-MLP | Last-block fine-tune of the same backbone (matched LR/budget) | E2b-matched MLP |
| T-VQC | Same fine-tune schedule | Same VQC |

**Must not change** relative to Paper B's circuit family: qubit count 8,
layers 2, angle-Y encoding, linear CNOT entanglement, no data re-uploading,
`default.qubit` + backprop unless a pre-declared simulator swap is amended
before any run. Deviating from the circuit family voids the "same family as B"
claim.

**Must match across heads within a regime:** train/val/test case lists, seed
(`42 + fold`), batch size, loss, augmentation policy, epoch budget, early
stopping, and threshold rule. The only intentional difference between MLP and
VQC cells is the head.

Primary metric: **case-balanced AUPRC** on the held-out case fold, with the
operating threshold (if used for secondary discrete metrics) locked on
validation only.

## Smallest effect of interest

**0.01 case-balanced AUPRC**, matching Paper C's declared scale. An interval
lying entirely inside ±0.01 for VQC − MLP supports "no effect of interest under
shift." An interval that includes zero but extends beyond ±0.01 is inconclusive
and must be reported as such — not as evidence for H1.

## Analysis plan

1. **Co-primary:** paired fold-wise Δ = AUPRC(VQC) − AUPRC(MLP) under the
   **frozen** regime (F-VQC − F-MLP), aggregated with the same corrected
   resampled procedure used in Paper B Stage-B, treating cases as the
   dependence unit.
2. **Co-primary partner:** the same Δ under the **fine-tune** regime
   (T-VQC − T-MLP). Holm correction across the two regime tests at family-wise
   0.05.
3. **Ordered secondary:** per-magnification strata; IDC→BreakHis transfer with
   frozen Paper-B weights; ROC-AUC; balanced accuracy / F1 at val-locked
   thresholds; calibration. Secondaries are descriptive and uncorrected.
4. **Falsification:** if both co-primary intervals exclude a material VQC gain,
   report that the Paper B null **persists under magnification/stain shift**.
   Do not reframe as a methods failure. If one regime shows a material gain and
   the other does not, report the interaction; do not pool regimes post hoc to
   manufacture a win.

## Stopping and disclosure

All committed folds of all four cells must be run and reported, including
cells unfavourable to H1. A failed run is retried once with identical config;
both attempts are recorded. Completed finite runs are not repeated because
results are inconvenient.

No magnification may be dropped after seeing scores. No head hyperparameter
may be retuned after any BreakHis test fold is scored. Exploratory pilots, if
any, must use a disjoint case subset declared in an amendment *before* the
confirmatory split is frozen.

## Lock procedure

1. Set this file's status to LOCKED (already done above for series priority).
2. Commit this file; create `papers/paper_d/` README pointing here.
3. Before first labelled run: write `preregistration/paper_d_lock.json` with
   source commit, protocol SHA-256, BreakHis archive digest, split-manifest
   digests, dependency-lock digest, and container image digest.
4. Verify every digest from a clean checkout; mismatch voids the lock.

## Series priority (locked)

Immediate follow-on after Paper B submission: **Paper D (BreakHis)**.
Paper C remains the classical evaluation/FOV track and is not displaced, but
D is the highest-value QML follow-on because it reuses B's circuit family on a
new estimand (domain shift) and answers the single-cohort limitation of the
Paper B study directly.

## Amendment — T&F Connection Science secondaries (2026-09-22)

**Status:** LOCKED amendment. Does **not** reopen the eight-qubit circuit
family, co-primary estimands, SEI (0.01), Holm rule, or the four training
arms. Venue primary for Paper D is now **Connection Science** (Taylor &
Francis); see `papers/paper_d/VENUE.md` and `papers/paper_d/APC_GATE.md`.

Ordered T&F secondaries (descriptive, uncorrected), run from frozen
checkpoints / saved `test_predictions.npz` where possible:

| ID | Experiment | Arms | Notes |
|----|------------|------|-------|
| **S1** | Per-magnification case-balanced AUPRC (40× / 100× / 200× / 400×) | All four | Align predictions to runtime `test.csv` row order |
| **S2** | Calibration: ECE (15 equal-width bins) + reliability diagrams | **F-MLP**, **F-VQC** | Pooled held-out folds; report mean ECE ± fold std |
| **S3** | Wall-clock (`train_time_s` from summaries) and trainable / total parameter counts | All four | Param counts from a single CPU instantiate of each head |
| **S4** | IDC→BreakHis transfer (frozen Paper-B weights → BreakHis test) | Optional F-MLP vs F-VQC | Skip if compute blocks submit; not required for Connection Science filing |

Export path: `papers/paper_d/scripts/export_tf_secondaries.py` →
`papers/paper_d/results/tf_secondaries.json` and manuscript figures
`fig03_per_magnification.*`, `fig04_reliability.*`.

S4 remains optional. Do not invent IDC→BreakHis numbers without a declared
run under this amendment.
