# Fair-warmup staged hybrid protocol (best-checkpoint stage transitions)

Status: declared 2026-08-27, before any training run with
`training.stage_init_from_best: true`.

## Why this protocol exists

Every end-to-end staged result reported so far was produced by a schedule in
which each stage resumes from the preceding stage's **terminal** weights, while
validation selection at the end of training compares each stage's **best**
checkpoint. A quantum stage therefore started from a slightly degraded
classical predictor and was then scored against the classical high-water mark.

The published analysis remains valid as a statement about that schedule, and it
is reported as such. It cannot answer whether quantum stages would survive
selection when initialised from the best classical checkpoint under a matched
budget. This protocol declares that experiment before it is run.

## Prior outcome knowledge

This declaration is not outcome-blind, and the following was known when it was
written. All five main E3 folds selected Stage A. The Stage-B best checkpoint
fell below the Stage-A best by 0.002 to 0.010 validation balanced accuracy, and
Stage A's own terminal checkpoint fell below its best by 0.000 to 0.008. Scored
against Stage A's terminal checkpoint instead of its best, Stage B was ahead in
three of the five folds (folds 1, 3, and 4). Stage C reached exactly 0.500
validation balanced accuracy in all five folds. In the eighteen width cells,
validation preferred a quantum-active route in four and none of those four beat
its same-cell Stage-A reference on test AUPRC.

The direction of that knowledge is explicitly unfavourable to the null the
current manuscript reports: it is the reason this rerun is worth doing, and it
means a positive result here must be treated as confirmatory of a pre-declared
hypothesis rather than discovered post hoc.

## Hypothesis

H1: with best-checkpoint stage transitions and a matched validation budget,
validation selects a quantum-active stage (Stage B or Stage C) in at least one
of the five folds.

H0: Stage A is selected in all five folds, as in the published schedule.

This is deliberately a weak, falsifiable hypothesis about **selection**, not
about test performance. Test performance is a secondary endpoint because
selection is the mechanism the published claim rests on.

## What changes, and what must not

Exactly one behaviour changes: `training.stage_init_from_best` is set to `true`,
so each stage begins from the best validation checkpoint of the nearest
preceding stage that produced one. The implementation is
`HybridTrainer._restore_preceding_stage_best`, is off by default, and records
every restore in `stage_transitions` in the run summary.

Everything else is held at the values used by the published five-fold runs:
dataset archive, split manifest (`data/splits/histopath_kaggle/`), backbone,
VQC width of 8 qubits with linear entanglement, focal loss with gamma 2, strong
augmentation, test-time augmentation over six views, Fbeta threshold tuning with
beta 1.5, batch size 64, AdamW with weight decay 1e-4, gradient clipping at
norm 1.0, learning rates 1e-4 for stages A and B and 1e-5 for stage C, and
seed 42 + fold.

Deviating from any of these makes the rerun non-comparable to the published
result and voids this protocol.

## Amendment 2026-09-04: execution environment and control arm

Declared before any run with `stage_init_from_best: true`.

This protocol originally assumed Kaggle T4 x2, the environment that produced
the published five-fold results. The experiment will instead run on the RTX
A4000 server, which has so far only executed the width matrix. A server
fair-warmup run compared against Kaggle-published numbers would confound the
checkpoint-transition change with the change of environment, which the
server width protocol already refused for the width cells.

The design is therefore paired and within-environment, and the arm labels below
supersede the single-arm framing:

- **Arm A0 (control).** `--no-stage-init-from-best` on the server. Reproduces
  the published schedule in the new environment. It is the paired baseline for
  A1 and, separately, an environment-replication observation against the
  published Kaggle results.
- **Arm A1 (treatment).** `--stage-init-from-best` on the server, identical in
  every other respect.

Only A1 versus A0 supports the hypothesis test. A0 versus the published Kaggle
runs is reported as a replication observation and is never pooled with either
arm. If A0 fails to reproduce the published selection outcome, that failure is
reported and the fair-warmup comparison is interpreted only within the server
environment.

Measured cost, from `train_time_s` in archived artifacts: 0.99-1.31 h per
E3-equivalent fold on this server (four server q8 cells), against 3.11-5.16 h
per fold on Kaggle. Ten cells therefore cost roughly 12 GPU-hours.

Tooling and the full plan live in `PaperB_PathA/`.

## Budget matching

The published runs gave Stage A 9 to 18 validation evaluations and Stage B 5.
Two arms are declared, in this order:

1. **Primary (schedule-matched).** Stage caps unchanged at 25 / 15 / 3 with
   early-stopping patience 5. This isolates the checkpoint-transition change as
   the single difference from the published runs.
2. **Secondary (budget-matched).** Early stopping disabled for stages A and B
   and both capped at 15 epochs, so each stage receives an equal number of
   validation evaluations. This addresses the unequal-looks asymmetry.

Arm 1 is the confirmatory analysis. Arm 2 is reported separately and never
pooled with arm 1.

## Selection rule

Selection remains validation balanced accuracy at the fixed 0.5 cutoff, chosen
before threshold tuning, because changing the selection metric at the same time
as the transition rule would confound the two. A pre-declared secondary
reporting of selection under validation AUPRC is permitted, labelled as such,
and may not be substituted for the primary rule.

## Analysis plan

Primary endpoint: the number of folds, out of five, in which validation selects
a quantum-active stage. Reported as a count with the per-fold selected stage and
margin. No test statistic is applied to a count of five.

Secondary endpoints: per-fold test balanced accuracy, F1, AUPRC, recall, and
precision of the selected checkpoint, compared with the published E2, E2b, and
E3 values on the same folds; and the per-fold Stage-B-minus-Stage-A validation
margin, to be compared with the 0.002 to 0.010 range observed under the
published schedule.

Fold-level pooling, where used, follows the existing corrected resampled t
interval with n_test/n_train evaluated on the full fold splits, as in the
published Stage-B analysis.

## Stopping and disclosure rules

All five folds of arm 1 must be run and reported, including folds whose result
is unfavourable to H1. No fold may be dropped for a numerical failure without
reporting the failure and its cause. If a run fails to complete, it is rerun
once with an identical configuration and both attempts are recorded.

If arm 1 selects Stage A in all five folds, the published conclusion is
strengthened and this protocol's result is reported as a negative confirmation.
If arm 1 selects a quantum stage in any fold, the published selection claim is
conditional on the terminal-weight schedule and the manuscript must say so; the
selected fold's test metrics must then be reported whether or not they favour
the quantum route.

## Execution

Five sequential runs, one per fold, on Kaggle T4 x2 under the 12-hour session
cap. Budget 3 to 5 hours per fold on the two-GPU configuration and 8 to 12
hours on a single T4, per the runbook. Arm 2 costs more because early stopping
is disabled.

Artifacts land in `results/histopath_kaggle_fold{FOLD}_e3_fairwarmup/` and must
include the run summary containing `stage_init_from_best` and
`stage_transitions`, the history, the progress record, and the stage-comparison
report.
