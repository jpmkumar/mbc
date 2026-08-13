# End-to-end VQC width ablation protocol

Declared on **2026-08-13 before folds 1–4 were run at 4 or 12 qubits**.
This is an outcome-informed secondary experiment, not a new confirmatory
primary analysis.

## Prior outcome knowledge

Fold 0 was already run at 4, 8 and 12 qubits before this declaration. Its
test AUPRC values were 0.892, 0.891 and 0.893, respectively, and every width
selected Stage A as its best validation stage. Those results motivated the
multi-fold replication and make Fold 0 exploratory. They are retained for
provenance and displayed in the five-fold descriptive summary, but all formal
replication statements are repeated over the untouched folds 1–4.

The completed frozen-head experiment also showed that the 8-qubit VQC and
matched MLP are practically equivalent after fair per-fold tuning. The width
ablation therefore asks whether changing the end-to-end representation and
circuit width alters that conclusion; it cannot independently confirm the
already observed null result.

## Scientific estimand

This is an **end-to-end system-width ablation**, not an isolated gate-count
experiment. For width `q`, the learned compression bottleneck is changed to
`q` dimensions and the VQC receives `q` angles. Consequently, a difference
between widths may arise from the learned classical compression, the quantum
circuit, or their interaction. It must not be described as a pure causal
effect of adding qubits.

The candidate widths are fixed to 4, 8 and 12 qubits. The 8-qubit E3 runs
already completed on all five folds are the reference. New computation is
limited to 4 and 12 qubits on folds 1–4; the existing Fold-0 width runs are
not repeated merely to erase their exploratory status.

## Fixed training protocol

- Dataset: 277,524 IDC patches from 279 patients.
- Splitting: the existing deterministic five-fold patient-level manifests;
  no patient may overlap train, validation or test within a fold.
- Experiment: E3 staged hybrid training.
- Seed: 42, with the existing fold-specific effective seed `42 + fold`.
- Widths: 4, 8 and 12; compression final dimension always equals width.
- VQC: two variational layers, angle-Y encoding, linear CNOT entanglement,
  no data re-uploading, full readout, `default.qubit` with backpropagation.
- Optimization and data bundle: the committed `configs/histopath.yaml`
  settings used by the 8-qubit paper runs, including focal loss, strong
  augmentation, TTA and validation-derived F-beta thresholding.
- Stages: A, B and C with their configured epoch budgets, early stopping and
  independent stage checkpoints.
- Hardware: Kaggle GPU T4 ×2 for classical stages; the simulator may execute
  on CPU. One width-fold pair is one immutable Kaggle version.

No width-specific learning rate, epoch budget, threshold rule, augmentation or
checkpoint-selection rule may be introduced after a run is seen.

## Outcomes

**Primary metric:** held-out patient-fold test AUPRC from the globally selected
checkpoint, where checkpoint stage and epoch are selected from validation data
only.

For each untouched fold 1–4, compute paired differences `q4 − q8` and
`q12 − q8`. Aggregate folds with the same Nadeau–Bengio corrected t procedure
used in the Stage-B analysis. Report the mean paired difference, corrected 90%
and 95% intervals and every per-fold value.

**Secondary metrics:** ROC AUC, balanced accuracy, F1, precision, recall,
selected stage, stage-specific validation score, runtime, trainable parameter
count and circuit parameter count. Where the modern stage-attribution artifact
is available, report Stage-B-minus-Stage-A and Stage-C-minus-Stage-A test AUPRC
within the same width. Do not impute unavailable stage-specific Fold-0 values
from its legacy run.

## Decision rules

The practical margin is fixed at ±0.01 AUPRC.

- A width is **practically equivalent** to q8 only if the corrected 90%
  interval for its paired AUPRC difference lies entirely inside ±0.01.
- A width is **meaningfully better or worse** only if the corrected 95%
  interval excludes zero and the mean absolute difference exceeds 0.01.
- Any other result is **inconclusive at the observed interval width**.
- A quantum-stage benefit is not inferred merely because q4 or q12 beats q8.
  It additionally requires Stage B or C to be selected from validation data,
  or a positive within-width quantum-stage AUPRC increment where stage
  attribution is available.

These rules govern folds 1–4. A five-fold descriptive analysis including the
previously seen Fold 0 is reported beside it but cannot upgrade the replication
claim.

## Multiplicity and interpretation

There are two planned comparisons against q8: q4 and q12. Report both without
selecting the more favorable width. The analysis is estimation-first; interval
widths and effect sizes govern the interpretation. If formal difference
p-values are supplied, Holm correction across the two width comparisons must
be shown. Equivalence TOST results are reported separately for both widths.

The strongest admissible conclusion is that the tested end-to-end systems are
equivalent, different or inconclusive within the stated margin. No claim of
quantum computational advantage, hardware speedup or general superiority is
permitted because every run uses a classical simulator.

## Resource plan

Eight new Kaggle versions are required: folds 1–4 at q4 and q12. Based on the
existing Fold-0 runs, q4 requires about 4.6 hours and q12 about 5.9 hours per
fold, for approximately 42 GPU-session hours in total. Allow 48–55 hours for
queueing and fold-dependent early stopping. Runs may execute concurrently on
separate accounts, but each output must retain its commit, fold, width, seed,
patient counts and stage-attribution artifacts.
