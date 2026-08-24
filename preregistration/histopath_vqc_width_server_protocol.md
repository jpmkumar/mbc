# Server-primary end-to-end VQC width protocol

Status: declared 2026-08-14 before any RTX A4000 model training.

## Why this protocol exists

The original width protocol fixed Kaggle T4 x2 as the execution environment.
After two q4 Kaggle replications had completed, a dedicated physical GPU
workstation became available and Kaggle quota became a material constraint.
Changing hardware for only the remaining cells would confound width with
execution environment. This protocol therefore restarts the complete
confirmatory matrix on one immutable server environment.

The server matrix is the primary width analysis. Existing and future Kaggle
cells remain a separately labelled historical/secondary replication and are
never pooled with server results.

## Prior outcome knowledge

Before this declaration:

- Fold 0 width checks were exploratory.
- Kaggle Fold 1/q4 selected Stage A; Stage B reduced test AUPRC by 0.11074,
  and Stage C had a numerical convergence failure.
- Kaggle Fold 2/q4 selected Stage B by a validation balanced-accuracy margin
  of 0.000205; Stage B minus Stage A test AUPRC was -0.00330.
- No Fold 1-4 q12 server result and no server training outcome had been seen.

These observations motivated neither a learning-rate change nor a training
algorithm change. They are disclosed so the server restart cannot be
misrepresented as outcome-blind.

## Primary matrix and estimand

Run every Cartesian pair:

- folds: 1, 2, 3, 4;
- widths: 4, 8, 12 qubits.

All twelve cells must use the same physical workstation, image family,
dependency lock, source revision, dataset archive and split manifest. The
primary estimands are paired test-AUPRC differences `q4 - q8` and `q12 - q8`
within each fold. Rerunning q8 is mandatory; a Kaggle q8 baseline cannot be
paired with server q4/q12 results.

## Fixed training protocol

The model and optimization settings remain those in
`preregistration/histopath_vqc_width_protocol.md`:

- patient-level five-fold split, seed 42;
- E3 staged training with independently checkpointed Stages A, B and C;
- width-matched final compression dimension;
- two variational layers, angle-Y encoding, linear CNOT entanglement and no
  data re-uploading;
- focal loss, the existing augmentation bundle, TTA and validation-derived
  F-beta thresholding;
- the checkpoint selected by validation balanced accuracy supplies the
  primary test result;
- no width-specific learning rate, epoch budget, threshold or retry rule.

The fold-specific effective seed remains `42 + fold`.

## Immutable execution environment

The qualification environment is:

- host GPU: NVIDIA RTX A4000, 16 GB;
- host CPU/RAM: Intel i9-12900K, 62 GB;
- base image:
  `nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`;
- dependency-additions lock SHA-256:
  `55ae29e55a5e3643fb59be8e3aaa2c1466e63efcd3559bc22b4addab4e7c829a`;
- PyTorch 2.13.0a0 NVIDIA 26.06, CUDA runtime 13.3, PennyLane 0.45.1;
- CUDA Forward Compatibility mode with host kernel driver 580.173.02.

The final experiment image must bake a clean `git archive` of the declared
source commit. Every result bundle records the image ID, source commit, GPU
name/UUID, driver/runtime versions, dependency-lock hash, dataset hash and
split-manifest hash. Code is not bind-mounted during training.

## Numerical validity and reruns

Every Stage A/B/C checkpoint is audited for non-finite values. A non-finite
stage is a numerical convergence failure and its metrics are missing, not
zero. If validation selects a non-finite stage, the cell is invalid.

There are no silent retries. Operational interruption before a result exists
(power loss, host reboot, storage failure) may resume from the recorded latest
checkpoint and must be disclosed. A completed finite run is not repeated
because its result is inconvenient.

## Analysis and interpretation

Use the Nadeau-Bengio corrected intervals and TOST rules already declared for
the width analysis. Report:

- every fold-width selected stage and test metric;
- paired q4-q8 and q12-q8 differences;
- corrected 90% and 95% intervals;
- equivalence decisions within the predeclared practical margin;
- numerical-failure frequency by width;
- wall-clock time and peak GPU memory as descriptive secondary outcomes.

No claim of quantum computational advantage or hardware speedup is permitted:
the VQC remains a classical state-vector simulation.

