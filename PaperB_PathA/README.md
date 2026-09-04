# PaperB Path A — fair-warmup staged hybrid rerun

Self-contained plan and tooling for the Path A experiment. Everything for this
experiment lives here; nothing under `papers/paper_b/` is modified by running
it.

Governing declaration:
`preregistration/staged_hybrid_fair_warmup_protocol.md`.

## Why this experiment exists

Every published staged result resumed each stage from the preceding stage's
**terminal** weights, while validation selection compared each stage's **best**
checkpoint. Stage B therefore started from a slightly degraded classical
predictor and was then scored against the classical high-water mark.

The size of that handicap matches the size of the margin Stage B lost by:

| Fold | Stage A best | Stage A terminal | Stage B best | B − A(best) | B − A(terminal) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8664 | 0.8582 | 0.8569 | −0.0096 | −0.0013 |
| 1 | 0.8970 | 0.8896 | 0.8933 | −0.0037 | **+0.0037** |
| 2 | 0.8803 | 0.8802 | 0.8760 | −0.0043 | −0.0043 |
| 3 | 0.8959 | 0.8883 | 0.8926 | −0.0033 | **+0.0042** |
| 4 | 0.8623 | 0.8570 | 0.8602 | −0.0020 | **+0.0032** |

Validation balanced accuracy. Measured against Stage A's terminal checkpoint,
Stage B is ahead in **three of five folds**. The published null is therefore
sound as a statement about that schedule but cannot settle what happens under a
fair one.

## The environment problem, and why there are two arms

The published five-fold E3 runs were executed on **Kaggle T4 x2**
(`results/histopath_kaggle_fold*_e3_v2/`). The available GPU system is the
**RTX A4000 server**, which so far has only ever run the width matrix.

Comparing a server fair-warmup run against Kaggle-published numbers would
confound the checkpoint change with the change of environment. The width
protocol already set the precedent for refusing that: a Kaggle q8 baseline
could not be paired with server q4/q12 cells, so q8 was rerun on the server.

Path A follows the same rule with a paired, within-environment design:

- **Arm A0 — control.** `--no-stage-init-from-best`. Reproduces the published
  schedule on the server. Serves both as the paired baseline for A1 and as an
  environment-replication check against the published Kaggle results.
- **Arm A1 — treatment.** `--stage-init-from-best`. The fair-warmup schedule.
  Identical in every other respect.

Only A1 versus A0 supports a claim. A0 versus the published Kaggle numbers is a
replication observation, not a test.

## Measured cost

Wall-clock is taken from `train_time_s` in the archived run artifacts, not
estimated.

| Configuration | Measured | Basis |
| --- | --- | --- |
| Published E3 fold, Kaggle T4 x2 | 3.11–5.16 h (mean 4.12) | five published folds |
| E3-equivalent fold, RTX A4000 | 0.99–1.31 h (mean 1.14) | four server q8 width cells |

The server is roughly 3.6x faster, so:

- Arm A0: 5 folds x ~1.15 h ≈ **5.8 h**
- Arm A1: 5 folds x ~1.15 h ≈ **5.8 h**
- **Both arms ≈ 12 GPU-hours** — one overnight run.

Optional Arm A2 (budget-matched, early stopping disabled, stages A and B capped
at 15 epochs each) raises per-fold epochs from ~21.6 to 33, so ~1.7 h/fold and
about **9 h** more. Run it only after A0 and A1 are complete and analysed.

## Fixed protocol

Everything below is held at the published values. Changing any of them voids
the comparison.

- splits: `data/splits/histopath_kaggle/` (the five-fold paper partition)
- experiment `E3`, 8 qubits, linear entanglement
- focal loss gamma 2, strong augmentation, TTA over six views
- F-beta threshold tuning, beta 1.5
- batch size 64, AdamW, weight decay 1e-4, gradient clipping 1.0
- learning rates 1e-4 (stages A/B), 1e-5 (stage C)
- stage caps 25 / 15 / 3, early-stopping patience 5
- seed 42 (effective seed is `42 + fold`)
- selection metric: validation balanced accuracy at the 0.5 cutoff

## Running it

Runs go through the qualified container by default, so each cell records the
same provenance the width matrix carries: pinned image id, dependency-lock
digest, dataset digest, and the five-fold manifest digest. Build the image
first if it is not already present:

```bash
scripts/build_histopath_server_image.sh
```

Then:

```bash
# Preview the full queue without launching anything.
PaperB_PathA/scripts/run_patha_queue.sh --dry-run

# Run all ten cells, one at a time, skipping any already complete.
PaperB_PathA/scripts/run_patha_queue.sh

# Or a single cell.
PaperB_PathA/scripts/run_patha_server.sh 0 control
PaperB_PathA/scripts/run_patha_server.sh 0 fair
```

The container wrapper refuses to start unless the quantum config, stage caps,
early-stopping patience, loss/TTA bundle, GPU model, and manifest digest all
match the declaration, and it refuses to overwrite a completed cell. Artifacts
land under `$MBC_PRIMARY_ROOT/results/path-a/fold{N}_{tag}/` with the
provenance JSON in `$MBC_PRIMARY_ROOT/bundles/path-a/`.

The declared five-fold manifest digest is
`4a0a72fa3c89250cd012b943374be2301c0eb8ea2f4dd7d968b66c04b76bdf83`, taken over
the git-tracked manifest files, so anyone with the repository can reproduce it:

```bash
python PaperB_PathA/scripts/run_patha_server.py --print-manifest-sha
```

A bare-Python fallback exists for debugging only. It records the git commit but
not the image or dataset digests, so it is **not** reportable provenance:

```bash
PaperB_PathA/scripts/run_patha_queue.sh --no-container
PaperB_PathA/scripts/run_patha_fold.sh 0 fair
```

Runs are tagged `termwarm` (A0) and `fairwarm` (A1) by
`--no-stage-init-from-best` / `--stage-init-from-best`, so the two arms cannot
overwrite each other.

## Analysis

```bash
python PaperB_PathA/scripts/compare_patha.py
```

Reports, per fold and per arm: the selected stage, the Stage-A and Stage-B best
validation scores, the selection margin, whether Stage C collapsed, and the
test metrics of the selected checkpoint. It also prints the A0-versus-published
replication check.

## Pre-declared decision rule

Primary endpoint: **the number of folds, out of five, in which validation
selects a quantum stage** under A1.

- **Stage A wins 5/5 in A1** — the published conclusion is strengthened. Report
  as a negative confirmation and remove the schedule caveat from the
  Limitations, replacing it with this result.
- **A quantum stage wins in any fold of A1** — the published selection claim is
  conditional on the terminal-weight schedule. The manuscript's central claim
  must be rewritten, and that fold's test metrics reported whether or not they
  favour the quantum route.

All five folds of both arms must be reported, including unfavourable ones. A
failed run is retried once with an identical configuration and both attempts
recorded. No fold may be dropped without recording the failure and its cause.
