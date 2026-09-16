# Paper D — execution plan

**Working title:** *Does a Paper-B–style hybrid VQC head transfer on BreakHis
under case-level splits, or does the null stay?*

**Thesis:** domain shift (magnification / staining) is a separate estimand from
Paper B's within-cohort capacity comparison. Reusing B's circuit family on
BreakHis closes the single-cohort limitation without another IDC bake-off.

**Quantum:** yes — same family as Paper B (8 qubits, 2 layers, angle-Y, linear
entanglement). "No gain under shift" is pre-allowed.

**Primary cohort:** BreakHis, all four magnifications as strata, case-ID CV.

**Compute:** campus RTX A4000 (preferred) or Kaggle T4; environment must be
recorded in the lock manifest before labelled runs.

---

## Phase 0 — folder and protocol (complete at declaration)

1. Create `papers/paper_d/` as the Mac + GitHub workspace.
2. Lock series priority: D is the immediate QML follow-on after B submit.
3. File [`preregistration/paper_d_breakhis_protocol.md`](../../preregistration/paper_d_breakhis_protocol.md).

**Gate:** no labelled BreakHis comparison until Phase 1 lock digests exist.

## Phase 1 — data and lock

1. Obtain BreakHis; record archive SHA-256.
2. Audit case / patient identifiers; build case-ID five-fold (or declared *k*)
   manifests under `data/splits/breakhis/`. Commit case lists only.
3. Freeze `config/circuit_registry.json` to Paper B's 8q / E2b-matched specs.
4. Run `python papers/paper_d/scripts/write_paperd_lock.py` →
   `preregistration/paper_d_lock.json`.
5. Verify with `python papers/paper_d/scripts/verify_paperd_lock.py`.

**Gate:** lock verify passes; protocol status remains LOCKED; no test-fold
scores inspected.

## Phase 2 — confirmatory cells

Extract slides (local):

```bash
tar -xzf papers/paper_d/data/BreaKHis_v1.tar.gz -C papers/paper_d/data
```

Run all four arms on every committed fold via
`papers/paper_d/scripts/run_breakhis_queue.sh`:

| ID | Regime | Head |
|----|--------|------|
| F-MLP | Frozen backbone | E2b-matched MLP |
| F-VQC | Frozen backbone | Paper-B VQC |
| T-MLP | Last-block fine-tune | E2b-matched MLP |
| T-VQC | Last-block fine-tune | Paper-B VQC |

Promote only curated summaries into `papers/paper_d/results/` (selective track,
same pattern as Paper B/C). Raw logs stay in root `results/breakhis_paper_d/`.

## Phase 3 — analysis and manuscript

1. Co-primary Δ under frozen and fine-tune regimes; Holm at 0.05; SEI ±0.01
   AUPRC.
2. Ordered secondaries per protocol (per-mag, IDC→BreakHis, calibration).
3. Draft manuscript locally under `manuscript/` (not pushed). Tag `paper-d-v1.0`
   when code + curated metrics match the submission.
