# Paper D — execution plan

**Working title:** *Does a Paper-B–style hybrid VQC head transfer on BreakHis
under case-level splits, or does the null stay?*

**Thesis:** domain shift (magnification / staining) is a separate estimand from
Paper B's within-cohort capacity comparison. Reusing B's circuit family on
BreakHis closes the single-cohort limitation without another IDC bake-off.

**Quantum:** yes — same family as Paper B (8 qubits, 2 layers, angle-Y, linear
entanglement). "No gain under shift" is pre-allowed.

**Primary cohort:** BreakHis, all four magnifications as strata, case-ID CV.

**Venue (locked 2026-09-22):** **Connection Science** (Taylor \& Francis).
See [`VENUE.md`](VENUE.md) and [`APC_GATE.md`](APC_GATE.md).

**Compute:** campus RTX A4000 (preferred) or Kaggle T4; environment must be
recorded in the lock manifest before labelled runs.

---

## Phase 0 — folder and protocol (complete)

1. Create `papers/paper_d/` as the Mac + GitHub workspace.
2. Lock series priority: D is the immediate QML follow-on after B submit.
3. File [`preregistration/paper_d_breakhis_protocol.md`](../../preregistration/paper_d_breakhis_protocol.md).

## Phase 1 — data and lock (complete)

Lock verify passes; confirmatory cells trained.

## Phase 2 — confirmatory cells (complete)

All 20 cells under `results/breakhis_paper_d/`. Curated:
`papers/paper_d/results/curated_metrics.json`.

## Phase 3 — T&F secondaries (complete 2026-09-22)

Protocol amendment dated 2026-09-22. Exporter:
`papers/paper_d/scripts/export_tf_secondaries.py` →
`papers/paper_d/results/tf_secondaries.json`.

| ID | Status |
|----|--------|
| S1 per-magnification | **Done** — fig03 |
| S2 calibration ECE | **Done** — F-MLP ECE 0.135±0.043; F-VQC 0.228±0.082; fig04 |
| S3 wall-clock / params | **Done** — in `tf_secondaries.json` |
| S4 IDC→BreakHis | **Skipped** (optional) |

Co-primary (unchanged): frozen Δ −0.0085, CI [−0.034, +0.017], Holm *p* = 0.80;
fine-tune Δ +0.0000, CI [−0.027, +0.027], Holm *p* = 0.997.

## Phase 4 — manuscript and submit

1. Manuscript: `manuscript/main.tex` (Connection Science / T\&F framing).
2. Checklist: [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md).
3. Tag `paper-d-tf-v1.0` when PDF matches curated JSON.
4. Turnitin (similarity + AI) → IQAC → submit `ccos20`.
