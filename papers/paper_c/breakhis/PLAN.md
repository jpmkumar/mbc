# Paper C+ — multi-cohort replication of grouping and field-of-view effects

**Working title:** *Do grouping optimism and field-of-view effects replicate
across breast-histology cohorts beyond IDC+BCSS?*

**Relation to Paper C:** follow-up study. Paper C stays locked and is cited as
the discovery cohort. **No** Paper C splits, thresholds, or probe choices are
changed to chase higher AUPRC.

**Quantum:** none. Classical frozen encoders only (same ungated panel as C:
DINOv2 ViT-L/14 + ImageNet ResNet-50), unless institutional gated access opens
*before* the C+ lock and is declared in the C+ protocol.

**Thesis:** Paper C found (i) significant random-vs-grouped optimism on IDC for
DINOv2 and (ii) no co-primary DINOv2 gain from matched \(K=9\) context, with a
null BCSS DINOv2 FOV replication. C+ asks whether protocol optimism replicates
on **new case-level cohorts**, using a stronger confirmatory head so absolute
grouped performance is competitive while Δ still isolates grouping.

**Adaptation (Option B, 2026-11-12):** confirmatory **frozen encoder + MLP**
(F-MLP). Linear probe = Paper C bridge secondary. Light fine-tune (T-MLP) =
ordered secondary.

---

## Cohort ladder (priority order)

| Priority | Cohort | Why | Status |
|---|---|---|---|
| 1 | **BreakHis** | Already on campus disk for Paper D; case IDs + magnifications; different stain/FOV regime | Ready to lock |
| 2 | **BACH** (ICIAR 2018) | Slide/ROI breast histology; public; distinct label taxonomy | Download if Phase 1 BreakHis lands |
| 3 | Optional third | Only if 1–2 leave a gap (e.g. Camelyon16 patch set **without** claiming FOV mosaics) | Deferred |

**Boundary with Paper D:** D asks whether a hybrid VQC head transfers on
BreakHis. C+ asks classical **grouping** questions on BreakHis with confirmatory
**F-MLP** (frozen encoder + MLP). Shared raw archive is fine; **splits, locks,
estimands, and manuscripts stay separate**.

---

## Co-primary estimands (draft — freeze in protocol before labelled runs)

1. **Protocol optimism (BreakHis):** random-patch minus case-ID-grouped
   case-balanced AUPRC for **DINOv2 F-MLP** (pooled mags; mag as bootstrap
   stratum).
2. **FOV on BreakHis:** **dropped from co-primary.** Tile audit shows isolated
   numbered crops with no neighbour coordinates (unlike IDC). FOV stays a
   Paper C finding; C+ cites it and does not fake mosaics.

Holm family size = **1** (BreakHis protocol optimism), plus a separate BACH
protocol contrast as replication (ordered or second Holm family — decide at
lock; default: BACH is confirmatory secondary until sample-size note is done).

---

## Phases

### Phase 0 — scaffold — complete

- [x] PLAN + README
- [x] Protocol Stage-1 LOCKED (2026-11-12)
- [x] FOV: not co-primary on BreakHis
- [x] Mag rule: pooled + mag-stratum bootstrap
- [x] New C+ case folds seed **32026** → `data/splits/paper_c_plus/breakhis/`
- [x] Smoke DINOv2 (1024-d) + ResNet-50 (2048-d) on 8 BreakHis tiles (A4000)
- [x] Option B adaptation: confirmatory F-MLP (protocol updated 2026-11-12)

### Phase 1 — BreakHis classical lock

- [x] Manifest + grouped/random folds built (7,909 images / 82 cases)
- [x] Inner train/val/cal case lists (seed_base 1042)
- [x] Implement `run_breakhis_mlp_cv.py` + embedding extract scripts
- [x] Flip protocol to LOCKED + `paper_c_plus_lock.json`
- [x] Full embedding extract (both encoders) then labelled F-MLP probes
- [x] Locked inference (seed-mean + 10k case bootstrap + Holm family=1)

### Phase 2 — labelled BreakHis probes

- [x] Random vs grouped **F-MLP**, seeds 42/43/44, seed-mean then case bootstrap
- [x] Secondaries: linear SGD bridge (DINOv2+ResNet); per-mag tables (F-MLP + linear)
- [ ] Optional T-MLP (deferred — not required for BreakHis lock claim)
- [x] Numbers in `results/curated_metrics.json`
- [x] Manuscript draft: `manuscript/main.tex` (local; cites Paper C + BreakHis lock)

### Phase 3 — BACH (second cohort)

- Download + license check; new lock appendix; protocol optimism replication

### Phase 4 — manuscript polish / venue

- **Target venue:** Wiley *Engineering Reports* (Gold OA; provisional JCR Q2 Engineering, Multidisciplinary; APC ~USD 2,500). See `manuscript/VENUE_ENGINEERING_REPORTS.md`.
- **Scope:** **one paper** = locked IDC/BCSS (Paper C) + BreakHis replication (C+); not two submissions.
- [x] Combined Eng.\ Reports draft (`manuscript/main.tex` + `main.docx` + cover letter)
- [ ] Confirm Q2 on WoS via `egateway.vit.ac.in` before IQAC/APC claim
- [ ] Author block + ORCID; Paper C DOI/preprint once available
- [ ] Optional forest-plot figure; expand refs; Word/PDF export for Free Format
- [ ] Turnitin (similarity + AI) → IQAC; VIT faculty corresponding author
- [ ] BACH only if we choose to enlarge scope before first submission

---

## What we will *not* do

- Amend Paper C protocol or re-run IDC/BCSS to “fix” the FOV null
- Fold QML into C+
- Swap in gated FMs after seeing BreakHis outcomes
- Invent BreakHis mosaics without coordinates
- Claim patient independence where only case/slide IDs exist

---

## Immediate next actions

1. Verify *Engineering Reports* Q2 on WoS (egateway) and screenshot for APC file.
2. Fill author block / ORCID / corresponding `@vit.ac.in` in `manuscript/main.tex`.
3. Export PDF/Word, run Turnitin (similarity + AI), then submit Free Format + cover letter.
4. Optional: Short Communication (≤4k words; lower APC) if editors prefer a compact replication.
