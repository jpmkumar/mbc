# Paper C+ protocol — multi-cohort replication of grouping (and FOV if feasible)

**Status:** LOCKED — outcome-locked; labelled comparative probes may proceed
only after Stage-2 lock verify.  
**Created:** 2026-11-11.  
**Locked:** 2026-11-12.  
**Parent study:** Paper C (`preregistration/paper_c_protocol.md`, LOCKED).  
**Lock condition:** status is `LOCKED`; complete Stage 2–3 in section 8 before
first labelled comparative F-MLP run.

## 0. Relationship to Paper C and Paper D

- Paper C discovered protocol optimism and a null DINOv2 FOV contrast on
  IDC+BCSS. Those locked estimates are **not** re-tuned here.
- Paper D uses BreakHis for hybrid QML transfer. C+ may use the same BreakHis
  archive for **classical** grouping/FOV questions only. C+ must freeze its own
  case splits and lock digests; it must not overwrite Paper D artifacts.

## 1. Research questions

1. Does random-patch cross-validation overestimate case-balanced AUPRC relative
   to case-ID-grouped CV on **BreakHis** under the same frozen DINOv2 linear
   probe used in Paper C?
2. *(Conditional)* If BreakHis tiles support a preregistered physical or
   grid-neighbourhood context definition, does expanded context improve grouped
   DINOv2 AUPRC relative to isolated tiles?
3. Does question 1 replicate on a **second** public breast cohort (BACH), with
   its own case/slide grouping unit?

## 2. Cohorts

### BreakHis (primary new cohort)

- Archive: BreaKHis_v1 (already materialized for Paper D).
- Grouping unit: case identifier encoded in BreakHis filenames / folder
  structure (same case-id definition as Paper D).
- Folds (**closed 2026-11-11**): **new C+ case folds** via
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=32026)` written
  to `data/splits/paper_c_plus/breakhis/`. Paper D’s `data/splits/breakhis/`
  (seed 42) is not overwritten. Random-patch arm uses image-level
  `StratifiedKFold` with the same seed.
- Magnification (**closed 2026-11-11**): **pooled** primary — all of
  40×/100×/200×/400×; magnification is a bootstrap stratum for uncertainty
  reporting. Per-mag tables are ordered secondary.
- Labels: binary benign vs malignant at image level (standard BreakHis binary
  task). Multi-class histology subtypes are ordered secondary.

### BACH (second cohort; Phase 3)

- ICIAR 2018 BACH; grouping unit = slide/case as defined in the release.
- Protocol-optimism contrast only unless native coordinates allow FOV.

### IDC / BCSS

- Not re-analysed as co-primary. Cited from Paper C.

## 3. Models and confirmatory adaptation (Option B — closed 2026-11-12)

**Encoders (frozen at confirmatory extract):**

1. DINOv2 ViT-L/14 — reference (`torch.hub`);
2. ImageNet ResNet-50 — control (`torchvision`).

Gated UNI2-h / Virchow2 / Phikon-v2 remain unavailable unless institutional
access is completed **before** Stage-1 lock and listed here with revisions.

**Confirmatory head (accuracy-first, still protocol-fair):** a two-layer MLP on
frozen embeddings, identical architecture for both encoders aside from input
width:

- input → Linear(`D`, 512) → ReLU → Dropout(0.2) → Linear(512, 1);
- `D=1024` for DINOv2, `D=2048` for ResNet-50;
- trained with Adam (lr \(1\times10^{-3}\), weight decay \(1\times10^{-4}\)),
  binary cross-entropy with **case-balanced** sample weights on both protocol
  arms;
- early stop on inner validation case-balanced AUPRC; temperature + threshold
  fit on the inner calibration split (same discipline as Paper C).

Encoders stay frozen in the confirmatory arm (**F-MLP**). A light encoder
fine-tune arm (**T-MLP**, last block / layer only, lr \(1\times10^{-5}\)) is an
**ordered secondary**, not co-primary.

A Paper-C-matched **linear SGD probe** on the same frozen embeddings is an
ordered secondary bridge to Paper C’s adaptation recipe.

## 4. Co-primary estimands

### Always locked

**Δ_protocol (BreakHis):** random-minus-grouped case-balanced AUPRC for
**DINOv2 + confirmatory F-MLP**, seeds averaged (42/43/44), paired case
bootstrap, 10 000 replicates, magnification as bootstrap stratum.

Smallest effect of interest: **0.01** case-balanced AUPRC (same as Paper C).

### Conditional second co-primary

**Δ_context (BreakHis):** expanded-context minus isolated-tile grouped
case-balanced AUPRC for DINOv2, **only if** Section 5 declares a feasible
context construction before lock. Otherwise FOV is omitted from the Holm
family and reported only as “not estimable on BreakHis tiles.”

Holm family size = number of co-primaries actually locked (1 or 2).

## 5. Field-of-view / context rule (locked)

BreakHis images are typically **pre-cropped tiles**, not a coordinate grid like
IDC. FOV co-primary requires one of:

- **A. Infeasible:** no unique slide-neighbour graph → FOV not co-primary;
- **B. Feasible:** explicit neighbour rule (e.g. same-slide tiles within a
  stated µm/pixel window, or fixed mosaic from WSI if WSIs are used).

**Audit (2026-11-11):** BreakHis files are isolated numbered crops
(e.g. `...-400-004.png`) without a coordinate grid. Neighbour mosaics like
Paper C’s IDC `K=9` are **not** constructible from the public tile release.

**Locked stance:** **A (infeasible)** — C+ co-primary is **protocol optimism
only** on BreakHis (then BACH). FOV replication stays with Paper C’s IDC/BCSS
locked results; C+ may discuss FOV only as “not estimable here.” Holm family
size = 1 (Δ_protocol).

## 6. Probe and inference

- Confirmatory: **F-MLP** (Section 3) under random-patch and case-ID-grouped
  five-fold CV using `data/splits/paper_c_plus/breakhis/` (seed 32026).
- Both protocol arms use the same MLP, Adam settings, early-stop rule and
  case-balanced weighting so Δ isolates grouping.
- Bundled random regime (class-balanced rather than case-balanced fitting) is
  an ordered secondary, as in Paper C.
- Seeds 42/43/44; average calibrated probabilities before bootstrap.
- No labelled metric inspection until LOCKED + lock verify.

## 7. Multiplicity and reporting

- Holm step-down on the locked co-primary family only (BreakHis Δ_protocol for
  DINOv2 F-MLP; FOV omitted).
- Ordered secondaries (not in Holm family):
  - ResNet-50 F-MLP protocol gap;
  - DINOv2 / ResNet **linear** probe protocol gap (Paper C bridge);
  - DINOv2 / ResNet **T-MLP** light fine-tune (if compute allows);
  - per-magnification gaps; BACH protocol gap; calibration.
- Absolute grouped AUPRC is reported for transparency; the co-primary claim is
  still the **difference** random−grouped, not a leaderboard score.
- Honest nulls are successes for replication science.

## 8. Lock procedure

Stage 1 — set this file to `LOCKED`, commit protocol + code + C+ splits.  
Stage 2 — write `preregistration/paper_c_plus_lock.json` with commit, protocol
SHA-256, BreakHis archive digest, split digests, image/env digests.  
Stage 3 — `verify` before first labelled comparative run.

## 9. Decisions

| Item | Status |
|---|---|
| Mag rule | **Closed:** pooled + mag stratum |
| FOV | **Closed:** not co-primary on BreakHis |
| Case folds | **Closed:** new C+ seed 32026 |
| Adaptation | **Closed (2026-11-12):** Option B — confirmatory **F-MLP**; linear + T-MLP secondary |
| BACH path/license | **Open** — before Phase 3 |

Builders: `papers/paper_c_plus/scripts/build_breakhis_cplus_splits.py`,
`build_breakhis_cplus_inner_splits.py`.  
Confirmatory F-MLP runner: `papers/paper_c_plus/scripts/run_breakhis_mlp_cv.py`.  
Extract: `papers/paper_c_plus/scripts/extract_breakhis_embeddings.py`.
