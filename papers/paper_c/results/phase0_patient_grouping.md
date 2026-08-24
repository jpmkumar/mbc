# Phase 0 result — the grouping variable

**Run:** 2026-08-21, campus RTX A4000 server (`SJT516SCOPE052`), archive at
`~/mbc-primary/datasets/breast-histopathology-images`. CPU only.

## Measured

| Quantity | Value | Expected |
|---|---|---|
| Top-level directories | 279 | 279 |
| Total patches | 277,524 | 277,524 |
| IDC-positive patches | 78,786 | 78,786 |
| Filenames failing the `{stem}_idx{N}_x{X}_y{Y}_class{C}.png` pattern | 0 | — |
| Distinct `idx` values across the whole archive | `{5}` | — |
| Directories carrying more than one `idx` | 0 | — |
| Filename stems appearing in more than one directory | 0 | — |
| Directories whose name differs from their filename stem | 0 | — |

Patch and positive counts match the canonical figures exactly, so the archive on
the server is the complete, uncorrupted distribution.

## What this establishes

**The archive is internally consistent at the directory level.** Every directory
contains exactly one filename stem, that stem equals the directory name, and no
stem is shared across directories. There is no aliasing: two directories are
never two views of the same identifier.

**The frozen folds stand.** `collect_patient_stats()` groups by directory name,
and the audit finds no structure that would make directory-level grouping
narrower than the true dependency unit *within what the archive exposes*. Paper B's
folds do not need re-auditing, and Paper C reuses them unchanged.

## What this does *not* establish — and the paper must say so

`idx` is **not** a slide index. It takes the single constant value 5 across all
277,524 files, inherited from the `IDC_regular_ps50_idx5` generation parameters
(patch size 50, index 5). It therefore carries **no** slide or patient
information whatsoever.

This matters, because it means the archive contains **no field that links two
directories to the same patient**. The 279 identifiers are self-consistent, but
nothing in the distribution can distinguish these two hypotheses:

- 279 directories are 279 distinct patients; or
- 279 directories are 279 *slides* drawn from the 162 patients described in the
  source study, some of whom contributed more than one whole-mount slide.

The counts are at least consistent with the second reading — 279 slides from 162
patients averages 1.7 slides per patient, which is unremarkable for whole-mount
sampling. **The audit cannot decide between them, and neither can anyone else
using this Kaggle distribution.**

## Consequence for Paper C

Directory-level grouping is the **strongest grouping the data permits**, not
demonstrably patient-level grouping. The honest statement for the manuscript is:

> Splits are disjoint at the level of the 279 case identifiers exposed by the
> public distribution. Because the release carries no patient field, and the
> source cohort is described as 162 whole-mount slides, we cannot exclude that
> some identifiers originate from the same patient. Any residual dependency of
> this kind inflates all reported estimates, including ours, and is not removable
> from this distribution.

This is a **strengthening** finding for the paper's thesis, not a weakening one.
The literature routinely calls directory-level splits "patient-level" without
evidence. Paper C can state precisely what is and is not guaranteed, which is
exactly the evaluation-integrity discipline the paper argues for (gaps G1 and
G10 of the gap analysis). It also means
the true leakage penalty measured in C1 is a **lower bound** on the penalty that
random patch splitting incurs relative to genuine patient disjointness.

## Follow-up worth doing (cheap, optional)

The patch coordinate extents per directory are a proxy for mount area. If some
directories show coordinate ranges consistent with being sub-regions of a shared
larger mount, that would be weak evidence for the slide reading. This is a
descriptive addition for the data card, not a gate on any experiment.
