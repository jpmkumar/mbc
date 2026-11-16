# Paper C — grouping protocol and FOV for frozen histology encoders

**One journal submission** (Wiley *Engineering Reports*): IDC/BCSS discovery
(Stage 1) plus BreakHis replication (Stage 2, formerly “Paper C+”).

**Canonical code path:** `papers/paper_c/`. Manuscript sources stay local under
`manuscript/` and are not pushed to GitHub.

| Stage | Cohort | Question | Lock |
|-------|--------|----------|------|
| 1 | IDC + BCSS | Random vs case-grouped optimism; \(K=9\) FOV | `preregistration/paper_c_lock.json` |
| 2 | BreakHis | Does protocol optimism replicate under F-MLP? | `preregistration/paper_c_plus_lock.json` |

**Standalone release repo (preferred citation):**
https://github.com/jpmkumar/mbc-paper-c

**Shared with Paper B:** Kaggle IDC archive and frozen case-ID folds. Paper C is
classical-only (no quantum). BreakHis stage does not modify Stage 1 locks.

## Layout

| Path | Purpose |
|------|---------|
| [`PLAN.md`](PLAN.md) | Stage 1 execution plan |
| [`breakhis/PLAN.md`](breakhis/PLAN.md) | Stage 2 (BreakHis) plan |
| [`SERVER_PAPER_C.md`](SERVER_PAPER_C.md) | Campus RTX A4000 runbook (IDC/BCSS) |
| [`config/model_registry.json`](config/model_registry.json) | Encoder panel |
| [`scripts/`](scripts/) | IDC/BCSS extraction, probes, locked inference |
| [`breakhis/scripts/`](breakhis/scripts/) | BreakHis splits, embeddings, F-MLP/linear, inference |
| [`results/`](results/) | Stage 1 curated metrics + audit notes |
| [`breakhis/results/`](breakhis/results/) | Stage 2 curated metrics |
| [`../../preregistration/paper_c_protocol.md`](../../preregistration/paper_c_protocol.md) | Stage 1 protocol |
| [`../../preregistration/paper_c_plus_protocol.md`](../../preregistration/paper_c_plus_protocol.md) | Stage 2 protocol |

Split manifests remain under `data/splits/paper_c/` and
`data/splits/paper_c_plus/breakhis/` (path names frozen in the lock JSON).
There is no separate `papers/paper_c_plus/` code tree.
