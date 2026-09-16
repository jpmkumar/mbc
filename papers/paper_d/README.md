# Paper D — BreakHis external histopath generalisation (hybrid QML)

**Canonical code path (Mac checkout and GitHub):** `papers/paper_d/`.

Manuscript LaTeX and submission packs stay **local** under
`papers/paper_d/manuscript/` (gitignored, same rule as A/B/C). Research code,
configs, split manifests, lock tooling, and this index are **tracked** and
appear at:

```text
https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_d
```

**Question:** Does a Paper-B–style hybrid VQC head transfer across magnification
and staining on BreakHis under case-level splits, or does the null stay?

**Governing preregistration:**
[`../../preregistration/paper_d_breakhis_protocol.md`](../../preregistration/paper_d_breakhis_protocol.md)

"No gain under shift" is a pre-allowed outcome. This is not another Kaggle IDC
bake-off; the estimand is domain shift with the same circuit family as Paper B.

## Directory layout

| Path | Git | Purpose |
|------|-----|---------|
| [`README.md`](README.md) | yes | This index |
| [`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md) | yes | Citation paths for the manuscript |
| [`PLAN.md`](PLAN.md) | yes | Phased execution gates |
| [`config/circuit_registry.json`](config/circuit_registry.json) | yes | Paper-B–matched VQC / MLP specs |
| [`scripts/write_paperd_lock.py`](scripts/write_paperd_lock.py) | yes | Stage-2 lock manifest writer |
| [`scripts/verify_paperd_lock.py`](scripts/verify_paperd_lock.py) | yes | Recompute digests before labelled runs |
| [`results/README.md`](results/README.md) | yes | What may be promoted from raw runs |
| [`data/README.md`](data/README.md) | yes | Local BreakHis archive slot |
| `data/BreaKHis_v1.tar.gz` | **no** | Official archive (~4GB; gitignored) |
| `../../data/splits/breakhis/` | yes (case lists) | Case-ID folds; freeze before lock |
| `../../preregistration/paper_d_breakhis_protocol.md` | yes | Binding protocol |
| `../../preregistration/paper_d_lock.json` | yes (after Stage 2) | External lock manifest |
| `manuscript/` | **no** | Local authoring only |
| Raw GPU logs | **no** | Root `results/` (gitignored) |

## Scope

| Item | Value |
|------|--------|
| Data | BreakHis (40× / 100× / 200× / 400×), case-ID CV |
| Heads | Paper-B VQC (8q, 2L, angle-Y, linear) vs E2b-matched MLP |
| Regimes | Frozen backbone; last-block fine-tune |
| Primary metric | Case-balanced AUPRC; SEI ±0.01 |
| Optional source | Paper B frozen IDC folds (cross-domain secondary only) |

## Commands

```bash
# After case-ID manifests exist under data/splits/breakhis/
python papers/paper_d/scripts/write_paperd_lock.py

# Before the first labelled comparative run
python papers/paper_d/scripts/verify_paperd_lock.py
```

## Relation to other papers

| Paper | Relation |
|-------|----------|
| **B** | Same VQC family; D tests whether B's null survives external shift |
| **C** | Classical FOV / grouping track — parallel, not displaced |
| **A** | Tabular WBCD — unrelated cohort |

See [`../README.md`](../README.md) for the A/B/C/D series table.
