# Paper B — freeze checklist (for release tag `paper-b-v1.0`)

When **every item below is checked**, tell the user Paper B is **frozen** and ready to:

```bash
git tag -a paper-b-v1.0 -m "Paper B histopath reproducibility release"
git push origin paper-b-v1.0
```

Citation URL for the manuscript:

```text
https://github.com/jpmkumar/mbc/tree/paper-b-v1.0/papers/paper_b
```

## Experiments (must be complete)

- [ ] Main E2/E2b/E3 five-fold — already complete
- [ ] Stage-B locked test v2 — five folds complete
- [ ] Stage-B nested v3 sensitivity — reported alongside v2
- [ ] Width ablation q4/q12 — Kaggle secondary folds 1–4 (or declared complete subset)
- [ ] Width ablation — server-primary matrix OR explicitly reported as secondary-only with Kaggle primary/secondary split documented
- [ ] Numerical convergence failures audited and reported (not silent reruns)
- [ ] No open confirmatory runs without disclosed reason

## Manuscript (must match results)

- [ ] `paper/main.tex` abstract/conclusion match final aggregated numbers
- [ ] Stage-B equivalence subsection added (if claiming v2/v3 outcomes)
- [ ] Multi-fold width results in text/tables (not fold-0 only)
- [ ] Figures match code (VQC diagram, main bars, optional Stage-B / width panels)
- [ ] `paper/references.bib` journal-only, compiles clean (`make -C paper pdf`)
- [ ] Authors, affiliations, ethics filled in

## Repository hygiene

- [ ] Sensitive/local files not committed (tokens, results/, AI drafts)
- [ ] `papers/paper_b/scripts/export_curated_metrics.py` run; snapshot matches manuscript tables
- [ ] Preregistration protocols match what was executed
- [ ] Kaggle vs server results **not pooled** in any table

## Current status (update when items complete)

Last reviewed: 2026-08-15

| Area | Status |
|------|--------|
| E2/E2b/E3 5-fold | Done |
| Stage-B v2/v3 | Done (5 folds) |
| Kaggle width q4 fold 3 | In progress |
| Kaggle width remaining | Pending |
| Server width matrix | Partial (q8 fold 1 only) |
| Manuscript Stage-B section | Not yet in main.tex |
| Release tag | **Not frozen** |

---

# Paper A — freeze checklist (for `paper-a-v1.0`)

Separate tag when tabular confirmatory runs exist.

- [ ] Confirmatory WBCD CV pipeline implemented under `papers/paper_a/`
- [ ] Matched MLP + VQC + classical baselines with proper CV
- [ ] Manuscript drafted
- [ ] `papers/paper_a/CODE_AVAILABILITY.md` updated with final paths

**Paper A status:** Not frozen (pilot metrics only).
