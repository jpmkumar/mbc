# Paper C — deep-learning IDC histopathology benchmark

**Canonical code path:** `papers/paper_c/`. Manuscript sources and submission
material are maintained outside this repository.

**Data:** (1) Kaggle Breast Histopathology Images (IDC) — 277,524 patches of
50×50 px at nominal 40× under 279 public case identifiers; and (2) BCSS — 151
breast-cancer slides/ROIs with patient/site identifiers and pixel-level tissue
masks. The IDC cohort is shared with Paper B; the scientific question and BCSS
replication are unique to Paper C.

**Scope:** modern deep learning and pathology foundation models. **No quantum
component** — hybrid QML on this cohort is Paper B ([`../paper_b/`](../paper_b/)).

**Shared folds:** Paper C reuses the frozen case-ID-grouped fold assignments in
`data/splits/histopath_kaggle/folds/fold_*/`, the same ones Paper B uses, so the
two papers remain directly comparable on identical public identifiers. The
archive does not prove that these identifiers are independent patients.

**Working claim:** grouping protocol and physical field of view can change both
the measured performance and ranking of pathology foundation models. Paper C
quantifies random-patch versus case-ID-grouped optimism, tests isolated 50×50
patches against approximately pretraining-matched 9×9 context, and replicates
the context mechanism on institution-held-out BCSS.

## Directory layout

| Path | Git | Purpose |
|------|-----|---------|
| [`PLAN.md`](PLAN.md) | yes | Phased execution plan with gates and deliverables |
| [`VALIDATION.md`](VALIDATION.md) | yes | Paired OOF design, case-cluster inference, calibration policy |
| [`INDEPENDENCE.md`](INDEPENDENCE.md) | yes | Why Paper C never depends on Paper B; overlap disclosure |
| [`SERVER_PAPER_C.md`](SERVER_PAPER_C.md) | yes | Campus RTX A4000 runbook and order of work |
| [`scripts/audit_patient_grouping.py`](scripts/audit_patient_grouping.py) | yes | Phase 0 archive audit (162 vs 279) |
| [`scripts/audit_idc_context.py`](scripts/audit_idc_context.py) | yes | Coordinate spacing and complete-neighbourhood audit |
| [`config/model_registry.json`](config/model_registry.json) | yes | Revision-pinned encoder panel, dimensions, licences and A4000 batches |
| [`../../preregistration/paper_c_protocol.md`](../../preregistration/paper_c_protocol.md) | yes | Draft confirmatory protocol; must be hash-locked before labelled comparisons |
| [`scripts/candidates.txt`](scripts/candidates.txt) | yes | Candidate reference keys and DOIs, grouped by theme |
| [`scripts/verify_crossref.py`](scripts/verify_crossref.py) | yes | Crossref admissibility check + BibTeX emitter |
| [`scripts/build_bib.py`](scripts/build_bib.py) | yes | Assembles `dlReferences.bib` from verified entries |
| `results/` | selective | Named audit/qualification reports are tracked; raw outputs are ignored |

## Reference policy

The manuscript bibliography admits peer-reviewed journal articles only. No
preprints, conference or workshop proceedings, LNCS chapters, or dataset cards.

Every entry in `dlReferences.bib` was retrieved from the Crossref REST API and
required to report `type=journal-article` with a journal title, a volume, and a
page range or article number. Two entries carry a manually supplied volume where
Crossref had not populated the field for a recent issue; both are marked inline.

## Commands

```bash
# Re-verify every DOI and regenerate the bibliography
python3 scripts/verify_crossref.py scripts/candidates.txt --bibtex
python3 scripts/build_bib.py

# Policy checks before any bibliography commit
rg -o "^@(\w+)" -r '$1' dlReferences.bib | sort | uniq -c   # expect only "article"
rg -in "arxiv|preprint" dlReferences.bib                    # expect no matches
```

## Adding a reference

1. Append `<bibkey> <DOI>` to the correct thematic section of `scripts/candidates.txt`.
2. Run the two commands above.
3. If the entry is rejected, fix the DOI or drop it — never hand-write metadata.
   A search-result snippet is not verification; a wrong DOI silently resolves to a
   different paper.

See [`../README.md`](../README.md) for the Paper A / B / C split.
