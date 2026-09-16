# BreakHis case-ID splits (Paper D)

Case-level fold membership will be **committed** here once the pre-outcome
audit finishes. Patch- or image-level manifests stay local / rebuildable.

Until the first freeze:

1. Do not invent folds ad hoc in notebooks.
2. Do not run labelled head comparisons.
3. After case lists exist, run
   `python papers/paper_d/scripts/write_paperd_lock.py` so their digests enter
   `preregistration/paper_d_lock.json`.

Expected layout (when frozen):

```text
data/splits/breakhis/
  README.md          (this file)
  fold_0/…_cases.csv
  fold_1/…
  …
  split_stats.json
```

Grouping unit: patient / case identifier. Image-level random CV is forbidden
for Paper D's primary analysis — see
`preregistration/paper_d_breakhis_protocol.md`.
