# BreakHis case-ID splits (Paper D)

Case-level fold membership is **committed** here. Image-level inventory is
rebuildable from the archive and stays local/gitignored.

## Frozen partition

- Splitter: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- Group unit: BreakHis case folder id (82 cases)
- Archive: `papers/paper_d/data/BreaKHis_v1.tar.gz` (gitignored)
- Archive SHA-256: `d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640`
- Builder: `papers/paper_d/scripts/build_breakhis_case_folds.py`
- Stats: [`split_stats.json`](split_stats.json)

Do not regenerate after confirmatory runs start. Load these lists.

```text
data/splits/breakhis/
  README.md
  split_stats.json
  fold_0/train_cases.csv
  fold_0/test_cases.csv
  ...
  fold_4/...
```

Grouping unit: patient / case identifier. Image-level random CV is forbidden
for Paper D's primary analysis — see
`preregistration/paper_d_breakhis_protocol.md`.
