# Code availability — Paper A (tabular WBCD)

Use these paths when citing the GitHub repository in **Paper A** (Wisconsin
Breast Cancer Diagnostic tabular hybrid QML).

## Repository

| Item | Value |
|------|-------|
| Repository | https://github.com/jpmkumar/mbc |
| Paper A workspace | [`papers/paper_a/`](.) |
| Development branch | `docs/histopath-writing-q1-guidelines` |

**Stable URL for submission** (after you tag a release):

```text
https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_a
```

Example after tagging: `git tag paper-a-v1.0 && git push origin paper-a-v1.0`

→ `https://github.com/jpmkumar/mbc/tree/paper-a-v1.0/papers/paper_a`

## What belongs to Paper A

| Path | Role |
|------|------|
| [`papers/paper_a/`](.) | Paper A index and curated exports |
| [`src/models/vqc.py`](../../src/models/vqc.py) | Shared VQC head implementation |
| [`src/models/hybrid_model.py`](../../src/models/hybrid_model.py) | Shared hybrid model |
| `papers/paper_a/scripts/` | Paper-A-specific export scripts (when added) |

Confirmatory WBCD training scripts will live under `papers/paper_a/scripts/` or
`experiments/wbcd/` when implemented — not under `papers/paper_b/`.

## What is **not** Paper A

| Path | Belongs to |
|------|------------|
| [`papers/paper_b/`](../../papers/paper_b/) | **Paper B** |
| [`scripts/train_histopath_cv.py`](../../scripts/train_histopath_cv.py) | **Paper B** |
| [`KAGGLE_HISTOPATH*.md`](../../KAGGLE_HISTOPATH.md) | **Paper B** |

## Suggested Data Availability sentence (Paper A)

> Code for the tabular Wisconsin Breast Cancer experiments is available at
> https://github.com/jpmkumar/mbc (directory `papers/paper_a/`, tag `<TAG>`).
> Shared quantum-classical model code is in `src/`. The diagnostic dataset is
> available from the UCI Machine Learning Repository.

Replace `<TAG>` with your release tag at submission time.

## Reproduce pilot metrics (reference)

```bash
git clone https://github.com/jpmkumar/mbc.git
cd mbc
python papers/paper_a/scripts/export_curated_metrics.py
# Writes papers/paper_a/results/
```

Full confirmatory WBCD pipeline: **pending** — see [`README.md`](README.md).
