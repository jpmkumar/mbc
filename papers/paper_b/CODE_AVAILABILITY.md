# Code availability — Paper B (histopathology IDC)

Use these paths when citing the GitHub repository in **Paper B** (Kaggle IDC
patch classification, case-ID-grouped CV, null / equivalence result).

The cohort is **277,524 patches under 279 public case identifiers**. The public
derivative does not establish that those identifiers are 279 independent
patients, so grouping and cross-validation are described here by case ID, matching
the manuscript.

## Repository

| Item | Value |
|------|-------|
| Repository | https://github.com/jpmkumar/mbc |
| Paper B workspace | [`papers/paper_b/`](.) |
| Development branch | `docs/histopath-writing-q1-guidelines` |

**Stable URL for submission** (after you tag a release):

```text
https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_b
```

Example:

```text
https://github.com/jpmkumar/mbc/tree/paper-b-v1.0/papers/paper_b
```

## What belongs to Paper B

| Path | Role |
|------|------|
| [`papers/paper_b/`](.) | Paper B index and curated exports |
| [`scripts/train_histopath_cv.py`](../../scripts/train_histopath_cv.py) | Main 5-fold training |
| [`scripts/run_histopath_width_kaggle.py`](../../scripts/run_histopath_width_kaggle.py) | Width ablation (Kaggle) |
| [`scripts/run_histopath_width_server.py`](../../scripts/run_histopath_width_server.py) | Width ablation (server) |
| [`scripts/evaluate_vqc_stage_b_locked.py`](../../scripts/evaluate_vqc_stage_b_locked.py) | Stage-B locked test |
| [`configs/histopath.yaml`](../../configs/histopath.yaml) | Fixed training bundle |
| [`data/download/split_histopath_archive.py`](../../data/download/split_histopath_archive.py) | Case-ID-grouped splits |
| [`preregistration/`](../../preregistration/) | Stage-B and width protocols |
| [`KAGGLE_HISTOPATH.md`](../../KAGGLE_HISTOPATH.md) | Primary Kaggle runbook |
| [`KAGGLE_HISTOPATH_WIDTH_ABLATION.md`](../../KAGGLE_HISTOPATH_WIDTH_ABLATION.md) | Width ablation runbook |
| [`SERVER_HISTOPATH.md`](../../SERVER_HISTOPATH.md) | Physical GPU server runbook |

## What is **not** Paper B

| Path | Belongs to |
|------|------------|
| [`papers/paper_a/`](../../papers/paper_a/) | **Paper A** (tabular WBCD) |

## Suggested Code Availability sentence (Paper B)

> Code for the histopathology experiments is available at
> https://github.com/jpmkumar/mbc (directory `papers/paper_b/`, tag `<TAG>`).
> Training scripts, case-ID-grouped split generation, and the preregistered
> protocols are included. Patch images are obtained from the Kaggle Breast
> Histopathology Images dataset; they are not redistributed in the repository.

Replace `<TAG>` with your release tag at submission time.

## Reproduce (after dataset download)

```bash
git clone https://github.com/jpmkumar/mbc.git
cd mbc
pip install -r requirements.txt

# See KAGGLE_HISTOPATH.md or SERVER_HISTOPATH.md for full workflow
python scripts/train_histopath_cv.py --fold 0 --experiment E3

# Refresh curated metrics for tables
python papers/paper_b/scripts/export_curated_metrics.py
```
