# Code availability — Paper B (histopathology IDC)

Use these paths when citing the GitHub repository in **Paper B** (Kaggle IDC
patch classification, patient-level CV, null / equivalence result).

## Repository

| Item | Value |
|------|-------|
| Repository | https://github.com/jpmkumar/mbc |
| Paper B workspace | [`papers/paper_b/`](.) |
| Manuscript (LaTeX) | [`paper/main.tex`](../../paper/main.tex) |
| Development branch | `docs/histopath-writing-q1-guidelines` |

**Stable URL for submission** (after you tag a release):

```text
https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_b
```

Example:

```text
https://github.com/jpmkumar/mbc/tree/paper-b-v1.0/papers/paper_b
https://github.com/jpmkumar/mbc/tree/paper-b-v1.0/paper/main.tex
```

## What belongs to Paper B

| Path | Role |
|------|------|
| [`papers/paper_b/`](.) | Paper B index, manifest, curated exports |
| [`paper/`](../../paper/) | LaTeX manuscript, bibliography, checklist |
| [`figures/`](../../figures/) | Publication figures |
| [`scripts/train_histopath_cv.py`](../../scripts/train_histopath_cv.py) | Main 5-fold training |
| [`scripts/run_histopath_width_kaggle.py`](../../scripts/run_histopath_width_kaggle.py) | Width ablation (Kaggle) |
| [`scripts/run_histopath_width_server.py`](../../scripts/run_histopath_width_server.py) | Width ablation (server) |
| [`scripts/evaluate_vqc_stage_b_locked.py`](../../scripts/evaluate_vqc_stage_b_locked.py) | Stage-B locked test |
| [`configs/histopath.yaml`](../../configs/histopath.yaml) | Fixed training bundle |
| [`data/download/split_histopath_archive.py`](../../data/download/split_histopath_archive.py) | Patient-level splits |
| [`preregistration/`](../../preregistration/) | Stage-B and width protocols |
| [`KAGGLE_HISTOPATH.md`](../../KAGGLE_HISTOPATH.md) | Primary Kaggle runbook |
| [`KAGGLE_HISTOPATH_WIDTH_ABLATION.md`](../../KAGGLE_HISTOPATH_WIDTH_ABLATION.md) | Width ablation runbook |
| [`SERVER_HISTOPATH.md`](../../SERVER_HISTOPATH.md) | Physical GPU server runbook |

## What is **not** Paper B

| Path | Belongs to |
|------|------------|
| [`papers/paper_a/`](../../papers/paper_a/) | **Paper A** (tabular WBCD) |
| [`publication/publication_metrics.json`](../../publication/publication_metrics.json) | Legacy mammo + WBCD pilot (mixed) |

## Suggested Code Availability sentence (Paper B)

> Code for the histopathology experiments is available at
> https://github.com/jpmkumar/mbc (directories `papers/paper_b/` and `paper/`,
> tag `<TAG>`). Training scripts, patient-level split generation, preregistered
> protocols, and figure sources are included. Patch images are obtained from
> the Kaggle Breast Histopathology Images dataset; they are not redistributed in
> the repository.

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

## Compile manuscript

```bash
make -C paper pdf
```
