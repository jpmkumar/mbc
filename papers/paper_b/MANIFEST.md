# Paper B path manifest

Canonical locations — prefer these over creating parallel copies.

## Manuscript (canonical LaTeX)

| Role | Path |
|------|------|
| Main TeX | `paper/main.tex` |
| Bibliography | `paper/references.bib` |
| Makefile | `paper/Makefile` |
| Ethics / checklist | `paper/ETHICS_STATEMENT.md`, `paper/SUBMISSION_CHECKLIST.md` |
| Writing plan | `paper/PAPER_WRITING_PLAN.md` |
| Guidelines | `paper/PAPER_WRITING_GUIDELINES.md` |

## Figures

| Role | Path |
|------|------|
| Shared export dir (used by `\graphicspath`) | `figures/` |
| VQC head TikZ master | `figures/Fig02_VQC_Head.tikz` |
| Build script | `figures/build_figures.sh` |
| Paper-B staging / submission copies | `papers/paper_b/figures/export/` |
| Paper-B-only sources (if any) | `papers/paper_b/figures/source/` |

## Curated results (Paper B workspace)

| Role | Path |
|------|------|
| Export script | `papers/paper_b/scripts/export_curated_metrics.py` |
| Curated JSON snapshot | `papers/paper_b/results/curated_metrics.json` |
| Export log | `papers/paper_b/results/export_log.txt` |

## Raw experiments (read-only for agents)

| Role | Path |
|------|------|
| Histopath CV summary | `results/histopath/cv_summary.json` |
| Stage-B cross-fold | `results/histopath/vqc_stage_b_crossfold_v2_final.json` |
| Experiment log | `results/histopath_experiment_log.md` |
| Width bundles | `results/histopath/width_q*_fold*_report.json` |

## Protocols

| Role | Path |
|------|------|
| Stage-B v2 | `preregistration/stage_b_protocol_v2.md` |
| Stage-B v3 | `preregistration/stage_b_protocol_v3_nested_secondary.md` |
| Width (Kaggle) | `preregistration/histopath_vqc_width_protocol.md` |
| Width (server) | `preregistration/histopath_vqc_width_server_protocol.md` |
