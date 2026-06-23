# Paper 1 Submission Checklist

Target venue: **IEEE Access** (rolling submission)

## Manuscript
- [x] LaTeX draft: `paper/main.tex` (updated with CBIS-DDSM results)
- [x] Bibliography: `paper/references.bib` (incl. Benedetti VQC)
- [x] Tables I–III: `publication/tables/publication_tables.tex`
- [x] Abstract with real numbers (70.1% balanced acc, 84.7% recall)
- [ ] Author affiliations and ORCID → `publication/AUTHORS.md`
- [x] Supplementary: `paper/supplementary/SUPPLEMENTARY.md`
- [x] Build: `make -C paper pdf`

## Figures (publication-ready, 300 DPI)
- [x] Fig 1: Architecture (`figures/fig1_architecture.png`)
- [x] Fig 2: Training stages (`figures/fig_training_stages.png`)
- [x] Fig 3: Metric comparison (`figures/fig_mammo_metrics_comparison.png`)
- [x] Fig 4: Confusion matrices (`figures/fig_confusion_matrices.png`)
- [x] Fig 5: Dataset splits (`figures/fig_dataset_splits.png`)
- [x] Fig 6: Pilot vs imaging (`figures/fig_pilot_vs_imaging.png`)
- [x] Fig 7: Case study XAI (`figures/case_study_0_mammo.png`)
- [x] Captions: `publication/FIGURE_CAPTIONS.md`

## Results & reproducibility
- [x] Canonical metrics: `publication/publication_metrics.json`
- [x] Generate script: `scripts/generate_publication.py`
- [x] Sync script: `scripts/sync_publication_metrics.py`
- [x] CSV export: `publication/tables/mammo_results.csv`
- [x] Summary: `publication/PUBLICATION_RESULTS.md`
- [x] Publication guide: `publication/README.md`

## Code & Data
- [x] Reproducible codebase + configs (`mammo_enhanced.yaml`)
- [x] CBIS-DDSM download guide (`data/download/CBIS-DDSM.md`)
- [x] `requirements.txt`
- [x] Colab workflow (`GITHUB.md`, `COLAB_RESUME.md`)
- [ ] Replace synthetic US/thermo with real BUSI + thermography
- [ ] Multi-seed runs (43, 44) optional

## Ethics
- [x] Data use statement (`paper/ETHICS_STATEMENT.md`)
- [ ] Conflicts of interest declaration (fill at submission)

## Pre-submission
- [ ] Internal review by co-author
- [ ] Plagiarism check
- [ ] Compile PDF: `make -C paper pdf`
- [ ] Verify all figure paths render

## Submit
- [ ] IEEE Access Author Portal: https://ieeeaccess.ieee.org/
