# Paper 1 Submission Checklist

Target venue: **IEEE Access** (rolling submission)

## Manuscript
- [x] LaTeX draft: `paper/main.tex`
- [x] Bibliography: `paper/references.bib`
- [x] Venue guide: `paper/VENUE.md`
- [x] Results table populated from `results/experiment_matrix_summary.json` (synthetic validation run)
- [ ] Update Table I with **real-dataset** numbers before final submission
- [ ] Author affiliations and ORCID
- [x] Supplementary material (`paper/supplementary/SUPPLEMENTARY.md`)

## Figures (minimum 6)
- [x] Fig 1: Architecture (`figures/fig1_architecture.png`)
- [x] VQC circuit diagram (`figures/vqc_circuit_diagram.txt`)
- [x] Fig 3: ROC curves (`figures/fig3_roc_curves.png`)
- [x] Fig 4: Cross-modality LOMO (`figures/fig4_cross_modality_lomo.png`)
- [x] Fig 5: Ablation (`figures/fig5_ablation.png`)
- [x] Case study panels (`figures/case_study_*.png`)

## Code & Data
- [x] Reproducible codebase with config-driven experiments
- [x] Dataset download guide and synthetic fallback
- [x] `requirements.txt`
- [x] End-to-end runner: `experiments/run_all.py`

## Ethics
- [x] Data use statement (`paper/ETHICS_STATEMENT.md`)
- [ ] Conflicts of interest declaration (fill at submission)

## Pre-submission
- [ ] Internal review by co-author/colleague
- [ ] Plagiarism check
- [ ] Compile LaTeX PDF: `cd paper && pdflatex main && bibtex main && pdflatex main`
- [ ] Replace synthetic data with CBIS-DDSM + BUSI + thermography datasets
- [ ] Run full experiments: `python experiments/run_experiments.py` (no `--quick`)

## Submit
- [ ] IEEE Access Author Portal: https://ieeeaccess.ieee.org/
