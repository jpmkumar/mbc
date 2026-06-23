# Publication Package — MBC Hybrid Quantum Framework

Everything needed to compile the IEEE Access manuscript and reproduce figures/tables.

## Quick commands

```bash
# 1. Regenerate all figures and LaTeX tables from canonical metrics
python scripts/generate_publication.py

# 2. Compile PDF (from repo root)
make -C paper pdf

# 3. Or manually
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Canonical results source

Edit **`publication/publication_metrics.json`** when new experiments finish, then rerun `generate_publication.py`.

| Experiment | Test balanced acc | Role in paper |
|------------|-------------------|---------------|
| Stage A baseline | 61.8% | Initial pipeline |
| Benedetti Stage B | 60.2% | VQC ablation (old head) |
| **Stage A enhanced** | **70.1%** | **Primary result** |
| Stage B enhanced | 69.0% | VQC ablation (CLAHE + Benedetti) |

## Files for manuscript

### LaTeX
| Path | Purpose |
|------|---------|
| `paper/main.tex` | Main IEEE Access draft |
| `paper/references.bib` | Bibliography |
| `publication/tables/publication_tables.tex` | Tables I–III (auto-generated) |

### Figures (300 DPI PNG)
| File | Label | Section |
|------|-------|---------|
| `fig1_architecture.png` | fig:arch | Method |
| `fig_training_stages.png` | fig:stages | Method |
| `fig_mammo_metrics_comparison.png` | fig:metrics | Results |
| `fig_confusion_matrices.png` | fig:confusion | Results |
| `fig_dataset_splits.png` | fig:dataset | Setup |
| `fig_pilot_vs_imaging.png` | fig:pilot | Discussion |
| `case_study_0_mammo.png` | fig:case | Explainability |

### Supporting docs
| Path | Purpose |
|------|---------|
| `publication/PUBLICATION_RESULTS.md` | Copy-ready result paragraphs |
| `publication/FIGURE_CAPTIONS.md` | All figure captions |
| `paper/ETHICS_STATEMENT.md` | Data ethics (paste into submission) |
| `paper/SUBMISSION_CHECKLIST.md` | Pre-submission checklist |
| `DATA_SCALE.md` | Dataset scale vs 100k+ papers |
| `paper/supplementary/SUPPLEMENTARY.md` | Hyperparameters |

## Before submission — fill in

1. **`paper/main.tex`** — author names, affiliations, emails
2. **`publication/AUTHORS.md`** — ORCID, funding, COI
3. Download **real BUSI** ultrasound (replace synthetic 80-image set)
4. Multi-seed runs (optional): seeds 42, 43, 44

## GitHub

Repository: https://github.com/jpmkumar/mbc

Data are not redistributed in Git; use `data/download/` scripts per `DATA_SCALE.md`.

## Citation (draft)

```bibtex
@article{mbc_hybrid_quantum_2025,
  title={Modality-Level Generalized Hybrid Quantum Framework for Breast Cancer Classification},
  author={TBD},
  journal={IEEE Access},
  year={2025},
  note={Code: https://github.com/jpmkumar/mbc}
}
```
