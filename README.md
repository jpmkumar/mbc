# Modality-Level Generalized Hybrid Quantum Framework for Breast Cancer Classification

Unified cross-modality breast cancer classification using EfficientNet-B0, Transformer-based modality-invariant learning, and an optional variational quantum circuit (VQC) head with multi-method explainability.

**Publication package:** [`publication/README.md`](publication/README.md) | **Paper PDF:** `make -C paper pdf`

## Primary result (CBIS-DDSM mammography)

| Metric | Enhanced Stage A |
|--------|------------------|
| Balanced accuracy | **70.1%** |
| Malignant recall | **84.7%** |
| AUC | **0.763** |
| Test set | n=445 |

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: CBIS-DDSM → data/processed/mammo/
python data/download/setup_datasets.py

# Train enhanced pipeline (Colab: see GITHUB.md)
python experiments/run_training.py \
  --config configs/mammo_enhanced.yaml \
  --experiment hybrid --modality mammo --stage all

# Histopathology IDC patches (Colab: see COLAB_HISTOPATH.md)
python scripts/train_histopath_cv.py --fold 0 --experiment E2

# Publication figures + LaTeX tables
python scripts/generate_publication.py
make -C paper pdf
```

## Project Structure

```
configs/              YAML configs (default, mammo_enhanced, benedetti_vqc)
data/download/        CBIS-DDSM, BUSI download scripts
src/                  Models, training, data pipeline
experiments/          run_training.py, generate_figures.py
publication/          Metrics JSON, tables, publication guide
figures/              Paper-ready PNG figures (300 DPI)
paper/                IEEE Access LaTeX manuscript
scripts/              analyze_results, generate_publication
```

## Datasets

| Modality | Dataset | Status in repo |
|----------|---------|----------------|
| Mammography | [CBIS-DDSM](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY) | **Real (~2966 ROIs)** |
| Histopathology | Kaggle Breast Histopathology (IDC patches) | **Scripts + 5-fold CV** — [`KAGGLE_HISTOPATH.md`](KAGGLE_HISTOPATH.md) |
| Ultrasound | [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | Synthetic placeholder |
| Thermography | Kaggle thermo DB | Synthetic placeholder |

See [`DATA_SCALE.md`](DATA_SCALE.md) for scale comparison with 100k+ image papers.

## Architecture

EfficientNet-B0 → Modality Transformer → Compression (8-D) → Classical head (Stage A) or VQC head (Stage B)

## Paper

- Manuscript: [`paper/main.tex`](paper/main.tex)
- Checklist: [`paper/SUBMISSION_CHECKLIST.md`](paper/SUBMISSION_CHECKLIST.md)
- GitHub: https://github.com/jpmkumar/mbc
