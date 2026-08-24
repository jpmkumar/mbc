# Modality-Level Generalized Hybrid Quantum Framework for Breast Cancer Classification

Unified cross-modality breast cancer classification using EfficientNet-B0, Transformer-based modality-invariant learning, and an optional variational quantum circuit (VQC) head with multi-method explainability.

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
```

## Project Structure

```
configs/              YAML configs (default, mammo_enhanced, benedetti_vqc)
data/download/        CBIS-DDSM, BUSI download scripts
src/                  Models, training, data pipeline
experiments/          run_training.py, generate_figures.py
scripts/              Training, evaluation, and analysis entry points
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

## Scope

This repository is a code release. Manuscript sources, figure artwork, and
submission material are maintained separately.

- GitHub: https://github.com/jpmkumar/mbc
