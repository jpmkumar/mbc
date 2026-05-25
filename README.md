# Modality-Level Generalized Hybrid Quantum Framework for Breast Cancer Classification

Unified cross-modality breast cancer classification using EfficientNet-B0, Transformer-based modality-invariant learning, and a variational quantum circuit (VQC) head with multi-method explainability (SHAP, Grad-CAM, Attention).

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic data (or place real data in data/processed/)
python data/download/generate_synthetic.py --samples 50
python data/download/setup_datasets.py

# 3. Train hybrid model (quick mode for validation)
python experiments/run_training.py --experiment hybrid --quick

# 4. Run experiment matrix
python experiments/run_experiments.py --quick

# 5. Generate XAI outputs and figures
python experiments/run_xai.py
python experiments/generate_figures.py
python experiments/generate_vqc_diagram.py
```

## Project Structure

```
configs/           Experiment YAML configs
data/
  download/        Dataset scripts and synthetic generator
  processed/       Images organized by modality/label
  splits/          Train/val/test CSV manifests
src/
  data/            Loaders, transforms, splits
  models/          Encoder, transformer, VQC, hybrid model
  train/           Two-stage training loop
  eval/            Metrics and experiment matrix
  xai/             Grad-CAM, attention, SHAP
experiments/       Runnable scripts
results/           Metrics JSON and checkpoints
figures/           Paper-ready figures
paper/             IEEE Access manuscript (LaTeX)
```

## Target Venue

**Primary:** [IEEE Access](https://ieeeaccess.ieee.org/) — see [`paper/VENUE.md`](paper/VENUE.md)

## Datasets

| Modality | Dataset | Location |
|----------|---------|----------|
| Mammography | CBIS-DDSM | TCIA (registration required) |
| Ultrasound | BUSI | Kaggle |
| Thermography | Breast Thermography DB | KCloud |

Run `python data/download/setup_datasets.py --instructions-only` for download guide.

**Data protocol:** 70/15/15 patient-level stratified split; benign=0, malignant=1; 224×224 RGB input.

## Architecture

See [`initial_paper_architecture.png`](initial_paper_architecture.png):

Input → Modality Token → EfficientNet-B0 → Transformer → Compression (2048→128→32→8) → Angle Encoding → VQC → Benign/Malignant → XAI

## Prior Work

Built on EQML pilot study (`/Users/muthu/ResTest/pilot_study/`) validating hybrid MLP+VQC on WBCD tabular data (~93–95% accuracy, 4–8 qubit sweep).

## Paper

Manuscript draft: [`paper/main.tex`](paper/main.tex)
