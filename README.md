# Hybrid classical–quantum breast cancer classification

Code release for a set of studies on breast cancer classification with hybrid
classical–quantum models: EfficientNet-B0 features, a Transformer encoder, a
compressed bottleneck, and either a classical head or a variational quantum
circuit (VQC) head, with explainability throughout.

This repository is code only. Manuscript sources are maintained separately.

**GitHub:** https://github.com/jpmkumar/mbc

## Journal papers (code paths for citation)

This repository supports **two papers**. Cite the **directory + release tag**, not the repo root alone.

| Paper | Topic | Code path | Availability |
|-------|-------|-----------|--------------|
| **A** | Wisconsin Breast Cancer **tabular** hybrid QML | [`papers/paper_a/`](papers/paper_a/) | [`papers/paper_a/CODE_AVAILABILITY.md`](papers/paper_a/CODE_AVAILABILITY.md) |
| **B** | IDC **histopathology** patches, patient-level CV | [`papers/paper_b/`](papers/paper_b/) | [`papers/paper_b/CODE_AVAILABILITY.md`](papers/paper_b/CODE_AVAILABILITY.md) |

Release tags freeze the URL a paper cites. `paper-b-v1.0` is the frozen snapshot
underlying Paper B. Details: [`papers/README.md`](papers/README.md).

## Paper B — hybrid quantum stages in IDC histopathology

The study behind the `paper-b-v1.0` tag. Kaggle Breast Histopathology Images,
277,524 patches under 279 public case identifiers, evaluated with case-ID-grouped
five-fold cross-validation against capacity-controlled classical baselines. The
result is a null one: the bolt-on variational quantum head is not selected by
validation and does not surpass its classical control.

- Code index: [`papers/paper_b/`](papers/paper_b/)
- Paths to cite: [`papers/paper_b/CODE_AVAILABILITY.md`](papers/paper_b/CODE_AVAILABILITY.md)
- Fold membership: [`data/splits/README.md`](data/splits/README.md) — load it, do not regenerate it
- Protocols declared before the confirmatory runs: [`preregistration/`](preregistration/)

## Paper A — tabular WBCD hybrid QML (separate publication)

- Code index: [`papers/paper_a/`](papers/paper_a/)

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
preregistration/      Protocols declared before the confirmatory runs
papers/               Per-paper code index and availability paths
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

## Earlier cross-modality work (not part of Paper A or Paper B)

The repository began as a cross-modality mammography study, and that code and its
results are still here. **These numbers belong to a different dataset, a
different modality, and a different evaluation protocol from either paper. They
are not comparable with Paper B's histopathology results and must not be read as
them.**

CBIS-DDSM mammography, enhanced Stage A, single held-out test set of n=445:

| Metric | Value |
|--------|-------|
| Balanced accuracy | 70.1% |
| Malignant recall | 84.7% |
| AUC | 0.763 |

## Scope

This repository is a code release. Manuscript sources, figure artwork, and
submission material are maintained separately.
