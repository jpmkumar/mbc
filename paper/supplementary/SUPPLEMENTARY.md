# Supplementary Material — Hyperparameters and Pilot Study Bridge

## Training Hyperparameters (enhanced mammography — primary result)

| Parameter | Value |
|-----------|-------|
| Config | `configs/mammo_enhanced.yaml` |
| Image size | 224 × 224 |
| Preprocessing | Grayscale + CLAHE (clip 2.0) |
| Batch size | 32 (AMP enabled) |
| Stage A epochs | 30 |
| Stage B epochs | 20 |
| Stage C epochs | 5 |
| LR classical | 1e-4 |
| LR quantum | 5e-5 |
| Malignant weight multiplier | 1.8 |
| Selection metric | Balanced accuracy |
| Checkpoint | `E3_hybrid_enhanced_seed42` |

## Training Hyperparameters (initial pipeline)

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Stage A epochs | 20 |
| Stage B epochs | 30 |
| LR quantum | 1e-4 |

## Data Splits

70% train / 15% validation / 15% test per modality, patient-level stratified.

## EQML Pilot Study (WBCD Tabular) — Bridge to Imaging

Prior feasibility on Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 features):

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| SVM (RBF) | 0.974 | 0.979 | 0.996 |
| VQC Standalone | 0.939 | 0.954 | 0.978 |
| Hybrid MLP+VQC | 0.877 | 0.908 | 0.938 |

Qubit sensitivity (pilot): 4 qubits → 92.1% acc; 6 qubits → 93.9%; 8 qubits → 94.7%.

Source: `/Users/muthu/ResTest/pilot_study/results/pilot_study_summary.json`

## Extended Ablations (E6, E7)

Run full experiment matrix without `--quick` flag:

```bash
python experiments/run_experiments.py
```

Experiments E6 (no tokens / no transformer) and E7 (qubit sweep 4/6/8) populate `results/experiment_matrix_summary.json`.
