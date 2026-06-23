# Figure captions (IEEE Access)

Use these in `paper/main.tex` or submission portal.

| File | Suggested caption |
|------|-------------------|
| `fig1_architecture.png` | Modality-Level Generalized Hybrid Quantum Framework. Input from any single modality receives a learnable token, passes through EfficientNet-B0 and a Transformer encoder, is compressed to eight dimensions, and is classified by a classical head (Stage A) or variational quantum circuit head (Stage B). |
| `fig_training_stages.png` | Two-stage hybrid training protocol: Stage A trains the classical head; Stage B freezes the backbone and trains the VQC head; optional Stage C performs joint fine-tuning. |
| `fig_mammo_metrics_comparison.png` | CBIS-DDSM test-set comparison ($n=445$): (a) balanced accuracy, recall, and precision; (b) F1 and AUC across baseline and enhanced configurations. |
| `fig_confusion_matrices.png` | Confusion matrices on CBIS-DDSM test set: (a) enhanced Stage A (recommended model); (b) enhanced Stage B VQC ablation. |
| `fig_dataset_splits.png` | CBIS-DDSM mammography ROI dataset: train/validation/test split sizes (70/15/15). |
| `fig_pilot_vs_imaging.png` | Comparison of EQML pilot study on WBCD tabular data ($n=569$) versus CBIS-DDSM mammography imaging ($n=445$ test). Tasks differ in modality and difficulty. |
| `case_study_0_mammo.png` | Explainability case study on mammography: input image, Grad-CAM saliency, Transformer attention, and prediction summary. |

Individual confusion matrices: `fig_confusion_<experiment_id>.png`.
