# Literature Review Gap Statement

**Research gap:** No prior work combines modality-level generalized breast imaging (mammography, ultrasound, thermography) with hybrid variational quantum classification and multi-method explainability (SHAP + Grad-CAM + Attention) **without requiring paired multimodal patient datasets**.

## Related Work Buckets

### 1. Breast Cancer Imaging AI
- Modality-specific CNN/ViT classifiers for mammography (CBIS-DDSM), ultrasound (BUSI), thermography
- Deep learning achieves strong per-modality performance but lacks cross-modality generalization

### 2. Cross-Modality / Domain Generalization
- Modality tokens and unified encoders (e.g., multimodal transformers)
- Domain adaptation and federated learning for heterogeneous medical imaging
- Gap: existing methods typically require paired or co-registered multimodal data

### 3. Hybrid Quantum-Classical Classification
- PennyLane/TorchLayer VQC for medical and general ML
- Quantum transfer learning (Azevedo et al., 2022; Mari et al., 2020)
- Prior EQML pilot: WBCD tabular hybrid MLP+VQC feasibility (~93–95% accuracy)

### 4. Explainable Medical AI
- Grad-CAM for CNN spatial attribution
- Transformer attention visualization
- SHAP for feature and gate-level importance in hybrid models

## Key References (starter bibliography)

1. Tan, M. & Le, Q. EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019.
2. Dosovitskiy, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition. ICLR 2021.
3. Schuld, M. & Killoran, N. Quantum Machine Learning in Feature Hilbert Spaces. PRL 2019.
4. Havlíček, V. et al. Supervised learning with quantum-enhanced feature spaces. Nature 2019.
5. Selvaraju, R. et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.
6. Lundberg, S. & Lee, S. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
7. Al-Dhabyani, W. et al. Dataset of breast ultrasound images. Data in Brief 2020 (BUSI).
8. Lee, R.S. et al. CBIS-DDSM: Curated Breast Imaging Subset of DDSM. Cancer Imaging Archive 2017.
9. Azevedo, L. et al. Quantum transfer learning for breast cancer detection. 2022.
10. Mari, R. et al. Transfer learning in hybrid classical-quantum neural networks. Quantum 2020.

## Contribution Alignment

1. Unified cross-modality breast cancer learning without paired datasets
2. Transformer-based modality-invariant representation learning with learnable modality tokens
3. Hybrid classical–quantum classification using angle-encoded VQC (4–8 qubits)
4. Explainable AI integration: SHAP + Grad-CAM + Attention Maps
