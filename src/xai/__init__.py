"""Explainability: Grad-CAM, attention maps, SHAP."""

from .gradcam import GradCAM
from .attention_viz import plot_attention_heatmap
from .shap_analysis import compute_shap_features, compute_vqc_gate_importance
from .case_study import generate_case_study_figure

__all__ = [
    "GradCAM",
    "plot_attention_heatmap",
    "compute_shap_features",
    "compute_vqc_gate_importance",
    "generate_case_study_figure",
]
