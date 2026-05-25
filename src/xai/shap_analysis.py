"""SHAP analysis for compressed features and VQC gate importance."""

import json
from pathlib import Path

import numpy as np
import shap
import torch


def compute_shap_features(model, loader, device: str, max_samples: int = 50) -> dict:
    """SHAP on compressed 8-d features using a wrapper classifier."""
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(features_list) >= max_samples:
                break
            images = batch["image"].to(device)
            labels = batch["label"]
            modality_ids = batch["modality_id"].to(device)
            feats = model.forward_features(images, modality_ids)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.vstack(features_list)[:max_samples]
    y = np.concatenate(labels_list)[:max_samples]

    def predict_fn(x):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            if model.use_quantum:
                logits = model.head(x_t)
            else:
                logits = model.head(x_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    background = shap.sample(X, min(10, len(X)))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X[:min(20, len(X))], nsamples=100)

    return {
        "mean_abs_shap": np.abs(shap_values).mean(axis=0).tolist(),
        "feature_dim": X.shape[1],
    }


def compute_vqc_gate_importance(model, n_perturbations: int = 20) -> dict:
    """Estimate VQC gate sensitivity via input perturbation (pilot study method)."""
    if not getattr(model, "use_quantum", False):
        return {"note": "Classical model — no VQC gates"}

    device = next(model.parameters()).device
    n_qubits = model.n_qubits
    base_input = torch.randn(1, n_qubits, device=device) * 0.5

    with torch.no_grad():
        base_out = model.head(base_input).softmax(dim=1)[0, 1].item()

    importances = []
    for q in range(n_qubits):
        deltas = []
        for _ in range(n_perturbations):
            perturbed = base_input.clone()
            perturbed[0, q] += 0.1
            with torch.no_grad():
                out = model.head(perturbed).softmax(dim=1)[0, 1].item()
            deltas.append(abs(out - base_out))
        importances.append({"qubit": q, "importance": float(np.mean(deltas))})

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return {"gate_importance": importances, "n_qubits": n_qubits}


def save_shap_results(results: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
