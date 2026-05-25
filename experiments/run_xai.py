#!/usr/bin/env python3
"""Generate XAI outputs: Grad-CAM, attention, SHAP, case studies."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import create_dataloaders
from src.data.splits import load_splits
from src.models.hybrid_model import HybridBreastCancerModel
from src.xai.case_study import generate_case_study_figure
from src.xai.shap_analysis import compute_shap_features, compute_vqc_gate_importance, save_shap_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/E3_hybrid_seed42.pt")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    splits = load_splits(str(ROOT / "data/splits"))
    loaders = create_dataloaders(
        splits,
        batch_size=8,
        image_size=config["data"]["image_size"],
        num_workers=0,
    )

    model_cfg = config["model"]
    model = HybridBreastCancerModel(
        num_modalities=model_cfg["num_modality_tokens"],
        num_classes=model_cfg["num_classes"],
        encoder_dim=model_cfg["encoder_dim"],
        projection_dim=model_cfg["projection_dim"],
        compression_dims=model_cfg["compression_dims"],
        transformer_layers=model_cfg["transformer"]["num_layers"],
        transformer_heads=model_cfg["transformer"]["num_heads"],
        n_qubits=model_cfg["quantum"]["n_qubits"],
        n_vqc_layers=model_cfg["quantum"]["n_layers"],
    )

    ckpt_path = ROOT / args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Case studies
    test_loader = loaders["test"]
    for i, batch in enumerate(test_loader):
        if i >= 3:
            break
        sample = {k: batch[k][0] for k in ("image", "label", "modality_id", "modality")}
        generate_case_study_figure(
            model, sample, device,
            str(figures_dir / f"case_study_{i}_{sample['modality']}.png"),
        )

    # SHAP
    shap_results = compute_shap_features(model, test_loader, device, max_samples=30)
    gate_results = compute_vqc_gate_importance(model)
    combined = {"shap": shap_results, "vqc_gates": gate_results}
    save_shap_results(combined, str(figures_dir / "shap_vqc_analysis.json"))
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
