#!/usr/bin/env python3
"""Train hybrid or classical unified model."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import create_dataloaders
from src.data.splits import load_splits
from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel
from src.train.seed import set_seed
from src.train.trainer import HybridTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="E3", choices=["E2", "E3", "classical", "hybrid"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--modality",
        default=None,
        choices=["mammo", "ultrasound", "thermo"],
        help="Train on one modality only (use mammo for real CBIS-DDSM)",
    )
    args = parser.parse_args()

    modality_filter = [args.modality] if args.modality else None

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    splits_dir = ROOT / "data/splits"
    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        mammo = ROOT / "data/processed/mammo"
        raise SystemExit(
            "Missing data/splits/train.csv — splits are not in GitHub.\n"
            "1. Link mammography: data/processed/mammo/\n"
            "2. Run: python data/download/setup_datasets.py\n"
            f"   (mammo dir exists: {mammo.exists()})"
        )

    splits = load_splits(str(splits_dir))
    loaders = create_dataloaders(
        splits,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
        modality_filter=modality_filter,
    )

    model_cfg = config["model"]
    kwargs = dict(
        num_modalities=model_cfg["num_modality_tokens"],
        num_classes=model_cfg["num_classes"],
        encoder_dim=model_cfg["encoder_dim"],
        projection_dim=model_cfg["projection_dim"],
        compression_dims=model_cfg["compression_dims"],
        transformer_layers=model_cfg["transformer"]["num_layers"],
        transformer_heads=model_cfg["transformer"]["num_heads"],
    )

    use_hybrid = args.experiment in ("E3", "hybrid")
    if use_hybrid:
        model = HybridBreastCancerModel(
            **kwargs,
            n_qubits=model_cfg["quantum"]["n_qubits"],
            n_vqc_layers=model_cfg["quantum"]["n_layers"],
        )
        exp_name = f"E3_hybrid_seed{args.seed}"
    else:
        model = ClassicalBreastCancerModel(**kwargs)
        exp_name = f"E2_classical_seed{args.seed}"

    trainer = HybridTrainer(
        model,
        loaders["train"],
        loaders["val"],
        config,
        test_loader=loaders.get("test"),
        experiment_name=exp_name,
    )
    if args.quick:
        # Enough epochs for EfficientNet to learn on real mammo (~2k images)
        trainer.stage_a_epochs = 15
        trainer.stage_b_epochs = 5
        trainer.stage_c_epochs = 2
        if not use_hybrid:
            trainer.stage_a_epochs = 10
            trainer.stage_b_epochs = 0
            trainer.stage_c_epochs = 0

    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
