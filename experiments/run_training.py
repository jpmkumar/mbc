#!/usr/bin/env python3
"""Train hybrid or classical unified model."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGE_CHOICES = ("a", "b", "c", "all")
STAGE_MAP = {"a": "stage_a", "b": "stage_b", "c": "stage_c"}


def _resolve_stages(stage_arg: str) -> list[str] | None:
    if stage_arg == "all":
        return None
    return [STAGE_MAP[stage_arg]]


def _default_resume_path(exp_name: str) -> Path | None:
    latest = ROOT / "results/checkpoints" / f"{exp_name}_latest.pt"
    return latest if latest.exists() else None


def main():
    parser = argparse.ArgumentParser(
        description="Train hybrid/classical model. Use --stage and --resume for Colab sessions."
    )
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
    parser.add_argument(
        "--stage",
        default="all",
        choices=STAGE_CHOICES,
        help="Run one stage (a=classical, b=VQC, c=joint) or all",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume (.pt). Default: results/checkpoints/{exp}_latest.pt if exists",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Ignore existing _latest.pt checkpoint",
    )
    parser.add_argument(
        "--reset-stages",
        action="store_true",
        help="Reset epoch progress for selected --stage (e.g. retry Stage B with new VQC)",
    )
    args = parser.parse_args()

    from src.data.dataloaders import create_dataloaders
    from src.data.splits import load_splits
    from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel
    from src.train.accelerator import configure_runtime, maybe_compile_model
    from src.train.seed import set_seed
    from src.train.trainer import HybridTrainer

    modality_filter = [args.modality] if args.modality else None

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    runtime = configure_runtime(config)

    use_hybrid = args.experiment in ("E3", "hybrid")

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
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    cache_features = train_cfg.get("cache_frozen_backbone_features", True)

    loaders = create_dataloaders(
        splits,
        batch_size=train_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=runtime["num_workers"],
        modality_filter=modality_filter,
        preprocess_config=data_cfg.get("preprocessing"),
        prefetch_factor=runtime["prefetch_factor"],
    )
    train_eval_loader = None
    if cache_features and use_hybrid:
        train_eval_loader = create_dataloaders(
            splits,
            batch_size=train_cfg["batch_size"],
            image_size=data_cfg["image_size"],
            num_workers=runtime["num_workers"],
            modality_filter=modality_filter,
            eval_train_transforms=True,
            preprocess_config=data_cfg.get("preprocessing"),
            prefetch_factor=runtime["prefetch_factor"],
        )["train"]

    model_cfg = config["model"]
    quantum_cfg = model_cfg.get("quantum", {})
    kwargs = dict(
        num_modalities=model_cfg["num_modality_tokens"],
        num_classes=model_cfg["num_classes"],
        encoder_dim=model_cfg["encoder_dim"],
        projection_dim=model_cfg["projection_dim"],
        compression_dims=model_cfg["compression_dims"],
        transformer_layers=model_cfg["transformer"]["num_layers"],
        transformer_heads=model_cfg["transformer"]["num_heads"],
    )

    suffix = config.get("project", {}).get("experiment_suffix", "")
    if use_hybrid:
        model = HybridBreastCancerModel(
            **kwargs,
            n_qubits=quantum_cfg.get("n_qubits", 8),
            n_vqc_layers=quantum_cfg.get("n_layers", 2),
            entanglement=quantum_cfg.get("entanglement", "linear"),
            quantum_feature_norm=quantum_cfg.get("feature_norm", True),
            quantum_full_readout=quantum_cfg.get("full_readout", True),
        )
        exp_name = f"E3_hybrid{suffix}_seed{args.seed}"
    else:
        model = ClassicalBreastCancerModel(**kwargs)
        exp_name = f"E2_classical{suffix}_seed{args.seed}"

    if train_cfg.get("compile_model", False):
        model = maybe_compile_model(model, enabled=True)

    trainer = HybridTrainer(
        model,
        loaders["train"],
        loaders["val"],
        config,
        test_loader=loaders.get("test"),
        experiment_name=exp_name,
        train_eval_loader=train_eval_loader,
    )
    if args.quick:
        trainer.stage_a_epochs = 15
        trainer.stage_b_epochs = 5
        trainer.stage_c_epochs = 2
        if not use_hybrid:
            trainer.stage_a_epochs = 10
            trainer.stage_b_epochs = 0
            trainer.stage_c_epochs = 0

    stages_filter = _resolve_stages(args.stage)
    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume:
        default_latest = _default_resume_path(exp_name)
        if default_latest:
            resume_path = str(default_latest)
            print(f"Auto-resuming from {resume_path}")
    elif resume_path:
        resume_path = str(resume_path)

    if stages_filter and stages_filter[0] != "stage_a" and resume_path is None:
        default_best = ROOT / "results/checkpoints" / f"{exp_name}.pt"
        if default_best.exists():
            resume_path = str(default_best)
            print(f"Stage {args.stage} requires weights — using {resume_path}")

    if args.reset_stages and stages_filter:
        for stage_name in stages_filter:
            trainer.stage_epochs_done[stage_name] = 0
        print(f"Reset stage progress for: {', '.join(stages_filter)}")

    metrics = trainer.train(stages_filter=stages_filter, resume_path=resume_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
