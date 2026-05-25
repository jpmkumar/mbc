"""Experiment matrix E1–E8 runner."""

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from src.data.dataloaders import create_dataloaders
from src.data.splits import load_splits
from src.models.hybrid_model import ClassicalBreastCancerModel, HybridBreastCancerModel
from src.train.seed import set_seed
from src.train.trainer import HybridTrainer

from .metrics import evaluate_by_modality, evaluate_model


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_model(config: dict, experiment: str, seed: int):
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

    if experiment in ("E2", "classical_unified"):
        return ClassicalBreastCancerModel(**kwargs)
    if experiment == "E6_no_tokens":
        return HybridBreastCancerModel(**kwargs, use_modality_tokens=False)
    if experiment == "E6_no_transformer":
        return HybridBreastCancerModel(**kwargs, use_transformer=False)
    if experiment.startswith("E7_qubits"):
        n_qubits = int(experiment.split("_")[-1])
        kwargs["compression_dims"] = [128, 32, n_qubits]
        return HybridBreastCancerModel(**kwargs, n_qubits=n_qubits)
    return HybridBreastCancerModel(
        **kwargs,
        n_qubits=model_cfg["quantum"]["n_qubits"],
        n_vqc_layers=model_cfg["quantum"]["n_layers"],
    )


def leave_one_modality_out(
    config_path: str,
    held_out: str,
    experiment: str = "E3",
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Train on all modalities except held_out; test on held_out."""
    config = _load_config(config_path)
    set_seed(seed)
    splits = load_splits("data/splits")
    all_modalities = config["data"]["modalities"]
    train_modalities = [m for m in all_modalities if m != held_out]

    loaders = create_dataloaders(
        splits,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
    )
    train_loader = create_dataloaders(
        {"train": splits["train"]},
        batch_size=config["training"]["batch_size"],
        modality_filter=train_modalities,
    )["train"]
    test_loader = create_dataloaders(
        {"test": splits["test"]},
        batch_size=config["training"]["batch_size"],
        modality_filter=[held_out],
    )["test"]

    model = _build_model(config, experiment, seed)
    exp_name = f"E4_lomo_{held_out}_seed{seed}"
    trainer = HybridTrainer(
        model, train_loader, loaders["val"], config,
        test_loader=loaders.get("test"), experiment_name=exp_name,
    )
    if quick:
        trainer.stage_a_epochs = 2
        trainer.stage_b_epochs = 2
        trainer.stage_c_epochs = 1
    train_metrics = trainer.train()
    device = trainer.device
    test_metrics = evaluate_model(model, test_loader, device)
    by_mod = evaluate_by_modality(model, test_loader, device)

    result = {
        "experiment": "E4",
        "held_out_modality": held_out,
        "train_modalities": train_modalities,
        "seed": seed,
        "val_metrics": train_metrics,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("labels", "preds", "probs", "roc")},
        "by_modality": by_mod,
    }
    out = Path(config["paths"]["results"]) / f"{exp_name}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_experiment_matrix(config_path: str, quick: bool = False) -> dict:
    """Run core experiments E1–E8."""
    config = _load_config(config_path)
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    seeds = [42] if quick else config["experiments"]["seeds"]
    experiments = ["E2", "E3"]
    if not quick:
        experiments += ["E6_no_tokens", "E6_no_transformer"]
        for q in config["experiments"]["qubit_sweep"]:
            experiments.append(f"E7_qubits_{q}")

    splits = load_splits("data/splits")
    loaders = create_dataloaders(
        splits,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
    )

    seed_metrics = {exp: [] for exp in experiments}

    for seed in seeds:
        set_seed(seed)
        for exp in experiments:
            model = _build_model(config, exp, seed)
            exp_name = f"{exp}_seed{seed}"
            trainer = HybridTrainer(
                model, loaders["train"], loaders["val"], config,
                test_loader=loaders.get("test"), experiment_name=exp_name,
            )
            if quick:
                trainer.stage_a_epochs = 2
                trainer.stage_b_epochs = 2
                trainer.stage_c_epochs = 1
            metrics = trainer.train()
            test_m = evaluate_model(model, loaders["test"], trainer.device)
            seed_metrics[exp].append(test_m["f1"])

    # E8: statistical significance between E2 and E3
    if len(seed_metrics.get("E2", [])) >= 2 and len(seed_metrics.get("E3", [])) >= 2:
        t_stat, p_val = stats.ttest_rel(seed_metrics["E3"], seed_metrics["E2"])
        all_results["E8_significance"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "E2_f1_mean": float(np.mean(seed_metrics["E2"])),
            "E3_f1_mean": float(np.mean(seed_metrics["E3"])),
        }

    for exp in experiments:
        all_results[exp] = {
            "f1_mean": float(np.mean(seed_metrics[exp])),
            "f1_std": float(np.std(seed_metrics[exp])),
            "runs": seed_metrics[exp],
        }

    # E4: leave-one-modality-out
    lomo_mods = config["data"]["modalities"][:1] if quick else config["data"]["modalities"]
    for mod in lomo_mods:
        try:
            all_results[f"E4_{mod}"] = leave_one_modality_out(
                config_path, mod, experiment="E3", seed=seeds[0], quick=quick
            )
        except Exception as e:
            all_results[f"E4_{mod}"] = {"error": str(e)}

    summary_path = results_dir / "experiment_matrix_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results
