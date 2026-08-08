import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.train.trainer import HybridTrainer


class RoutedToyModel(torch.nn.Module):
    use_quantum = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.stage = "stage_a"

    def set_training_stage(self, stage: str):
        self.stage = stage

    def forward(self, images, modality_ids):
        del modality_ids
        feature = images[:, 0, 0, 0]
        route_sign = 1.0 if self.stage in ("stage_a", "stage_c") else -1.0
        score = route_sign * self.weight * feature
        return torch.stack((-score, score), dim=1)


def make_trainer(tmp_path: Path) -> HybridTrainer:
    trainer = HybridTrainer.__new__(HybridTrainer)
    trainer.model = RoutedToyModel()
    trainer.experiment_name = "E3_histopath_fold0_seed42"
    trainer.results_dir = tmp_path / "results"
    trainer.ckpt_dir = tmp_path / "checkpoints"
    trainer.results_dir.mkdir()
    trainer.ckpt_dir.mkdir()
    trainer.selection_metric = "balanced_accuracy"
    trainer.config = {
        "run": {
            "experiment": "E3",
            "fold": 0,
            "seed": 42,
            "effective_seed": 42,
        },
        "training": {
            "eval_threshold": 0.5,
            "tune_threshold": True,
            "threshold_metric": "f1",
            "tta": False,
        },
    }
    trainer.stage_epochs_done = {stage: 1 for stage in trainer.STAGES}
    trainer.total_epochs = 3
    trainer.history = {"train_loss": [], "val_metrics": []}
    trainer.best_stage = "stage_c"
    trainer.best_score = 0.9
    trainer.best_by_stage = {
        stage: {
            "score": 0.6 + index / 10,
            "state_dict": trainer._snapshot_state(trainer.model),
            "stage_epoch": index,
            "global_epoch": index,
            "val_metrics": {"balanced_accuracy": 0.6 + index / 10},
        }
        for index, stage in enumerate(trainer.STAGES, start=1)
    }
    trainer.best_state = trainer.best_by_stage["stage_c"]["state_dict"]
    trainer.classical_device = torch.device("cpu")
    trainer.stage_a_epochs = 1
    trainer.stage_b_epochs = 1
    trainer.stage_c_epochs = 1

    batch = {
        "image": torch.tensor([[[[-1.0]]], [[[1.0]]]]),
        "label": torch.tensor([0, 1]),
        "modality_id": torch.tensor([0, 0]),
    }
    trainer.val_loader = [batch]
    trainer.test_loader = [batch]
    return trainer


class StageAttributionTests(unittest.TestCase):
    def test_stage_checkpoint_reloads_identical_route_and_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp))
            trainer.model.set_training_stage("stage_b")
            batch = trainer.val_loader[0]
            expected = trainer.model(
                batch["image"], batch["modality_id"]
            ).detach()
            trainer._save_stage_best_checkpoint("stage_b")

            trainer.model.weight.data.fill_(9.0)
            trainer.model.set_training_stage("stage_a")
            trainer._load_checkpoint(trainer._best_stage_ckpt_path("stage_b"))
            actual = trainer.model(
                batch["image"], batch["modality_id"]
            ).detach()

            self.assertEqual(trainer.model.stage, "stage_b")
            torch.testing.assert_close(actual, expected)

    def test_stage_comparison_contains_all_routes_and_global_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp))
            for stage in trainer.STAGES:
                trainer._save_stage_best_checkpoint(stage)
            trainer._save_global_best_checkpoint()

            comparison = trainer._evaluate_stage_attribution()

            self.assertEqual(set(comparison["stages"]), set(trainer.STAGES))
            self.assertEqual(comparison["global_selected"]["stage"], "stage_c")
            self.assertEqual(trainer.model.stage, "stage_c")
            for stage in trainer.STAGES:
                metrics = comparison["stages"][stage]
                self.assertIn("specificity", metrics["validation_metrics"])
                self.assertIn("threshold", metrics["threshold_tuning"])

            saved = json.loads(trainer._stage_comparison_path().read_text())
            self.assertEqual(saved["global_best_stage"], "stage_c")

    def test_checkpoint_records_run_and_circuit_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp))
            trainer.config["model"] = {
                "quantum": {
                    "n_qubits": 8,
                    "n_layers": 2,
                    "encoding": "angle_y",
                    "entanglement": "linear",
                }
            }
            trainer._save_stage_best_checkpoint("stage_a")

            payload = torch.load(
                trainer._best_stage_ckpt_path("stage_a"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["stage"], "stage_a")
            self.assertEqual(payload["config"]["run"]["fold"], 0)
            self.assertEqual(payload["config"]["run"]["seed"], 42)
            self.assertEqual(
                payload["config"]["model"]["quantum"]["encoding"], "angle_y"
            )


if __name__ == "__main__":
    unittest.main()
