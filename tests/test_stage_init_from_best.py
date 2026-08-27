import tempfile
import unittest
from pathlib import Path

import torch

from src.train.trainer import HybridTrainer


class ToyModel(torch.nn.Module):
    use_quantum = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.stage = "stage_a"

    def set_training_stage(self, stage: str):
        self.stage = stage

    def forward(self, images, modality_ids):
        del modality_ids
        return torch.stack((-self.weight * images[:, 0, 0, 0],) * 2, dim=1)


def make_trainer(tmp_path: Path, enabled: bool) -> HybridTrainer:
    trainer = HybridTrainer.__new__(HybridTrainer)
    trainer.model = ToyModel()
    trainer.selection_metric = "balanced_accuracy"
    trainer.stage_init_from_best = enabled
    trainer.stage_transitions = []
    trainer.best_by_stage = {
        stage: {"score": float("-inf"), "state_dict": None, "stage_epoch": None}
        for stage in HybridTrainer.STAGES
    }
    return trainer


class StageInitFromBestTest(unittest.TestCase):
    def test_restores_best_preceding_stage_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp), enabled=True)

            # Stage A peaked at weight 5.0, then drifted to a worse terminal 1.0.
            trainer.best_by_stage["stage_a"] = {
                "score": 0.9,
                "state_dict": {"weight": torch.tensor(5.0)},
                "stage_epoch": 7,
            }
            trainer.model.weight.data = torch.tensor(1.0)

            restored = trainer._restore_preceding_stage_best("stage_b")

            self.assertIsNotNone(restored)
            self.assertEqual(restored["restored_from"], "stage_a")
            self.assertAlmostEqual(restored["restored_score"], 0.9)
            self.assertAlmostEqual(trainer.model.weight.item(), 5.0)

    def test_skips_stages_without_a_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp), enabled=True)
            trainer.best_by_stage["stage_a"] = {
                "score": 0.8,
                "state_dict": {"weight": torch.tensor(3.0)},
                "stage_epoch": 2,
            }
            # Stage B never produced a checkpoint, so Stage C must fall back to A.
            restored = trainer._restore_preceding_stage_best("stage_c")

            self.assertEqual(restored["restored_from"], "stage_a")
            self.assertAlmostEqual(trainer.model.weight.item(), 3.0)

    def test_first_stage_has_nothing_to_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp), enabled=True)
            self.assertIsNone(trainer._restore_preceding_stage_best("stage_a"))

    def test_published_default_leaves_terminal_weights_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(Path(tmp), enabled=False)
            trainer.best_by_stage["stage_a"] = {
                "score": 0.9,
                "state_dict": {"weight": torch.tensor(5.0)},
                "stage_epoch": 7,
            }
            trainer.model.weight.data = torch.tensor(1.0)

            # The flag is consulted by the training loop, not by the helper, so
            # the published behaviour is that no restore call is made at all.
            self.assertFalse(trainer.stage_init_from_best)
            self.assertAlmostEqual(trainer.model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
