from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.run_histopath_width_kaggle import (
    build_train_command,
    checkpoint_finiteness,
    select_result,
    validate_pair,
)
from scripts.train_histopath_cv import _apply_quantum_ablation_overrides


def args_for_width(width: int):
    return SimpleNamespace(
        n_qubits=width,
        n_layers=None,
        entanglement=None,
        encoding=None,
        data_reuploading=None,
    )


class QuantumWidthOverrideTests(unittest.TestCase):
    def test_width_changes_both_circuit_and_compression_bottleneck(self):
        for width in (4, 12):
            with self.subTest(width=width):
                config = {
                    "project": {"experiment_suffix": "_histopath"},
                    "model": {
                        "compression_dims": [128, 32, 8],
                        "quantum": {"n_qubits": 8},
                    },
                }

                _apply_quantum_ablation_overrides(config, args_for_width(width))

                self.assertEqual(config["model"]["quantum"]["n_qubits"], width)
                self.assertEqual(
                    config["model"]["compression_dims"], [128, 32, width]
                )
                self.assertEqual(
                    config["project"]["experiment_suffix"],
                    f"_histopath_q{width}",
                )

    def test_override_does_not_mutate_earlier_compression_layers(self):
        config = {
            "project": {},
            "model": {
                "compression_dims": [256, 64, 8],
                "quantum": {"n_qubits": 8},
            },
        }

        _apply_quantum_ablation_overrides(config, args_for_width(4))

        self.assertEqual(config["model"]["compression_dims"][:2], [256, 64])


class KaggleWidthRunnerTests(unittest.TestCase):
    def test_only_declared_fold_width_pairs_are_accepted(self):
        validate_pair(1, 4)
        validate_pair(4, 12)
        for fold, width in ((0, 4), (5, 12), (1, 8)):
            with self.subTest(fold=fold, width=width):
                with self.assertRaises(ValueError):
                    validate_pair(fold, width)

    def test_train_command_contains_the_declared_pair_and_seed(self):
        command = build_train_command(3, 12, Path("/kaggle/input/breast"))

        self.assertIn("--fold", command)
        self.assertEqual(command[command.index("--fold") + 1], "3")
        self.assertEqual(command[command.index("--n-qubits") + 1], "12")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(
            command[command.index("--archive-path") + 1],
            "/kaggle/input/breast",
        )

    def test_result_selection_refuses_missing_or_duplicate_records(self):
        row = {"fold": 2, "n_qubits": 4}
        summary = {"results": {"E3": [row]}}

        self.assertIs(select_result(summary, 2, 4), row)
        with self.assertRaises(RuntimeError):
            select_result(summary, 2, 12)
        with self.assertRaises(RuntimeError):
            select_result({"results": {"E3": [row, row]}}, 2, 4)

    def test_checkpoint_audit_distinguishes_finite_and_corrupt_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            finite_path = Path(tmp) / "finite.pt"
            corrupt_path = Path(tmp) / "corrupt.pt"
            torch.save(
                {"model_state_dict": {"weight": torch.tensor([1.0, 2.0])}},
                finite_path,
            )
            torch.save(
                {
                    "model_state_dict": {
                        "weight": torch.tensor([1.0, float("nan"), float("inf")])
                    }
                },
                corrupt_path,
            )

            finite = checkpoint_finiteness(finite_path)
            corrupt = checkpoint_finiteness(corrupt_path)

        self.assertTrue(finite["numerically_valid"])
        self.assertEqual(finite["nonfinite_values"], 0)
        self.assertFalse(corrupt["numerically_valid"])
        self.assertEqual(corrupt["nonfinite_values"], 2)
        self.assertEqual(corrupt["affected_tensor_count"], 1)


if __name__ == "__main__":
    unittest.main()
