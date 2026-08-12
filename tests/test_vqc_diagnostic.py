import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_histopath_vqc import (
    _optimal_balanced_threshold,
    apply_feature_transform,
    balanced_indices,
)
from src.utils.metrics import compute_metrics_at_threshold


class VQCTrainabilityDiagnosticTests(unittest.TestCase):
    def test_balanced_indices_are_reproducible(self):
        labels = torch.tensor([0] * 10 + [1] * 6)
        first = balanced_indices(labels, per_class=4, seed=42)
        second = balanced_indices(labels, per_class=4, seed=42)

        torch.testing.assert_close(first, second)
        self.assertEqual(torch.bincount(labels[first], minlength=2).tolist(), [4, 4])

    def test_tuned_threshold_recovers_separation_above_default_boundary(self):
        tuned = _optimal_balanced_threshold(
            labels=[0, 0, 1, 1],
            probs=[0.60, 0.61, 0.80, 0.81],
        )

        self.assertEqual(tuned["balanced_accuracy"], 1.0)
        self.assertGreater(tuned["threshold"], 0.61)
        self.assertLess(tuned["threshold"], 0.80)

    def test_fast_threshold_matches_exhaustive_sweep(self):
        labels = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        probs = np.array([0.2, 0.7, 0.4, 0.8, 0.7, 0.1, 0.4, 0.9])
        unique = np.unique(probs)
        thresholds = np.concatenate(
            (
                [np.nextafter(unique[0], -np.inf)],
                (unique[:-1] + unique[1:]) / 2.0,
                [np.nextafter(unique[-1], np.inf)],
            )
        )
        exhaustive = [
            compute_metrics_at_threshold(labels, probs, threshold)
            for threshold in thresholds
        ]
        expected_score = max(
            row["balanced_accuracy"] for row in exhaustive
        )

        actual = _optimal_balanced_threshold(labels, probs)

        self.assertAlmostEqual(
            actual["balanced_accuracy"], expected_score
        )

    def test_standardization_is_fitted_on_training_cache_only(self):
        fit = torch.tensor([[0.0, 10.0], [2.0, 14.0]])
        train = torch.tensor([[1.0, 12.0]])
        val = torch.tensor([[3.0, 16.0]])

        transformed_train, transformed_val, metadata = (
            apply_feature_transform(
                fit,
                train,
                val,
                transform="standardize",
            )
        )

        torch.testing.assert_close(
            transformed_train, torch.tensor([[0.0, 0.0]])
        )
        torch.testing.assert_close(
            transformed_val, torch.tensor([[2.0, 2.0]])
        )
        self.assertEqual(metadata["fit_scope"], "full_training_cache")

    def test_cli_writes_manifest_summary_and_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "features.pt"
            output_dir = root / "diagnostic"
            generator = torch.Generator().manual_seed(7)
            cache = {}
            for split, samples in (("train", 20), ("val", 12)):
                labels = torch.tensor([0, 1] * (samples // 2))
                features = torch.randn(samples, 8, generator=generator)
                features[:, 0] += labels.float() * 2 - 1
                cache[split] = {
                    "features": features,
                    "labels": labels,
                    "modality_ids": torch.zeros(samples, dtype=torch.long),
                }
            cache["test"] = None
            torch.save(cache, cache_path)

            subprocess.run(
                [
                    sys.executable,
                    "scripts/diagnose_histopath_vqc.py",
                    "--feature-cache",
                    str(cache_path),
                    "--output-dir",
                    str(output_dir),
                    "--models",
                    "linear",
                    "mlp",
                    "vqc",
                    "--seeds",
                    "42",
                    "--train-per-class",
                    "4",
                    "--val-per-class",
                    "3",
                    "--epochs",
                    "2",
                    "--max-steps",
                    "3",
                    "--eval-every-steps",
                    "2",
                    "--batch-size",
                    "4",
                    "--stop-after-success",
                    "0",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            manifest = json.loads(
                (output_dir / "sample_manifest.json").read_text()
            )
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(manifest["train_class_counts"], [4, 4])
            self.assertIn(
                "effective_rank",
                manifest["selected_train_feature_diagnostics"],
            )
            self.assertEqual(len(summary["runs"]), 3)
            self.assertEqual(
                {run["optimizer_steps_completed"] for run in summary["runs"]},
                {3},
            )
            self.assertTrue(
                all(
                    "best_train_tuned_balanced_accuracy" in run
                    for run in summary["runs"]
                )
            )
            self.assertTrue((output_dir / "epoch_metrics.csv").exists())
            self.assertTrue(
                (output_dir / "linear_seed42_best_train.pt").exists()
            )
            self.assertTrue((output_dir / "linear_seed42_best_auc.pt").exists())
            self.assertTrue((output_dir / "linear_seed42_best_loss.pt").exists())
            self.assertTrue(
                (output_dir / "linear_seed42_best_val_auprc.pt").exists()
            )
            self.assertTrue((output_dir / "linear_seed42_final.pt").exists())


if __name__ == "__main__":
    unittest.main()
