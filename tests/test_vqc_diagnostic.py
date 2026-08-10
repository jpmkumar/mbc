import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.diagnose_histopath_vqc import balanced_indices


class VQCTrainabilityDiagnosticTests(unittest.TestCase):
    def test_balanced_indices_are_reproducible(self):
        labels = torch.tensor([0] * 10 + [1] * 6)
        first = balanced_indices(labels, per_class=4, seed=42)
        second = balanced_indices(labels, per_class=4, seed=42)

        torch.testing.assert_close(first, second)
        self.assertEqual(torch.bincount(labels[first], minlength=2).tolist(), [4, 4])

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
            self.assertEqual(len(summary["runs"]), 3)
            self.assertFalse(summary["decision"]["proceed_to_full_benchmark"])
            self.assertTrue((output_dir / "epoch_metrics.csv").exists())
            self.assertTrue(
                (output_dir / "linear_seed42_best_train.pt").exists()
            )


if __name__ == "__main__":
    unittest.main()
