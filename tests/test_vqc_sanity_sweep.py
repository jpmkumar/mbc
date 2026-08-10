import unittest

from scripts.run_vqc_sanity_sweep import build_summary


class ControlledSanitySweepTests(unittest.TestCase):
    def test_verdict_identifies_vqc_specific_failure(self):
        runs = []
        for seed, mlp_score, vqc_score in (
            (42, 0.99, 0.80),
            (43, 0.97, 0.85),
            (44, 0.90, 0.96),
        ):
            runs.extend(
                [
                    {
                        "samples_per_class": 16,
                        "learning_rate": 0.003,
                        "model": "mlp",
                        "seed": seed,
                        "best_train_tuned_balanced_accuracy": mlp_score,
                    },
                    {
                        "samples_per_class": 16,
                        "learning_rate": 0.003,
                        "model": "vqc",
                        "seed": seed,
                        "best_train_tuned_balanced_accuracy": vqc_score,
                    },
                ]
            )

        summary = build_summary(
            runs,
            sizes=[16],
            learning_rates=[0.003],
            success_threshold=0.95,
        )

        self.assertTrue(summary["mlp_passes_any_setting"])
        self.assertFalse(summary["vqc_passes_any_setting"])
        self.assertIn("VQC-specific", summary["verdict"])

    def test_verdict_validates_both_heads(self):
        runs = []
        for model in ("mlp", "vqc"):
            for seed in (42, 43, 44):
                runs.append(
                    {
                        "samples_per_class": 32,
                        "learning_rate": 0.01,
                        "model": model,
                        "seed": seed,
                        "best_train_tuned_balanced_accuracy": 0.98,
                    }
                )

        summary = build_summary(
            runs,
            sizes=[32],
            learning_rates=[0.01],
            success_threshold=0.95,
        )

        self.assertTrue(summary["mlp_passes_any_setting"])
        self.assertTrue(summary["vqc_passes_any_setting"])
        self.assertIn("mechanics as validated", summary["verdict"])


if __name__ == "__main__":
    unittest.main()
