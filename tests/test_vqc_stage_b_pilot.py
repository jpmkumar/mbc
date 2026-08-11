import unittest

from scripts.run_vqc_stage_b_pilot import build_summary


class StageBPilotTests(unittest.TestCase):
    def test_summary_selects_by_validation_auprc(self):
        runs = []
        for model, raw_scores, standardized_scores in (
            ("mlp", [0.80, 0.82, 0.81], [0.84, 0.85, 0.83]),
            ("vqc", [0.78, 0.79, 0.80], [0.86, 0.87, 0.85]),
        ):
            for transform, scores in (
                ("raw", raw_scores),
                ("standardize", standardized_scores),
            ):
                for seed, score in zip((42, 43, 44), scores):
                    runs.append(
                        {
                            "feature_transform": transform,
                            "learning_rate": 0.003,
                            "model": model,
                            "seed": seed,
                            "best_val_auprc": score,
                            "best_val_auc": score + 0.05,
                            "best_val_tuned_balanced_accuracy": score - 0.02,
                            "best_train_tuned_balanced_accuracy": 0.95,
                            "runtime_s": 10.0,
                        }
                    )

        summary = build_summary(runs)

        self.assertEqual(summary["selection_endpoint"], "validation_auprc")
        self.assertEqual(
            summary["best_by_model"]["mlp"]["feature_transform"],
            "standardize",
        )
        self.assertEqual(
            summary["best_by_model"]["vqc"]["feature_transform"],
            "standardize",
        )
        self.assertAlmostEqual(
            summary["best_vqc_minus_mlp_val_auprc"],
            0.02,
        )
        self.assertIn("confirm", summary["verdict"])

    def test_summary_reports_practical_tie(self):
        runs = []
        for model, score in (("mlp", 0.85), ("vqc", 0.855)):
            for seed in (42, 43, 44):
                runs.append(
                    {
                        "feature_transform": "raw",
                        "learning_rate": 0.001,
                        "model": model,
                        "seed": seed,
                        "best_val_auprc": score,
                        "best_val_auc": 0.90,
                        "best_val_tuned_balanced_accuracy": 0.84,
                        "best_train_tuned_balanced_accuracy": 0.90,
                        "runtime_s": 10.0,
                    }
                )

        summary = build_summary(runs)

        self.assertIn("practically tied", summary["verdict"])


if __name__ == "__main__":
    unittest.main()
