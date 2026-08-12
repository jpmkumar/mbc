import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_vqc_stage_b_locked import (
    load_locked_settings,
    setting_tag,
    summarize_results,
)


class StageBLockedTestTests(unittest.TestCase):
    def test_locked_settings_require_untouched_test_and_validation_selection(self):
        payload = {
            "selection_endpoint": "validation_auprc",
            "held_out_test_evaluated": False,
            "best_by_model": {
                "mlp": {
                    "model": "mlp",
                    "feature_transform": "raw",
                    "learning_rate": 0.001,
                },
                "vqc": {
                    "model": "vqc",
                    "feature_transform": "raw",
                    "learning_rate": 0.01,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "locked.json"
            path.write_text(json.dumps(payload))
            settings = load_locked_settings(path)
            self.assertEqual(setting_tag(settings["mlp"]), "raw_lr0p001")
            self.assertEqual(setting_tag(settings["vqc"]), "raw_lr0p01")

            payload["held_out_test_evaluated"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "evaluated=false"):
                load_locked_settings(path)

    def test_summary_reports_paired_practical_tie_without_equivalence_claim(self):
        results = []
        for model, values in (
            ("mlp", [0.90, 0.91, 0.92]),
            ("vqc", [0.901, 0.909, 0.921]),
        ):
            for seed, auprc in zip((42, 43, 44), values):
                results.append(
                    {
                        "model": model,
                        "seed": seed,
                        "feature_transform": "raw",
                        "learning_rate": 0.001 if model == "mlp" else 0.01,
                        "test_metrics": {
                            "auprc": auprc,
                            "auc": 0.95,
                            "balanced_accuracy": 0.88,
                            "f1": 0.82,
                            "precision": 0.76,
                            "recall": 0.90,
                        },
                    }
                )

        summary = summarize_results(results)

        self.assertTrue(summary["held_out_test_evaluated"])
        self.assertAlmostEqual(
            summary["mean_vqc_minus_mlp_test_auprc"],
            1.0 / 3000.0,
        )
        self.assertIn("no detected quantum advantage", summary["verdict"])
        self.assertIn("not a formal equivalence", summary["verdict"])


if __name__ == "__main__":
    unittest.main()
