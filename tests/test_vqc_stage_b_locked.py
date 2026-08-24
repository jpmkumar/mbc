import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_vqc_stage_b_locked import (
    load_locked_settings,
    patient_cluster_bootstrap,
    patient_row_groups,
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
        self.assertIsNone(summary["patient_cluster_bootstrap"])
        self.assertIn("no patient IDs", summary["uncertainty_note"])
        self.assertEqual(summary["fold"], 0)

    def test_summary_labels_the_evaluated_fold(self):
        results = [
            {
                "model": model,
                "seed": 42,
                "feature_transform": "raw",
                "learning_rate": 0.001,
                "test_metrics": {
                    "auprc": 0.90,
                    "auc": 0.95,
                    "balanced_accuracy": 0.88,
                    "f1": 0.82,
                    "precision": 0.76,
                    "recall": 0.90,
                },
            }
            for model in ("mlp", "vqc")
        ]

        summary = summarize_results(results, fold=1)

        self.assertEqual(summary["fold"], 1)
        self.assertEqual(
            summary["protocol"], "one_time_validation_locked_fold1_test"
        )
        self.assertIn("Fold 1", summary["verdict"])


class SeedStabilityTests(unittest.TestCase):
    @staticmethod
    def build_results(mlp_values, vqc_values):
        results = []
        for model, values in (("mlp", mlp_values), ("vqc", vqc_values)):
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
        return results

    def test_one_bad_seed_is_flagged_and_median_is_reported(self):
        results = self.build_results(
            [0.90457, 0.89128, 0.90468], [0.90452, 0.90460, 0.90462]
        )

        summary = summarize_results(results, fold=2)

        self.assertEqual(summary["unstable_models"], ["mlp"])
        self.assertTrue(summary["seed_stability"]["mlp"]["unstable"])
        self.assertFalse(summary["seed_stability"]["vqc"]["unstable"])
        self.assertGreater(summary["mean_vqc_minus_mlp_test_auprc"], 0.004)
        self.assertLess(summary["median_vqc_minus_mlp_test_auprc"], 0.0)
        self.assertIn("poorly", summary["verdict"])
        self.assertIn("seed median", summary["verdict"])

    def test_stable_seeds_add_no_caution(self):
        results = self.build_results(
            [0.90457, 0.90461, 0.90468], [0.90452, 0.90460, 0.90462]
        )

        summary = summarize_results(results, fold=1)

        self.assertEqual(summary["unstable_models"], [])
        self.assertNotIn("poorly", summary["verdict"])


class PatientClusterBootstrapTests(unittest.TestCase):
    def test_groups_keep_every_row_of_a_patient_together(self):
        patient_ids = ["b", "a", "b", "c", "a", "b"]

        groups = patient_row_groups(patient_ids)

        self.assertEqual(len(groups), 3)
        self.assertEqual(
            sorted(sorted(group.tolist()) for group in groups),
            [[0, 2, 5], [1, 4], [3]],
        )
        self.assertEqual(
            sorted(np.concatenate(groups).tolist()), list(range(6))
        )

    def test_identical_heads_give_an_interval_covering_zero(self):
        generator = np.random.default_rng(0)
        labels = np.repeat([0, 1], 100)
        patient_ids = [f"p{index // 10}" for index in range(200)]
        scores = generator.random(200) * 0.4 + labels * 0.3
        probabilities = {
            "mlp": {42: scores, 43: scores},
            "vqc": {42: scores, 43: scores},
        }

        bootstrap = patient_cluster_bootstrap(
            labels, patient_ids, probabilities, replicates=200, seed=7
        )

        self.assertEqual(bootstrap["unique_test_patients"], 20)
        self.assertAlmostEqual(bootstrap["ci_lower"], 0.0)
        self.assertAlmostEqual(bootstrap["ci_upper"], 0.0)
        self.assertFalse(bootstrap["excludes_zero"])
        self.assertTrue(bootstrap["within_practical_margin"])

    def test_clearly_better_head_gives_an_interval_above_zero(self):
        generator = np.random.default_rng(1)
        labels = np.repeat([0, 1], 150)
        patient_ids = [f"p{index // 10}" for index in range(300)]
        weak = generator.random(300)
        strong = generator.random(300) * 0.2 + labels * 0.8
        probabilities = {"mlp": {42: weak}, "vqc": {42: strong}}

        bootstrap = patient_cluster_bootstrap(
            labels, patient_ids, probabilities, replicates=200, seed=11
        )

        self.assertGreater(bootstrap["ci_lower"], 0.0)
        self.assertTrue(bootstrap["excludes_zero"])
        self.assertFalse(bootstrap["within_practical_margin"])
        self.assertLessEqual(bootstrap["usable_replicates"], 200)

    def test_bootstrap_verdict_reports_patient_level_uncertainty(self):
        results = [
            {
                "model": model,
                "seed": 42,
                "feature_transform": "raw",
                "learning_rate": 0.001,
                "test_metrics": {
                    "auprc": 0.90,
                    "auc": 0.95,
                    "balanced_accuracy": 0.88,
                    "f1": 0.82,
                    "precision": 0.76,
                    "recall": 0.90,
                },
            }
            for model in ("mlp", "vqc")
        ]
        bootstrap = {
            "excludes_zero": False,
            "within_practical_margin": True,
        }

        summary = summarize_results(results, fold=1, bootstrap=bootstrap)

        self.assertIs(summary["patient_cluster_bootstrap"], bootstrap)
        self.assertIn("resamples whole test patients", summary["uncertainty_note"])
        self.assertIn("inside the practical margin", summary["verdict"])
        self.assertNotIn("excluding zero", summary["verdict"])

        summary = summarize_results(
            results,
            fold=1,
            bootstrap={"excludes_zero": True, "within_practical_margin": True},
        )

        self.assertIn("inside the practical margin", summary["verdict"])
        self.assertIn("too small to matter clinically", summary["verdict"])


if __name__ == "__main__":
    unittest.main()
