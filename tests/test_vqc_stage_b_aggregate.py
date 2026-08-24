import unittest

from scripts.aggregate_vqc_stage_b_folds import (
    corrected_interval,
    decide,
    equivalence_test,
    pooled_instability,
)


class CorrectedIntervalTests(unittest.TestCase):
    def test_correction_widens_the_naive_interval(self):
        deltas = [-0.006, -0.0003, 0.0044, 0.001, -0.002]

        interval = corrected_interval(deltas, test_train_ratio=0.27, confidence=0.90)

        self.assertGreater(
            interval["corrected_standard_error"],
            interval["naive_standard_error"],
        )
        self.assertLess(interval["naive_lower"], 0.0)
        self.assertLess(interval["lower"], interval["naive_lower"])
        self.assertGreater(interval["upper"], interval["naive_upper"])
        self.assertAlmostEqual(interval["variance_inflation"], 1.0 + 5 * 0.27)

    def test_two_folds_are_the_minimum(self):
        with self.assertRaises(ValueError):
            corrected_interval([0.001], test_train_ratio=0.27, confidence=0.90)


class DecisionRuleTests(unittest.TestCase):
    @staticmethod
    def evaluate(deltas, ratio=0.27):
        interval_90 = corrected_interval(deltas, ratio, 0.90)
        interval_95 = corrected_interval(deltas, ratio, 0.95)
        tost = equivalence_test(deltas, ratio)
        return decide(interval_90, interval_95, tost), tost

    def test_tight_gaps_around_zero_give_equivalence(self):
        (decision, wording), tost = self.evaluate(
            [-0.0004, 0.0002, -0.0001, 0.0003, 0.0000]
        )

        self.assertEqual(decision, "practical_equivalence")
        self.assertTrue(tost["equivalent"])
        self.assertIn("not", wording)

    def test_large_consistent_gap_gives_a_difference(self):
        (decision, wording), _ = self.evaluate(
            [0.031, 0.029, 0.033, 0.030, 0.032]
        )

        self.assertEqual(decision, "difference")
        self.assertIn("VQC", wording)

    def test_wide_scatter_inside_the_margin_is_inconclusive(self):
        (decision, _), tost = self.evaluate(
            [-0.009, 0.008, -0.007, 0.009, -0.006]
        )

        self.assertEqual(decision, "inconclusive")
        self.assertFalse(tost["equivalent"])

    def test_equivalence_needs_more_than_a_near_zero_mean(self):
        """A mean near zero with wide scatter must not pass as equivalent."""
        _, tost = self.evaluate([-0.02, 0.02, -0.02, 0.02, 0.0])

        self.assertFalse(tost["equivalent"])


class InstabilityTests(unittest.TestCase):
    def test_pooled_report_lists_the_flagged_folds(self):
        folds = [
            {
                "fold": 1,
                "unstable_models": ["mlp"],
                "seed_stability": {
                    "mlp": {"spread": 0.0118},
                    "vqc": {"spread": 0.0006},
                },
            },
            {
                "fold": 2,
                "unstable_models": ["mlp"],
                "seed_stability": {
                    "mlp": {"spread": 0.0134},
                    "vqc": {"spread": 0.0056},
                },
            },
        ]

        report = pooled_instability(folds)

        self.assertEqual(report["mlp"]["folds_flagged_unstable"], [1, 2])
        self.assertEqual(report["vqc"]["folds_flagged_unstable"], [])
        self.assertAlmostEqual(report["mlp"]["max_seed_spread"], 0.0134)
        self.assertAlmostEqual(report["vqc"]["max_seed_spread"], 0.0056)


if __name__ == "__main__":
    unittest.main()
