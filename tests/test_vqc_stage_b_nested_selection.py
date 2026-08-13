import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_vqc_stage_b_nested import (
    collect_validation_auprc,
    select_learning_rate,
    setting_tag,
    summarize,
)


def write_cell(root: Path, transform: str, rate: float, runs: list[dict]) -> None:
    cell = root / setting_tag(transform, rate)
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "summary.json").write_text(json.dumps({"runs": runs}))


def runs_for(model: str, values: dict[int, float]) -> list[dict]:
    return [
        {"model": model, "seed": seed, "best_val_auprc": value}
        for seed, value in values.items()
    ]


class CollectionTests(unittest.TestCase):
    def test_seeds_are_returned_in_requested_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cell(
                root,
                "raw",
                0.001,
                runs_for("mlp", {44: 0.3, 42: 0.1, 43: 0.2}),
            )

            values = collect_validation_auprc(
                root / setting_tag("raw", 0.001), "mlp", (42, 43, 44)
            )

        self.assertEqual(values, [0.1, 0.2, 0.3])

    def test_missing_seed_is_refused_rather_than_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cell(root, "raw", 0.001, runs_for("mlp", {42: 0.1, 43: 0.2}))

            with self.assertRaises(ValueError):
                collect_validation_auprc(
                    root / setting_tag("raw", 0.001), "mlp", (42, 43, 44)
                )

    def test_extension_summaries_are_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / setting_tag("raw", 0.001)
            cell.mkdir(parents=True)
            (cell / "summary.json").write_text(
                json.dumps({"runs": runs_for("mlp", {42: 0.1})})
            )
            (cell / "mlp_seed_extension.json").write_text(
                json.dumps({"runs": runs_for("mlp", {43: 0.2})})
            )

            values = collect_validation_auprc(cell, "mlp", (42, 43))

        self.assertEqual(values, [0.1, 0.2])

    def test_other_heads_in_the_same_cell_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = runs_for("mlp", {42: 0.9}) + runs_for("vqc", {42: 0.1})
            write_cell(root, "raw", 0.001, runs)

            values = collect_validation_auprc(
                root / setting_tag("raw", 0.001), "mlp", (42,)
            )

        self.assertEqual(values, [0.9])


class SelectionTests(unittest.TestCase):
    @staticmethod
    def build(low: dict[int, float], high: dict[int, float]) -> Path:
        root = Path(tempfile.mkdtemp())
        write_cell(root, "raw", 0.001, runs_for("mlp", low))
        write_cell(root, "raw", 0.01, runs_for("mlp", high))
        return root

    def test_higher_mean_validation_auprc_wins(self):
        root = self.build({42: 0.93, 43: 0.92}, {42: 0.949, 43: 0.949})

        record = select_learning_rate(root, "mlp", "raw", (0.001, 0.01), (42, 43))

        self.assertEqual(record["learning_rate"], 0.01)
        self.assertFalse(record["resolved_by_tie_rule"])
        self.assertAlmostEqual(record["validation_auprc_sensitivity"], 0.024)

    def test_near_ties_resolve_to_the_smaller_rate(self):
        root = self.build({42: 0.9400000, 43: 0.9400000}, {42: 0.9400001, 43: 0.9400001})

        record = select_learning_rate(root, "mlp", "raw", (0.001, 0.01), (42, 43))

        self.assertEqual(record["learning_rate"], 0.001)
        self.assertTrue(record["resolved_by_tie_rule"])

    def test_unstable_cell_loses_on_its_mean_not_its_best_seed(self):
        # One excellent seed must not win the cell for a rate that fails often,
        # which is the fold-3 failure mode this rule exists to correct.
        root = self.build({42: 0.947, 43: 0.926, 44: 0.929}, {42: 0.9489, 43: 0.9489, 44: 0.9489})

        record = select_learning_rate(root, "mlp", "raw", (0.001, 0.01), (42, 43, 44))

        self.assertEqual(record["learning_rate"], 0.01)

    def test_candidate_rates_are_all_reported(self):
        root = self.build({42: 0.93}, {42: 0.94})

        record = select_learning_rate(root, "mlp", "raw", (0.001, 0.01), (42,))

        self.assertEqual(
            [row["learning_rate"] for row in record["candidates"]], [0.001, 0.01]
        )


class SummaryTests(unittest.TestCase):
    @staticmethod
    def selections(mlp_rate: float, vqc_rate: float) -> dict[str, dict]:
        return {
            "mlp": {"model": "mlp", "learning_rate": mlp_rate},
            "vqc": {"model": "vqc", "learning_rate": vqc_rate},
        }

    def test_summary_is_accepted_by_the_locked_evaluator_contract(self):
        summary = summarize(self.selections(0.001, 0.01), fold=1)

        self.assertIs(summary["held_out_test_evaluated"], False)
        self.assertEqual(summary["selection_endpoint"], "validation_auprc")
        self.assertEqual(set(summary["best_by_model"]), {"mlp", "vqc"})
        self.assertEqual(summary["analysis_role"], "secondary")

    def test_matching_the_locked_rates_is_flagged(self):
        summary = summarize(self.selections(0.001, 0.01), fold=1)

        self.assertTrue(summary["identical_to_locked_v2"])

    def test_departing_from_the_locked_mlp_rate_is_flagged(self):
        summary = summarize(self.selections(0.01, 0.01), fold=3)

        self.assertFalse(summary["identical_to_locked_v2"])
        self.assertFalse(summary["reproduces_locked_v2_choice"]["mlp"])
        self.assertTrue(summary["reproduces_locked_v2_choice"]["vqc"])


if __name__ == "__main__":
    unittest.main()
