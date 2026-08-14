from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.run_histopath_width_server import (
    EXPECTED_LOCK_SHA,
    EXPECTED_SPLIT_SHA,
    build_train_command,
    capture_provenance,
    validate_pair,
    verify_outer_fold,
    verify_runtime_fold,
)


def write_patients(path: Path, patients: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": sorted(patients)}).to_csv(path, index=False)


class ServerWidthRunnerTests(unittest.TestCase):
    def test_accepts_complete_declared_server_matrix(self):
        for fold in (1, 2, 3, 4):
            for width in (4, 8, 12):
                validate_pair(fold, width)
        for pair in ((0, 8), (5, 8), (1, 6)):
            with self.assertRaises(ValueError):
                validate_pair(*pair)

    def test_train_command_fixes_experiment_seed_and_split_root(self):
        command = build_train_command(
            3,
            12,
            Path("/datasets/histopath"),
            Path("/splits/histopath"),
        )
        self.assertEqual(command[command.index("--fold") + 1], "3")
        self.assertEqual(command[command.index("--n-qubits") + 1], "12")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(
            command[command.index("--splits-dir") + 1],
            "/splits/histopath",
        )

    def test_patient_checks_accept_disjoint_complete_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = {str(value) for value in range(224)}
            test = {str(value) for value in range(224, 279)}
            write_patients(root / "folds/fold_1/train.csv", train)
            write_patients(root / "folds/fold_1/test.csv", test)
            counts = verify_outer_fold(root, 1)
            self.assertEqual(counts, {"outer_train": 224, "outer_test": 55})

            runtime = {
                "train": {str(value) for value in range(180)},
                "val": {str(value) for value in range(180, 224)},
                "test": test,
            }
            for split, patients in runtime.items():
                write_patients(root / f"runtime/fold_1/{split}.csv", patients)
            self.assertEqual(
                verify_runtime_fold(root, 1),
                {"train": 180, "val": 44, "test": 55},
            )

    def test_patient_checks_reject_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = {str(value) for value in range(225)}
            test = {str(value) for value in range(224, 279)}
            write_patients(root / "folds/fold_1/train.csv", train)
            write_patients(root / "folds/fold_1/test.csv", test)
            with self.assertRaises(RuntimeError):
                verify_outer_fold(root, 1)

    def test_provenance_requires_all_immutable_identifiers(self):
        complete = {
            "MBC_GIT_COMMIT": "a" * 40,
            "MBC_IMAGE_ID": "sha256:image",
            "MBC_ENV_LOCK_SHA": EXPECTED_LOCK_SHA,
            "MBC_DATASET_SHA": "c" * 64,
            "MBC_SPLIT_MANIFEST_SHA": EXPECTED_SPLIT_SHA,
        }
        with patch.dict("os.environ", complete, clear=True), patch(
            "scripts.run_histopath_width_server.nvidia_smi_metadata",
            return_value=[{"name": "NVIDIA RTX A4000"}],
        ):
            result = capture_provenance(2, 8)
        self.assertEqual(result["source_commit"], "a" * 40)
        self.assertEqual(result["effective_seed"], 44)
        self.assertEqual(result["gpus"][0]["name"], "NVIDIA RTX A4000")

        incomplete = dict(complete)
        del incomplete["MBC_DATASET_SHA"]
        with patch.dict("os.environ", incomplete, clear=True):
            with self.assertRaises(RuntimeError):
                capture_provenance(2, 8)


if __name__ == "__main__":
    unittest.main()

