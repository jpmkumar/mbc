from types import SimpleNamespace
import unittest

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


if __name__ == "__main__":
    unittest.main()
