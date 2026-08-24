import unittest

import torch

from src.train.feature_cache import (
    build_feature_loader,
    extract_compressed_features,
)


class ToyFeatureModel(torch.nn.Module):
    def forward_features(self, images, modality_ids):
        return images.flatten(start_dim=1) + modality_ids[:, None].float()


class FeatureCacheTests(unittest.TestCase):
    def test_extraction_preserves_patient_alignment(self):
        model = ToyFeatureModel()
        batches = [
            {
                "image": torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]]),
                "label": torch.tensor([0, 1]),
                "modality_id": torch.tensor([0, 0]),
                "patient_id": torch.tensor([10253, 10254]),
            },
            {
                "image": torch.tensor([[[[5.0, 6.0]]]]),
                "label": torch.tensor([1]),
                "modality_id": torch.tensor([0]),
                "patient_id": torch.tensor([10253]),
            },
        ]

        cached = extract_compressed_features(
            model,
            batches,
            torch.device("cpu"),
            use_amp=False,
        )

        self.assertEqual(
            cached["patient_ids"],
            ["10253", "10254", "10253"],
        )
        self.assertEqual(len(cached["features"]), len(cached["patient_ids"]))
        torch.testing.assert_close(
            cached["labels"],
            torch.tensor([0, 1, 1]),
        )

    def test_feature_loader_ignores_metadata_columns(self):
        cached = {
            "features": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "labels": torch.tensor([0, 1]),
            "modality_ids": torch.tensor([0, 0]),
            "patient_ids": ["patient_a", "patient_b"],
        }

        batch = next(iter(build_feature_loader(cached, batch_size=2)))

        self.assertEqual(len(batch), 3)
        torch.testing.assert_close(batch[0], cached["features"])


if __name__ == "__main__":
    unittest.main()
