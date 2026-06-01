"""Unified dataset for mammography, ultrasound, and thermography."""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import MODALITY_TO_ID
from .preprocessing import preprocess_image


class UnifiedBreastDataset(Dataset):
    """Single dataset mixing all modalities via a manifest CSV."""

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable] = None,
        modality_filter: Optional[list[str]] = None,
        preprocess_config: Optional[dict] = None,
        train_modality: Optional[str] = None,
    ):
        self.manifest = pd.read_csv(manifest_path)
        if modality_filter:
            self.manifest = self.manifest[
                self.manifest["modality"].isin(modality_filter)
            ].reset_index(drop=True)
        self.transform = transform
        self.preprocess_config = preprocess_config
        self.train_modality = train_modality
        # Manifest paths are relative to data/processed
        self.root = Path(manifest_path).parent.parent / "processed"

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img_path = self.root / row["filepath"]
        modality = row["modality"]
        image = Image.open(img_path).convert("RGB")
        image = preprocess_image(image, modality, self.preprocess_config)
        if self.transform:
            image = self.transform(image)

        label = int(row["label"])
        modality_id = MODALITY_TO_ID[row["modality"]]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "modality_id": torch.tensor(modality_id, dtype=torch.long),
            "modality": row["modality"],
            "patient_id": row.get("patient_id", f"sample_{idx}"),
            "filepath": str(img_path),
        }
