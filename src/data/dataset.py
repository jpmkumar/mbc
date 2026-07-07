"""Unified dataset for mammography, ultrasound, and thermography."""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import HISTOPATH_MODALITY, HISTOPATH_MODALITY_ID, MODALITY_TO_ID
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
        data_root: str | Path | None = None,
        max_samples: int | None = None,
        max_eval_samples: int | None = None,
    ):
        self.manifest = pd.read_csv(manifest_path)
        if modality_filter:
            self.manifest = self.manifest[
                self.manifest["modality"].isin(modality_filter)
            ].reset_index(drop=True)
        sample_limit = max_samples
        if sample_limit is None and max_eval_samples is not None:
            sample_limit = max_eval_samples
        if sample_limit is not None and len(self.manifest) > sample_limit:
            self.manifest = self.manifest.sample(
                n=sample_limit, random_state=42
            ).reset_index(drop=True)
        self.transform = transform
        self.preprocess_config = preprocess_config
        self.train_modality = train_modality
        if data_root is not None:
            self.root = Path(data_root)
        elif len(self.manifest) and self.manifest.iloc[0]["modality"] == HISTOPATH_MODALITY:
            raise ValueError(
                "Histopath manifests require data_root pointing to the archive folder."
            )
        else:
            # Default layout for mammo / ultrasound / thermo
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
        modality_id = (
            HISTOPATH_MODALITY_ID
            if modality == HISTOPATH_MODALITY
            else MODALITY_TO_ID[row["modality"]]
        )
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "modality_id": torch.tensor(modality_id, dtype=torch.long),
            "modality": row["modality"],
            "patient_id": row.get("patient_id", f"sample_{idx}"),
            "filepath": str(img_path),
        }
