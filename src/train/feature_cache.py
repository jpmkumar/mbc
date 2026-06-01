"""Pre-extract compressed features when the classical backbone is frozen."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@torch.no_grad()
def extract_compressed_features(
    model,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Caching features",
) -> dict[str, torch.Tensor]:
    """Run frozen backbone once and store 8-d compressed vectors."""
    was_training = model.training
    model.eval()
    if hasattr(model, "set_backbone_eval_mode"):
        model.set_backbone_eval_mode(True)

    features, labels, modality_ids = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        mods = batch["modality_id"].to(device, non_blocking=True)
        compressed = model.forward_features(images, mods)
        features.append(compressed.detach().cpu())
        labels.append(batch["label"])
        modality_ids.append(batch["modality_id"])

    if was_training:
        model.train()
    if hasattr(model, "set_backbone_eval_mode"):
        model.set_backbone_eval_mode(model._backbone_frozen)

    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "modality_ids": torch.cat(modality_ids),
    }


def build_feature_loader(
    cached: dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        cached["features"],
        cached["labels"],
        cached["modality_ids"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
