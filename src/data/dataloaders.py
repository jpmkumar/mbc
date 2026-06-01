"""Dataloader factory."""

import torch
from torch.utils.data import DataLoader

from .dataset import UnifiedBreastDataset
from .transforms import get_eval_transforms, get_train_transforms


def create_dataloaders(
    splits: dict[str, str],
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
    modality_filter: list[str] | None = None,
    eval_train_transforms: bool = False,
):
    loaders = {}
    for split_name, manifest_path in splits.items():
        if split_name not in ("train", "val", "test"):
            continue
        if split_name == "train" and not eval_train_transforms:
            transform = get_train_transforms(image_size)
        else:
            transform = get_eval_transforms(image_size)
        dataset = UnifiedBreastDataset(
            manifest_path,
            transform=transform,
            modality_filter=modality_filter,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders
