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
    preprocess_config: dict | None = None,
    prefetch_factor: int = 2,
    data_root: str | None = None,
    max_samples: int | None = None,
    max_eval_samples: int | None = None,
    augment_config: dict | None = None,
):
    train_modality = modality_filter[0] if modality_filter and len(modality_filter) == 1 else None
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    loaders = {}
    for split_name, manifest_path in splits.items():
        if split_name not in ("train", "val", "test"):
            continue
        if split_name == "train" and not eval_train_transforms:
            transform = get_train_transforms(
                image_size,
                modality=train_modality,
                augment_config=augment_config,
            )
        else:
            transform = get_eval_transforms(image_size)
        limit = max_samples if split_name == "train" else max_eval_samples
        dataset = UnifiedBreastDataset(
            manifest_path,
            transform=transform,
            modality_filter=modality_filter,
            preprocess_config=preprocess_config,
            train_modality=train_modality,
            data_root=data_root,
            max_samples=limit,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **loader_kwargs,
        )
    return loaders
