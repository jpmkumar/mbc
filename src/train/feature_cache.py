"""Pre-extract compressed features when the classical backbone is frozen."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _patient_id_string(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


@torch.no_grad()
def extract_compressed_features(
    model,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Caching features",
    use_amp: bool = False,
    amp_device_type: str = "cuda",
) -> dict[str, torch.Tensor | list[str]]:
    """Run frozen backbone once and store 8-d compressed vectors."""
    was_training = model.training
    model.eval()
    if hasattr(model, "set_backbone_eval_mode"):
        model.set_backbone_eval_mode(True)

    features, labels, modality_ids = [], [], []
    patient_ids: list[str] = []
    sample_offset = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        mods = batch["modality_id"].to(device, non_blocking=True)
        with torch.autocast(amp_device_type, enabled=use_amp):
            compressed = model.forward_features(images, mods)
        features.append(compressed.detach().cpu())
        labels.append(batch["label"])
        modality_ids.append(batch["modality_id"])
        batch_patient_ids = batch.get("patient_id")
        if batch_patient_ids is None:
            patient_ids.extend(
                f"sample_{sample_offset + index}"
                for index in range(len(batch["label"]))
            )
        else:
            patient_ids.extend(
                _patient_id_string(value) for value in batch_patient_ids
            )
        sample_offset += len(batch["label"])

    if was_training:
        model.train()
    if hasattr(model, "set_backbone_eval_mode"):
        model.set_backbone_eval_mode(model._backbone_frozen)

    cached_features = torch.cat(features)
    if len(patient_ids) != len(cached_features):
        raise RuntimeError(
            "Patient IDs are not aligned with cached feature rows: "
            f"{len(patient_ids)} IDs for {len(cached_features)} features."
        )
    return {
        "features": cached_features,
        "labels": torch.cat(labels),
        "modality_ids": torch.cat(modality_ids),
        "patient_ids": patient_ids,
    }


def build_feature_loader(
    cached: dict[str, torch.Tensor | list[str]],
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
