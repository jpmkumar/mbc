"""Image transforms for breast imaging modalities."""

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, modality: str | None = None):
    """Training augmentations; milder for grayscale mammo/US."""
    rotation = 5 if modality in ("mammo", "ultrasound") else 15
    jitter = (0.1, 0.1) if modality in ("mammo", "ultrasound") else (0.2, 0.2)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(rotation),
            transforms.ColorJitter(brightness=jitter[0], contrast=jitter[1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
