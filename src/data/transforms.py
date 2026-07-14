"""Image transforms for breast imaging modalities."""

import random

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomDiscreteRotation:
    """Rotate a PIL image by a random multiple of 90 degrees.

    Discrete 90-degree steps avoid interpolation artifacts and black
    borders, which is ideal for square histopathology patches that are
    rotation-invariant.
    """

    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = list(angles)

    def __call__(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        return img.rotate(angle)


def get_train_transforms(
    image_size: int = 224,
    modality: str | None = None,
    augment_config: dict | None = None,
):
    """Training augmentations; milder for grayscale mammo/US.

    When ``augment_config['strong']`` is set (recommended for histopath),
    adds vertical flips, full 90-degree rotations, and stronger color
    jitter (saturation/hue) to better cover H&E stain variation.
    """
    augment_config = augment_config or {}
    strong = bool(augment_config.get("strong", False))

    rotation = 5 if modality in ("mammo", "ultrasound") else 15
    jitter = (0.1, 0.1) if modality in ("mammo", "ultrasound") else (0.2, 0.2)

    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if strong:
        ops.append(transforms.RandomVerticalFlip(p=0.5))
        ops.append(RandomDiscreteRotation((0, 90, 180, 270)))
        ops.append(
            transforms.ColorJitter(
                brightness=augment_config.get("brightness", 0.2),
                contrast=augment_config.get("contrast", 0.2),
                saturation=augment_config.get("saturation", 0.1),
                hue=augment_config.get("hue", 0.05),
            )
        )
    else:
        ops.append(transforms.RandomRotation(rotation))
        ops.append(
            transforms.ColorJitter(brightness=jitter[0], contrast=jitter[1])
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def get_eval_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
