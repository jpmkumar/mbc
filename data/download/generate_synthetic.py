#!/usr/bin/env python3
"""Generate synthetic multi-modality dataset for pipeline validation."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MODALITIES = {
    "mammo": {"color": (180, 180, 200), "texture": "smooth"},
    "ultrasound": {"color": (60, 60, 80), "texture": "speckle"},
    "thermo": {"color": (200, 80, 80), "texture": "gradient"},
}


def _make_image(modality: str, label: int, size: int = 224, seed: int = 0) -> Image.Image:
    rng = np.random.RandomState(seed)
    cfg = MODALITIES[modality]
    base = np.full((size, size, 3), cfg["color"], dtype=np.uint8)
    img = Image.fromarray(base)

    draw = ImageDraw.Draw(img)
    cx, cy = size // 2 + rng.randint(-20, 20), size // 2 + rng.randint(-20, 20)
    radius = 30 + (20 if label == 1 else 0) + rng.randint(-5, 5)

    if label == 1:  # malignant — irregular mass
        points = []
        for angle in range(0, 360, 30):
            r = radius + rng.randint(-8, 8)
            rad = np.deg2rad(angle)
            points.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
        draw.polygon(points, fill=(220, 220, 240) if modality == "mammo" else (120, 120, 140))
    else:  # benign — smooth oval
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(200, 200, 220) if modality == "mammo" else (100, 100, 120),
        )

    if cfg["texture"] == "speckle":
        arr = np.array(img)
        noise = rng.randint(0, 40, arr.shape, dtype=np.uint8)
        arr = np.clip(arr.astype(int) + noise - 20, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif cfg["texture"] == "gradient":
        arr = np.array(img, dtype=float)
        y = np.linspace(0, 1, size).reshape(-1, 1)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + y * 60, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    return img.filter(ImageFilter.GaussianBlur(radius=1))


def generate_dataset(output_root: str, samples_per_class: int = 50, seed: int = 42):
    output_root = Path(output_root)
    rng = np.random.RandomState(seed)

    for modality in MODALITIES:
        for label_name, label in [("benign", 0), ("malignant", 1)]:
            out_dir = output_root / modality / label_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(samples_per_class):
                s = int(rng.randint(0, 1e6))
                img = _make_image(modality, label, seed=s)
                img.save(out_dir / f"{modality}_{label_name}_{i:04d}.png")

    print(f"Generated synthetic dataset at {output_root}")
    print(f"  Modalities: {list(MODALITIES.keys())}")
    print(f"  Samples per class per modality: {samples_per_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, args.samples, args.seed)
