"""Modality-specific image preprocessing (mammography CLAHE, grayscale, etc.)."""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


DEFAULT_MODALITY_PREPROCESS = {
    "mammo": {"grayscale": True, "clahe": True, "clip_limit": 2.0, "grid_size": 8},
    "ultrasound": {"grayscale": True, "clahe": True, "clip_limit": 2.0, "grid_size": 8},
    "thermo": {"grayscale": False, "clahe": False, "clip_limit": 2.0, "grid_size": 8},
    "histopath": {"grayscale": False, "clahe": False, "clip_limit": 2.0, "grid_size": 8},
}


def _resolve_modality_cfg(modality: str, preprocess_config: dict) -> dict:
    if not preprocess_config.get("enabled", False):
        return {}
    defaults = DEFAULT_MODALITY_PREPROCESS.get(modality, {})
    overrides = preprocess_config.get(modality, {})
    merged = {**defaults, **overrides}
    return merged


def apply_clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> Image.Image:
    if cv2 is None:
        return image
    gray = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)


def to_rgb_grayscale(image: Image.Image) -> Image.Image:
    """Single-channel mammo/US as 3-channel for ImageNet backbones."""
    gray = image.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


def preprocess_image(
    image: Image.Image,
    modality: str,
    preprocess_config: dict | None,
) -> Image.Image:
    """Apply modality-specific preprocessing before torchvision transforms."""
    if not preprocess_config:
        return image

    cfg = _resolve_modality_cfg(modality, preprocess_config)
    if not cfg:
        return image

    if cfg.get("grayscale", False):
        image = image.convert("L")
        if cfg.get("clahe", False):
            image = apply_clahe(
                image,
                clip_limit=cfg.get("clip_limit", 2.0),
                grid_size=cfg.get("grid_size", 8),
            )
        image = to_rgb_grayscale(image)
    elif cfg.get("clahe", False):
        image = apply_clahe(
            image,
            clip_limit=cfg.get("clip_limit", 2.0),
            grid_size=cfg.get("grid_size", 8),
        ).convert("RGB")

    return image


def preprocess_cache_tag(preprocess_config: dict | None) -> str:
    if not preprocess_config or not preprocess_config.get("enabled", False):
        return "raw"
    parts = ["pre"]
    for modality in ("mammo", "ultrasound", "thermo"):
        cfg = _resolve_modality_cfg(modality, preprocess_config)
        if cfg.get("grayscale"):
            parts.append(f"{modality[:2]}g")
        if cfg.get("clahe"):
            parts.append(f"{modality[:2]}c")
    return "_".join(parts) if len(parts) > 1 else "pre"
