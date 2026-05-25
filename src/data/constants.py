"""Modality identifiers for the unified framework."""

MODALITIES = ("mammo", "ultrasound", "thermo")
MODALITY_TOKENS = ("[MAMMO]", "[ULTRA]", "[THERMO]")
MODALITY_TO_ID = {m: i for i, m in enumerate(MODALITIES)}
ID_TO_MODALITY = {i: m for m, i in MODALITY_TO_ID.items()}
LABEL_NAMES = ("benign", "malignant")
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
