"""Modality identifiers for the unified framework."""

MODALITIES = ("mammo", "ultrasound", "thermo")
HISTOPATH_MODALITY = "histopath"
ALL_MODALITIES = MODALITIES + (HISTOPATH_MODALITY,)
MODALITY_TOKENS = ("[MAMMO]", "[ULTRA]", "[THERMO]", "[HISTO]")
MODALITY_TO_ID = {m: i for i, m in enumerate(ALL_MODALITIES)}
ID_TO_MODALITY = {i: m for m, i in MODALITY_TO_ID.items()}
HISTOPATH_MODALITY_ID = 0  # single-modality histopath models use token index 0
LABEL_NAMES = ("benign", "malignant")
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
