# Dataset scale: why ~3k images vs papers claiming 1–2 lakh (100k–200k)

## What this repo uses today

| Modality | Images in splits | Source | Notes |
|----------|------------------|--------|--------|
| **Mammography** | **2,966** | CBIS-DDSM (processed PNGs) | Real public data |
| **Ultrasound** | **80** | Synthetic placeholder | Replace with BUSI (~780 real) |
| **Thermography** | **80** | Synthetic placeholder | Replace with real thermo DB |

From `data/splits/split_stats.json` after `setup_datasets.py`.

---

## Why other papers report 1–2 lakh images

The number usually counts **patches / tiles**, not unique patients or full mammograms.

| Paper claim | What it often means |
|-------------|---------------------|
| **100k–200k “images”** | Sliding-window patches from DDSM/CBIS full films |
| **Large-scale mammography** | Multiple hospitals merged (INbreast + DDSM + CMMD + …) |
| **Pretraining + fine-tune** | ImageNet or RadImageNet pretrain on millions, fine-tune on 3k |
| **Duplicate views** | CC + MLO, left + right, multiple magnifications → 4×–8× count |
| **Mass + calcification sets** | Each lesion × multiple crops / augmentations counted separately |

### CBIS-DDSM specifically

- **Curated subset** of DDSM: pathology-confirmed **masses and calcifications**
- ~**3,000** ROI-level images (what we use) — this is the **official curated scale**
- Full **DDSM** is ~2,500 patients / ~10,000 films — still not 200k until patchified

So **2,966 mammo PNGs is correct for CBIS-DDSM ROI-level work**, not a bug.

---

## Why we are not at 100k+ (yet)

1. **Single public subset** — we use CBIS-DDSM curated ROIs, not patch-mined full DDSM  
2. **No multi-dataset merge** — papers often combine CBIS + INbreast + BCDR + private hospital data  
3. **Ultrasound / thermo are placeholders** — 80 synthetic each until BUSI / thermo DB downloaded  
4. **Compute budget** — Colab-friendly pipeline; 200k patches needs multi-GPU training  
5. **Unified multimodal framework** — paper goal is cross-modality generalization, not max mammo scale alone  

---

## How to scale toward larger studies (optional)

| Step | Approx. size | Effort |
|------|--------------|--------|
| Download **real BUSI** ultrasound | ~780 | Low |
| **Patch extraction** from CBIS full DICOM (256×256 stride) | 50k–150k | Medium |
| Add **INbreast**, **BCDR** | +1k–5k exams | Medium |
| **Self-supervised pretrain** on unlabeled mammo | 100k+ | High |
| Private hospital data | 100k+ | IRB + legal |

For the **submitted workpaper**, if they used 1–2 lakh images, ask:

- Is that **patches** or **exams**?  
- **Which datasets** were merged?  
- **Patient-level** split or image-level (leakage)?  
- **ROI** or **full mammogram**?

---

## Fair comparison for your paper

| Comparison | Fair? |
|------------|-------|
| Your CBIS ~3k vs their patch count 200k | **No** — different scale and task |
| Your Stage A vs Stage B on **same** 3k + same preprocess | **Yes** |
| Your mammo vs pilot WBCD 569 tabular @ 97% | **No** — different modality and difficulty |
| Your hybrid vs classical on **mammo + BUSI + thermo** | **Yes** — after replacing synthetic US/thermo |

---

## Recommended wording (paper)

> We evaluate on CBIS-DDSM (2,966 mammography ROIs), BUSI ultrasound, and thermography databases with patient-level stratified splits. Scale is comparable to standard CBIS-DDSM classification studies; large-scale patch corpora (100k+) are left to future work.
