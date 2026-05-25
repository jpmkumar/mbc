# CBIS-DDSM Mammography Download Guide

**Source:** [CBIS-DDSM on TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)  
**DOI:** [10.7937/K9/TCIA.2016.7O02S9CY](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

## Option A — Automated (recommended, no NBIA install)

Uses TCIA REST API + label CSVs. **No registration required** for CBIS-DDSM.

```bash
source .venv/bin/activate
pip install pydicom requests pandas tqdm pillow

# Test subset: 100 benign + 100 malignant cropped mammograms (~3 GB)
python data/download/download_cbis_ddsm.py

# Full dataset: ~3,500 cropped cases (tens of GB, hours)
python data/download/download_cbis_ddsm.py --full
```

Output layout:
```
data/processed/mammo/
  benign/*.png
  malignant/*.png
data/raw/cbis-ddsm/
  mass_case_description_*.csv
  calc_case_description_*.csv
  download_progress.json   # resume support
```

## Option B — Official NBIA Data Retriever

1. Register at [TCIA](https://www.cancerimagingarchive.net/)
2. Install [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
3. Download manifest: [CBIS-DDSM-All.tcia](https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia) (163 GB full)
4. Or smaller subsets (cropped only):
   - [Mass-Training cropped](https://www.cancerimagingarchive.net/wp-content/uploads/Mass-Training_ROI-mask_and_crpped_images_1-doiJNLP-07gmVj4b.tcia)
   - [Calc-Training cropped](https://www.cancerimagingarchive.net/wp-content/uploads/Calc-Training_ROI-mask_and_crpped_images-doiJNLP-kTGQKqBk.tcia)

## Labels

Pathology labels come from TCIA CSV files:
- `pathology`: **BENIGN** or **MALIGNANT**
- Cropped images used by default (lesion-focused, smaller than full mammograms)

## Citation (required in paper)

```
Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016).
Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
```
