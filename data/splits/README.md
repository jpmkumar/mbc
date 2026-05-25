# Train/val/test splits

CSV manifests are **generated locally** — not stored on GitHub.

After placing images under `data/processed/`, run:

```bash
python data/download/setup_datasets.py
```

This creates `train.csv`, `val.csv`, `test.csv`, and `split_stats.json`.

**Colab:** mount Drive → unzip `mbc_mammo.zip` → run `setup_datasets.py` → train.
