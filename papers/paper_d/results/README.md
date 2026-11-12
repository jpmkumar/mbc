# Paper D curated results

Raw GPU runs: repository-root `results/breakhis_paper_d/` (gitignored).

| File | Contents |
|------|----------|
| `curated_metrics.json` | Co-primary four-arm AUPRC, paired Δ, Holm |
| `tf_secondaries.json` | S1 per-mag, S2 ECE/reliability, S3 wall-clock/params |
| `breakhis_archive_audit.md` | Archive audit notes |

Regenerate:

```bash
python papers/paper_d/scripts/export_curated_metrics.py
python papers/paper_d/scripts/export_tf_secondaries.py
```

Figures land in `manuscript/figures/` (fig01–fig04).
