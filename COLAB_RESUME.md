# Resume Training Across Colab Sessions

Training saves a checkpoint **after every epoch**. If Colab disconnects, you can continue without starting over.

## Checkpoints saved

| File | Purpose |
|------|---------|
| `results/checkpoints/E3_hybrid_seed42_latest.pt` | Full resume state (epoch progress, history) |
| `results/checkpoints/E3_hybrid_seed42.pt` | Best model weights |
| `results/E3_hybrid_seed42_progress.json` | Human-readable progress |

**Copy to Drive after each session:**

```python
!mkdir -p /content/drive/MyDrive/mbc_colab/checkpoints
!cp -r results/checkpoints/* /content/drive/MyDrive/mbc_colab/checkpoints/
!cp results/*_progress.json /content/drive/MyDrive/mbc_colab/results/ 2>/dev/null
```

**Restore before resuming:**

```python
!mkdir -p results/checkpoints
!cp /content/drive/MyDrive/mbc_colab/checkpoints/* results/checkpoints/
```

---

## Option A — Auto-resume (easiest)

If `_latest.pt` exists, training continues automatically:

```python
%cd /content/mbc
!git pull
!python data/download/setup_datasets.py
!python experiments/run_training.py --experiment hybrid --modality mammo
```

---

## Option B — One stage per Colab session

### Session 1 — Stage A (classical head, ~2 h)

```python
!python experiments/run_training.py --experiment hybrid --modality mammo --stage a --no-auto-resume
!cp -r results/checkpoints/* /content/drive/MyDrive/mbc_colab/checkpoints/
```

### Session 2 — Stage B (VQC, ~3–4 h)

```python
!cp /content/drive/MyDrive/mbc_colab/checkpoints/* results/checkpoints/
!python experiments/run_training.py --experiment hybrid --modality mammo --stage b
!cp -r results/checkpoints/* /content/drive/MyDrive/mbc_colab/checkpoints/
```

### Session 3 — Stage C (joint fine-tune, ~30 min)

```python
!cp /content/drive/MyDrive/mbc_colab/checkpoints/* results/checkpoints/
!python experiments/run_training.py --experiment hybrid --modality mammo --stage c
```

---

## Option C — Manual resume path

```python
!python experiments/run_training.py --experiment hybrid --modality mammo \
  --resume results/checkpoints/E3_hybrid_seed42_latest.pt
```

---

## Fresh start (ignore old checkpoints)

```python
!python experiments/run_training.py --experiment hybrid --modality mammo --no-auto-resume
```

Or delete old checkpoints:

```python
!rm -f results/checkpoints/E3_hybrid_seed42*.pt
```

---

## Check progress

```python
!cat results/E3_hybrid_seed42_progress.json
```

Example:

```json
{
  "stage_epochs_done": {"stage_a": 20, "stage_b": 12, "stage_c": 0},
  "stage_targets": {"stage_a": 20, "stage_b": 30, "stage_c": 5}
}
```

Stage B is 12/30 — safe to resume.

---

## Push code updates from Mac

```bash
cd "/Users/muthu/ResTest/paper1/mbc"
git add .
git commit -m "Add --stage and --resume training"
git push
```

Colab: `!git pull` before resuming.
