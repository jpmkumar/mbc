# Paper D — RTX A4000 BreakHis runbook

Campus GPU server commands for the four confirmatory arms
(`F-MLP`, `F-VQC`, `T-MLP`, `T-VQC`) × five case-ID folds.

Governing protocol: `preregistration/paper_d_breakhis_protocol.md`.
Lock must verify before labelled runs.

## Qualified host (inventory)

| Item | Value |
|------|--------|
| Hostname | `SJT516SCOPE052` |
| User | `kumar_24phd0656` |
| Clone | `~/projects/mbc` |
| Data root | `~/mbc-primary` |
| GPU | NVIDIA RTX A4000 16 GB |
| CPU / RAM | Intel i9-12900K / ~62 GB |
| Driver (last recorded) | 580.173.02 |
| Stack | NGC PyTorch 26.06 image and/or host `.venv` (PennyLane 0.45.1) |

Use **`python3`** or `.venv/bin/python` — bare `python` may be missing.
GitHub HTTPS password auth fails; use SSH or a PAT for `git pull`.

## 1. Pull code

```bash
cd "$HOME/projects/mbc"
git fetch origin --prune
git switch main
git pull --ff-only
git log -1 --oneline
```

## 2. Environment

```bash
source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

## 3. BreakHis archive (once)

Copy `BreaKHis_v1.tar.gz` onto the server (scp/rsync), then:

```bash
mkdir -p papers/paper_d/data
SRC="${SRC:-$HOME/mbc-primary/datasets/BreaKHis_v1.tar.gz}"
cp -f "$SRC" papers/paper_d/data/BreaKHis_v1.tar.gz

EXPECTED=d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640
sha256sum papers/paper_d/data/BreaKHis_v1.tar.gz
# expect: $EXPECTED

tar -xzf papers/paper_d/data/BreaKHis_v1.tar.gz -C papers/paper_d/data
test -d papers/paper_d/data/BreaKHis_v1/histology_slides
find papers/paper_d/data/BreaKHis_v1 -name '*.png' | wc -l
# expect: 7909
```

## 4. Verify lock

```bash
python3 papers/paper_d/scripts/verify_paperd_lock.py
# must print: LOCK VERIFY OK
```

## 5. Full matrix (recommended)

ntfy uses the same topic file as Path A / width queue:
`~/.config/mbc/ntfy_topic` (or `$MBC_NTFY_TOPIC_FILE`). If that file is
missing, the queue still runs; notifications are skipped.

```bash
# confirm topic (optional)
cat ~/.config/mbc/ntfy_topic

papers/paper_d/scripts/run_breakhis_queue.sh --dry-run
papers/paper_d/scripts/run_breakhis_queue.sh
# you get: Paper D started → per-cell Paper D i/N → Paper D complete (or FAILED)
```

Artifacts: `results/breakhis_paper_d/fold{N}_{arm}/summary.json`  
Queue log: `~/mbc-primary/logs/paper_d_queue.log`

## 6. Single cell

```bash
python3 papers/paper_d/scripts/prepare_breakhis_manifests.py --fold 0
python3 papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm F-MLP
python3 papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm F-VQC
python3 papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm T-MLP
python3 papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm T-VQC
```

## 7. Detached session (tmux)

```bash
tmux new -s paperd
cd "$HOME/projects/mbc" && source .venv/bin/activate
papers/paper_d/scripts/run_breakhis_queue.sh 2>&1 | tee results/breakhis_paper_d/queue.log
# detach: Ctrl-b d
```

## Notes

- Do not retune hyperparameters after any test fold is scored.
- Runtime manifests under `data/splits/breakhis/runtime/` are rebuildable.
- Archive and extracted slides stay local; never commit them.
