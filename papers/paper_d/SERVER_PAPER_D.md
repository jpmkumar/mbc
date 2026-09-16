# Paper D — RTX A4000 BreakHis runbook

Campus GPU server commands for the four confirmatory arms
(`F-MLP`, `F-VQC`, `T-MLP`, `T-VQC`) × five case-ID folds.

Governing protocol: `preregistration/paper_d_breakhis_protocol.md`.
Lock must verify before labelled runs.

## 1. Pull code

```bash
cd "$HOME/projects/mbc"   # or your clone path
git fetch origin --prune
git switch main
git pull --ff-only
git log -1 --oneline
```

## 2. Environment

Use the same venv / histopath image you use for Paper B (PennyLane + torch).

```bash
source .venv/bin/activate   # or enter the qualified container
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. BreakHis archive (once)

Copy `BreaKHis_v1.tar.gz` onto the server (scp/rsync), then:

```bash
mkdir -p papers/paper_d/data
# if the tarball is elsewhere, point SRC at it:
SRC="${SRC:-$HOME/mbc-primary/datasets/BreaKHis_v1.tar.gz}"
cp -f "$SRC" papers/paper_d/data/BreaKHis_v1.tar.gz

# must match the lock
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
python papers/paper_d/scripts/verify_paperd_lock.py
# must print: LOCK VERIFY OK
```

## 5. Full matrix (recommended)

```bash
# dry-run first
papers/paper_d/scripts/run_breakhis_queue.sh --dry-run

# all 5 folds × 4 arms (skips cells that already have summary.json)
papers/paper_d/scripts/run_breakhis_queue.sh
```

Artifacts: `results/breakhis_paper_d/fold{N}_{arm}/summary.json`

## 6. Single cell

```bash
python papers/paper_d/scripts/prepare_breakhis_manifests.py --fold 0
python papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm F-MLP
python papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm F-VQC
python papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm T-MLP
python papers/paper_d/scripts/train_breakhis_cv.py --fold 0 --arm T-VQC
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
