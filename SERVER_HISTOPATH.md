# Physical GPU server: primary histopathology experiments

The RTX A4000 server is the primary environment for the complete q4/q8/q12
width matrix. Kaggle outputs are historical/secondary replications and must
not be pooled with server results. The governing declaration is
`preregistration/histopath_vqc_width_server_protocol.md`.

## Qualified server

- NVIDIA RTX A4000 (16 GB), Intel i9-12900K, 62 GB RAM
- NVIDIA driver 580.173.02
- NGC PyTorch 26.06 base image, pinned by digest
- PyTorch 2.13.0a0 NVIDIA 26.06, CUDA runtime 13.3
- PennyLane 0.45.1
- 279 patients and 277,524 patches independently verified
- canonical five-fold manifest SHA-256:
  `ac9d06510ca3555e6d481f1f870ab92fc69411ee3b9fa53da9aa7a60ce9bd013`

Do not commit Kaggle credentials, raw data, caches, checkpoints or result
bundles. They remain under `$HOME/mbc-primary`.

## 1. Pull and verify

```bash
cd "$HOME/projects/mbc"
git fetch origin --prune
git switch docs/histopath-writing-q1-guidelines
git pull --ff-only
git status --short --branch
```

The tracked tree must be clean before building. The image builder refuses a
dirty tracked source tree.

## 2. Build the immutable experiment image

```bash
chmod +x \
  scripts/build_histopath_server_image.sh \
  scripts/run_histopath_width_server.sh

./scripts/build_histopath_server_image.sh
```

This performs two builds:

1. a digest-pinned NGC environment plus the exact dependency delta;
2. a code image built from `git archive HEAD`, not a source bind mount.

The generated image tag, image ID, commit and lock hash are recorded locally
in `results/server_setup/current_server_image.env`.

## 3. Train-only qualification smoke test

This uses real augmented training patches and the full E3 backbone/compression
architecture, but never loads validation or test samples.

```bash
PRIMARY_ROOT="$HOME/mbc-primary"
source results/server_setup/current_server_image.env

docker run --rm \
  --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  --env USER=mbc \
  --env LOGNAME=mbc \
  --env HOME=/tmp \
  --env XDG_CACHE_HOME=/cache \
  --env TORCH_HOME=/cache/torch \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src="$PRIMARY_ROOT/datasets/breast-histopathology-images",dst=/datasets/histopath,readonly \
  --mount type=bind,src="$PRIMARY_ROOT/splits/histopath",dst=/opt/mbc/data/splits/histopath \
  --mount type=bind,src="$PRIMARY_ROOT/cache",dst=/cache \
  "$MBC_IMAGE" \
  python scripts/smoke_histopath_server.py \
    --archive-path /datasets/histopath \
    --splits-dir /opt/mbc/data/splits/histopath \
    --fold 1 \
    --n-qubits 8 \
    --steps 5
```

Require `"status": "PASS"`, finite first/last loss and nonzero throughput.
Save the printed JSON under `$PRIMARY_ROOT/provenance`.

## 4. Run the twelve primary cells

Run one cell at a time on the single RTX A4000. A conservative order interleaves
widths so calendar-time or thermal drift is not confounded with width:

```text
fold 1: q8, q4, q12
fold 2: q12, q8, q4
fold 3: q4, q12, q8
fold 4: q8, q12, q4
```

Start the first cell:

```bash
./scripts/run_histopath_width_server.sh 1 8
```

The launcher automatically:

- uses the non-root host UID with a valid container identity;
- mounts data read-only and stores caches/results persistently;
- captures source, image, GPU, driver, runtime, dataset and split provenance;
- audits every stage checkpoint for non-finite values;
- refuses to overwrite a completed fold-width result;
- creates a downloadable ZIP under `$HOME/mbc-primary/bundles`.

Run it inside `tmux` so a disconnected terminal does not stop training:

```bash
tmux new -s mbc-width
./scripts/run_histopath_width_server.sh 1 8
# Detach: Ctrl-b, then d
```

Reattach with `tmux attach -t mbc-width`.

## 5. Monitor without changing the run

In another terminal:

```bash
nvidia-smi
tail -f "$HOME/mbc-primary/logs/server_width_q8_fold1.log"
```

Do not launch another GPU cell concurrently. Do not change batch size,
learning rate, epochs or code between cells.

## 6. Failure handling

- Non-finite Stage A/B/C checkpoints are recorded as numerical convergence
  failures; their metrics are not interpreted as zero.
- If validation selects a non-finite stage, the runner exits and the cell is
  invalid.
- Do not rerun a completed finite cell.
- For power, storage or host interruption, preserve the partial result and log
  before deciding whether checkpoint resume is technically possible. Record
  the interruption in the experiment log.

After all twelve bundles exist, run the predeclared paired width aggregation
on server results only. Kaggle and server estimates are reported separately.

