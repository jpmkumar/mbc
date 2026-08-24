# Campus GPU runbook — Paper C

**Server:** NVIDIA RTX A4000 (16 GB), Intel i9-12900K, 62 GB RAM — the same
qualified host used for the Paper B width matrix
([`../../SERVER_HISTOPATH.md`](../../SERVER_HISTOPATH.md)). That matrix is
complete and no GPU work is queued, so the card is free.

**Conventions inherited from Paper B:** immutable digest-pinned image, code baked
from `git archive` rather than bind-mounted, data mounted read-only, full
provenance capture per run, refuse-to-overwrite on completed results, `tmux` for
long jobs. Do not deviate — Paper C's credibility rests on protocol discipline.

---

## Order of work

The GPU is **not** the bottleneck right now. Two things must happen first, and
only one of them needs the card.

| Step | Needs GPU | Blocking? | Why now |
|---|---|---|---|
| 0a. Request HuggingFace access to the gated encoders | no | **yes, long lead time** | approval can take days |
| 0b. Patient-grouping audit | no | **yes, blocks Phase 1** | defines what patient-disjoint means |
| 0c. Three-way inner splits | no | blocks Phase 1 | needs 0b settled |
| 1. Foundation-model embedding extraction | **yes** | no | safe to run before preregistration |
| 2. Leakage experiment (C1) | **yes** | after 0b + preregistration | the headline |

### Why embedding extraction is safe to run before preregistration

It is a frozen forward pass. It consumes no labels, makes no model selection, and
is independent of the fold structure — the embedding of a patch is a function of
the image alone. Caching it early cannot bias any later hypothesis test, and it
converts every downstream probe and ablation from hours into minutes. This is the
right way to spend an idle A4000 while Phase 0 is settled on CPU.

---

## Step 0a — Gated model access *(do this first, today)*

All four candidate encoders are gated on HuggingFace and require manual approval:

| Model | Repo | Architecture | Embedding dim |
|---|---|---|---|
| UNI | `MahmoodLab/UNI` | ViT-L/16 @ 224 | 1024 |
| UNI2-h | `MahmoodLab/UNI2-h` | ViT-H/14 @ 224 | 1536 |
| Virchow | `paige-ai/Virchow` | ViT-H/14 @ 224 | 1280 (2560 concat) |
| CONCH | `MahmoodLab/CONCH` | ViT-B/16 | 512 |

Request access for at least **UNI and Virchow** now; they are the two the plan
treats as primary. Note the CC-BY-NC-ND licence on the MahmoodLab weights —
research use is fine, but the licence must be stated in the paper.

Then on the server:

```bash
export HF_TOKEN=<your token>
huggingface-cli whoami          # confirm the token is live
```

Never commit the token. Keep it in `$HOME/mbc-primary/.env`, which is outside
the repo.

## Step 0b — Patient-grouping audit *(CPU, minutes)*

Resolves the 162-slides versus 279-directories question from the archive itself,
by parsing the `{stem}_idx{N}_x{X}_y{Y}_class{C}.png` filename pattern. If any
directory carries more than one `idx` value, or any filename stem spans two
directories, then directories alias slides and the frozen folds must be
re-audited before Phase 1.

```bash
cd "$HOME/projects/mbc"
python papers/paper_c/scripts/audit_patient_grouping.py \
  --archive-path "$HOME/mbc-primary/datasets/breast-histopathology-images" \
  --output papers/paper_c/results/phase0_patient_grouping.json
```

**DONE 2026-08-21.** An inline equivalent of this script ran on the server and
returned 279 directories, 277,524 patches, 78,786 positives, zero unparsed
filenames, and no directory aliasing. Folds stand; C1 is unblocked. Full result
and the important caveat about `idx` in
[`results/phase0_patient_grouping.md`](results/phase0_patient_grouping.md).

The committed script above has still not itself been executed — it additionally
records per-directory coordinate extents for the data card, which the inline
version skipped.

## Step 0c — Three-way inner splits *(CPU)*

`src/data/histopath_splits.py` already provides `split_train_val_patients()`,
which carves a patient-level validation set out of a fold's training patients,
stratified by IDC-ratio bin. Paper C needs a **third** split — an inner
calibration set used only for temperature scaling and the conformal quantile, per
[`VALIDATION.md`](VALIDATION.md) §1.

Extend that function rather than writing a new one, keep the same ratio-bin
stratification and seed convention, and freeze the output alongside the existing
fold files. Do not regenerate the outer folds.

## Step 0d — Build the Paper C image *(one time)*

Paper B's image bakes PennyLane and has no `timm`, so Paper C gets its own image
built the same way. Paper B's files are untouched, so its results stay
reproducible.

```bash
cd "$HOME/projects/mbc"
git pull --ff-only
chmod +x papers/paper_c/scripts/*.sh

# 1. Resolve the dependency delta against the pinned NGC base (one time).
./papers/paper_c/scripts/resolve_paperc_lock.sh
# -> prints the lock SHA-256; commit docker/requirements-paperc.lock

# 2. Pin the hash and build. The tree must be clean; the builder refuses
#    to bake a dirty source tree, exactly as Paper B's does.
export MBC_PAPERC_LOCK_SHA=<sha printed above>
./papers/paper_c/scripts/build_paperc_image.sh
```

The build ends with a CUDA + `timm` qualification check and writes
`results/server_setup/current_paperc_image.env`.

Store the HuggingFace token outside the repo, where the launcher reads it:

```bash
echo 'HF_TOKEN=hf_...' >> "$HOME/mbc-primary/.env"
chmod 600 "$HOME/mbc-primary/.env"
```

## Step 1 — Embedding extraction *(GPU)*

Throughput estimate for the A4000 in fp16, batched: a ViT-L/16 at 224² should
clear the full 277,524 patches in well under an hour; a ViT-H/14 in one to two
hours. Storage is modest — fp16 at 1024 dimensions is roughly 570 MB per encoder.

Two input transforms must be cached **separately**, because the difference
between them is the C2 experiment:

- `upsample224` — the 50×50 patch bilinearly resized to 224², the naive baseline;
- `mosaic{K}` — a K×K neighbourhood assembled from adjacent patches using the
  `xX_yY` coordinates, then resized. This recovers genuine field of view at the
  same magnification.

`extract_embeddings.py` keys each cache as
`$HOME/mbc-primary/embeddings/{encoder}_{transform}/` and records the transform,
encoder repo, pooling rule, image ID, git commit and dataset hash in
`provenance.json`. It writes float16 into a preallocated `.npy` memmap in shards,
checkpoints completed shards, and resumes after an interruption. It refuses to
overwrite a cache already marked complete.

**Smoke test first** — 5,000 patches, about a minute, proves gated download,
transform and cache format end to end:

```bash
tmux new -s paperc-embed
cd "$HOME/projects/mbc"
./papers/paper_c/scripts/run_extract_embeddings.sh uni upsample224 --limit 5000
```

Delete `$HOME/mbc-primary/embeddings/uni_upsample224` afterwards, since the
smoke cache is marked complete at 5,000 rows and would block the full pass.

Then the six real passes, one at a time:

```bash
for t in upsample224 mosaic3 mosaic5; do
  ./papers/paper_c/scripts/run_extract_embeddings.sh uni "$t"
done
for t in upsample224 mosaic3 mosaic5; do
  ./papers/paper_c/scripts/run_extract_embeddings.sh virchow "$t"
done
```

Detach with Ctrl-b then d. Monitor from a second terminal with `nvidia-smi` and
`tail -f "$HOME/mbc-primary/logs/paperc_embed_uni_upsample224.log"`. Do not launch
a second GPU job concurrently.

**Mosaic caveat that affects experiment design.** Mosaics assemble neighbouring
patches, and under a *random patch split* a neighbour of a test patch can sit in
the training set. That is a second leakage channel and would confound C1, so keep
C1 on `upsample224`. Under patient-disjoint splits the issue vanishes, because all
neighbours belong to the same patient and therefore the same side of the split.
That mosaics make random patch splitting even leakier is a reportable observation
in its own right.

## Step 2 — Leakage experiment *(GPU, after preregistration)*

Blocked on 0b, 0c, and a preregistration document under `preregistration/`
declaring the primary endpoint and multiplicity rule. Both arms — random patch
split and patient-disjoint split — must select hyperparameters from their own
inner-validation data under an identical declared budget, per
[`VALIDATION.md`](VALIDATION.md) §4. A rate locked from one arm and applied to the
other would inflate the measured leakage gap in the paper's own favour.

---

## Status

Written and syntax-checked, but **not yet executed on the server**:

- `scripts/extract_embeddings.py` — the extraction pass;
- `scripts/run_extract_embeddings.sh` — provenance-capturing launcher;
- `scripts/build_paperc_image.sh`, `scripts/resolve_paperc_lock.sh`,
  `docker/Dockerfile.paperc` — the Paper C image;
- `scripts/audit_patient_grouping.py` — an inline equivalent ran and passed, but
  this committed version, which also records coordinate extents, has not.

Still to write:

- the three-way inner train/val/cal split (extend `split_train_val_patients()`
  in `src/data/histopath_splits.py`; CPU, does not block extraction);
- the Phase 1 preregistration under `preregistration/`;
- the C1 training runner.

Expect the first real run to need small fixes — the encoder `create_model`
keyword sets for gated repos are the most likely failure point, and Virchow's
pooling recipe concatenates the class token with the mean of the patch tokens for
2,560 dimensions, which the script probes at startup rather than assuming.
