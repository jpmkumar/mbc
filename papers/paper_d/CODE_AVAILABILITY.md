# Code availability — Paper D (BreakHis hybrid QML transfer)

Use these paths when citing the GitHub repository in **Paper D** (BreakHis
multi-magnification histopathology, case-ID-grouped CV, Paper-B–matched VQC vs
E2b MLP under freeze / fine-tune).

## Repository

| Item | Value |
|------|-------|
| Repository | https://github.com/jpmkumar/mbc |
| Paper D workspace | [`papers/paper_d/`](.) |

**Stable URL for submission** (after you tag a release):

```text
https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_d
```

Example:

```text
https://github.com/jpmkumar/mbc/tree/paper-d-v1.0/papers/paper_d
```

## What belongs to Paper D

| Path | Role |
|------|------|
| [`papers/paper_d/`](.) | Paper D index, config, lock scripts |
| [`preregistration/paper_d_breakhis_protocol.md`](../../preregistration/paper_d_breakhis_protocol.md) | Binding protocol |
| [`preregistration/paper_d_lock.json`](../../preregistration/paper_d_lock.json) | Lock manifest (after Stage 2) |
| [`data/splits/breakhis/`](../../data/splits/breakhis/) | Case-ID fold membership (when frozen) |
| [`config/circuit_registry.json`](config/circuit_registry.json) | Matched VQC / MLP specs |
| Shared [`src/models/vqc.py`](../../src/models/vqc.py) | Circuit family shared with Paper B |
| Shared [`src/models/hybrid_model.py`](../../src/models/hybrid_model.py) | Hybrid stack |

## What is **not** Paper D

| Path | Belongs to |
|------|------------|
| [`papers/paper_a/`](../../papers/paper_a/) | **Paper A** (tabular WBCD) |
| [`papers/paper_b/`](../../papers/paper_b/) | **Paper B** (Kaggle IDC hybrid) |
| [`papers/paper_c/`](../../papers/paper_c/) | **Paper C** (classical FOV / grouping) |

## Suggested Code Availability sentence (Paper D)

> Code for the BreakHis transfer experiments is available at
> https://github.com/jpmkumar/mbc (directory `papers/paper_d/`, tag `<TAG>`).
> The preregistered protocol, case-ID-grouped split manifests, and circuit
> registry are included. BreakHis images are obtained from the public archive;
> they are not redistributed in the repository.
