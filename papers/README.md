# MBC publication workspaces

This repository hosts **four journal papers** from the same codebase. Each paper
has a **dedicated directory** for citation in Data/Code Availability statements.
You do **not** need separate GitHub repositories or branches.

Papers B and C share the Kaggle IDC cohort and the same frozen patient folds, but
answer different questions: **B** asks whether a hybrid quantum stage helps (null
result), **C** asks whether the classical evaluation is valid at all and whether
pathology foundation models transfer to 50×50 patches. Paper C is classical-only
by decision — see [`paper_c/survey/QUANTUM_DECISION.md`](paper_c/survey/QUANTUM_DECISION.md).

**Immediate QML follow-on (locked 2026-09-16):** **Paper D (BreakHis)** —
same circuit family as B, new estimand (magnification/stain shift under case-ID
CV). Preregistration:
[`../preregistration/paper_d_breakhis_protocol.md`](../preregistration/paper_d_breakhis_protocol.md).
"No gain under shift" is an allowed outcome.

**Repository:** https://github.com/jpmkumar/mbc

## Citation URLs (copy for manuscripts)

Replace `<TAG>` with a release tag at submission (recommended for reviewers).

| Paper | Dataset | Cite this path | Availability doc |
|-------|---------|----------------|------------------|
| **A** | Wisconsin Breast Cancer (**tabular**) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_a` | [`paper_a/CODE_AVAILABILITY.md`](paper_a/CODE_AVAILABILITY.md) |
| **B** | Kaggle IDC **histopathology** (hybrid QML) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_b` | [`paper_b/CODE_AVAILABILITY.md`](paper_b/CODE_AVAILABILITY.md) |
| **C** | Kaggle IDC **histopathology** (deep learning) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_c` | TBD under `paper_c/` |
| **D** | BreakHis **histopathology** (hybrid QML transfer) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_d` | TBD under `paper_d/` (protocol locked) |

**Development branch** (moving target — do not cite in final submission):

```text
https://github.com/jpmkumar/mbc/tree/docs/histopath-writing-q1-guidelines/papers/paper_b
```

## Create stable tags before submission

```bash
# After Paper B experiments and manuscript are frozen:
git tag -a paper-b-v1.0 -m "Paper B histopath reproducibility release"
git push origin paper-b-v1.0

# After Paper A confirmatory runs are frozen:
git tag -a paper-a-v1.0 -m "Paper A tabular WBCD reproducibility release"
git push origin paper-a-v1.0

# After Paper D BreakHis transfer code + curated metrics are frozen:
git tag -a paper-d-v1.0 -m "Paper D BreakHis transfer reproducibility release"
git push origin paper-d-v1.0
```

Reviewers then open exactly the code that matches each paper.

## Scope summary

| | Paper A | Paper B | Paper C | Paper D |
|---|---------|---------|---------|---------|
| **Folder** | [`paper_a/`](paper_a/) | [`paper_b/`](paper_b/) | [`paper_c/`](paper_c/) | [`paper_d/`](paper_d/) |
| **Data** | Wisconsin (tabular) | Kaggle IDC patches | Kaggle IDC patches | BreakHis (multi-mag) |
| **Question** | Hybrid QML on tabular | Does the quantum stage help? | Is the classical evaluation valid? | Does B's VQC transfer under shift? |
| **Quantum** | yes | yes | **no** (by decision) | yes (same family as B) |
| **Shared code** | `src/`, root `requirements.txt` | same | same | same |
| **Priority** | — | submit first | parallel classical track | **immediate QML follow-on** |

Manuscript sources are maintained outside this repository; only the code each
paper cites is released here.
