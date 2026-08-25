# MBC publication workspaces

This repository hosts **three journal papers** from the same codebase. Each paper
has a **dedicated directory** for citation in Data/Code Availability statements.
You do **not** need separate GitHub repositories or branches.

Papers B and C share the Kaggle IDC cohort and the same frozen patient folds, but
answer different questions: **B** asks whether a hybrid quantum stage helps (null
result), **C** asks whether the classical evaluation is valid at all and whether
pathology foundation models transfer to 50×50 patches. Paper C is classical-only
by decision — see [`paper_c/survey/QUANTUM_DECISION.md`](paper_c/survey/QUANTUM_DECISION.md).

**Repository:** https://github.com/jpmkumar/mbc

## Citation URLs (copy for manuscripts)

Replace `<TAG>` with a release tag at submission (recommended for reviewers).

| Paper | Dataset | Cite this path | Availability doc |
|-------|---------|----------------|------------------|
| **A** | Wisconsin Breast Cancer (**tabular**) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_a` | [`paper_a/CODE_AVAILABILITY.md`](paper_a/CODE_AVAILABILITY.md) |
| **B** | Kaggle IDC **histopathology** (hybrid QML) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_b` | [`paper_b/CODE_AVAILABILITY.md`](paper_b/CODE_AVAILABILITY.md) |
| **C** | Kaggle IDC **histopathology** (deep learning) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_c` | TBD under `paper_c/` |

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
```

Reviewers then open exactly the code that matches each paper.

## Scope summary

| | Paper A | Paper B | Paper C |
|---|---------|---------|---------|
| **Folder** | [`paper_a/`](paper_a/) | [`paper_b/`](paper_b/) | [`paper_c/`](paper_c/) |
| **Data** | Wisconsin (tabular) | Kaggle IDC patches | Kaggle IDC patches |
| **Question** | Hybrid QML on tabular | Does the quantum stage help? | Is the classical evaluation valid? |
| **Quantum** | yes | yes | **no** (by decision) |
| **Shared code** | `src/`, root `requirements.txt` | same | same |

Manuscript sources are maintained outside this repository; only the code each
paper cites is released here.
