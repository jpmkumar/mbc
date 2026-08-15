# MBC publication workspaces

This repository hosts **two journal papers** from the same codebase. Each paper
has a **dedicated directory** for citation in Data/Code Availability statements.
You do **not** need separate GitHub repositories or branches.

**Repository:** https://github.com/jpmkumar/mbc

## Citation URLs (copy for manuscripts)

Replace `<TAG>` with a release tag at submission (recommended for reviewers).

| Paper | Dataset | Cite this path | Availability doc |
|-------|---------|----------------|------------------|
| **A** | Wisconsin Breast Cancer (**tabular**) | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_a` | [`paper_a/CODE_AVAILABILITY.md`](paper_a/CODE_AVAILABILITY.md) |
| **B** | Kaggle IDC **histopathology** | `https://github.com/jpmkumar/mbc/tree/<TAG>/papers/paper_b` and `.../paper` | [`paper_b/CODE_AVAILABILITY.md`](paper_b/CODE_AVAILABILITY.md) |

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

| | Paper A | Paper B |
|---|---------|---------|
| **Folder** | [`paper_a/`](paper_a/) | [`paper_b/`](paper_b/) |
| **Manuscript** | TBD under `paper_a/` | [`paper/main.tex`](../paper/main.tex) |
| **Shared code** | `src/`, root `requirements.txt` | same |

See [`HOW_TO_USE_AGENTS.md`](HOW_TO_USE_AGENTS.md) for separate Cursor agents.
