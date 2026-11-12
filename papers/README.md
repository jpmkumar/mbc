# MBC publication workspaces

This monorepo hosts the shared research code for the MBC paper series. Each
paper also has a **private standalone GitHub repository** for citation in
Data/Code Availability statements (manuscript drafts and AI tooling notes are
excluded from those releases).

| Paper | Standalone repo (private) | Monorepo path |
|-------|---------------------------|---------------|
| **A** | https://github.com/jpmkumar/mbc-paper-a | [`paper_a/`](paper_a/) |
| **B** | https://github.com/jpmkumar/mbc-paper-b | [`paper_b/`](paper_b/) |
| **C** | https://github.com/jpmkumar/mbc-paper-c | [`paper_c/`](paper_c/) |
| **C+** | https://github.com/jpmkumar/mbc-paper-c-plus | [`paper_c_plus/`](paper_c_plus/) |
| **D** | https://github.com/jpmkumar/mbc-paper-d | [`paper_d/`](paper_d/) |

**Monorepo:** https://github.com/jpmkumar/mbc

Papers B and C share the Kaggle IDC cohort and the same frozen patient folds, but
answer different questions: **B** asks whether a hybrid quantum stage helps;
**C** asks whether random-patch CV overestimates case-grouped performance and
whether FOV helps under frozen DINOv2/ResNet. Paper C is classical-only.

**Paper C+** replicates the protocol-optimism contrast on BreakHis with a
confirmatory frozen-encoder F-MLP (FOV not co-primary on BreakHis tiles).

**Paper D (BreakHis)** uses the same circuit family as B under case-ID CV.
Preregistration:
[`../preregistration/paper_d_breakhis_protocol.md`](../preregistration/paper_d_breakhis_protocol.md).

## Citation guidance

Prefer the standalone paper repo URL for manuscript availability statements.
For shared libraries (`src/`, Docker images, multi-paper runners), cite a
**tagged commit** of this monorepo.

```bash
git tag -a paper-b-v1.0 -m "Paper B histopath reproducibility release"
git push origin paper-b-v1.0
```

## Scope summary

| | Paper A | Paper B | Paper C | Paper C+ | Paper D |
|---|---------|---------|---------|----------|---------|
| **Folder** | [`paper_a/`](paper_a/) | [`paper_b/`](paper_b/) | [`paper_c/`](paper_c/) | [`paper_c_plus/`](paper_c_plus/) | [`paper_d/`](paper_d/) |
| **Data** | Wisconsin (tabular) | Kaggle IDC | IDC + BCSS | BreakHis | BreakHis |
| **Question** | Hybrid QML on tabular | Does the quantum stage help? | Protocol optimism / FOV | Does optimism replicate? | Does B's VQC transfer? |
| **Quantum** | yes | yes | **no** | **no** | yes |

Manuscript sources, venue drafts, and AI-assistant notes stay local and are not
pushed to GitHub.
