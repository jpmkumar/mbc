# MBC publication workspaces

Two-paper plan (decided 2026-07):

| Workspace | Data | Claim | Manuscript |
|-----------|------|-------|------------|
| [`paper_a/`](paper_a/) | **Wisconsin Breast Cancer (WBCD)** — tabular clinical features | Hybrid classical–quantum pipeline on **native low-D features** | To be drafted |
| [`paper_b/`](paper_b/) | **Kaggle IDC histopathology** — image patches, patient-level CV | Bolt-on VQC **does not beat** matched classical controls (null / equivalence) | [`paper/main.tex`](../paper/main.tex) |

Shared training code stays at repo root (`src/`, `scripts/`, `configs/`).
Raw checkpoints live in gitignored `results/`. Each workspace keeps curated
tables, figure exports, and local notes.

**Current active experiments** (width ablation, Stage-B, server runs) belong to **Paper B**.
