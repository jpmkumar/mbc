# Paper A — pre-submission revision (2026-11-24)

Response to an internal critical review. Backups of everything replaced carry
the suffix `_before_revision_2026-11-24`.

## What changed in the manuscript (`manuscript_frontiers/main.tex`)

| Review point | Change |
|---|---|
| Primary contrast is a strawman; abstract vs conclusion contradict | Abstract, Introduction, Results, Discussion and Conclusion now say the same thing: the confirmatory gap is real but mostly a read-out / decision-rule effect; a trained all-qubit read-out matches a *linear* classifier on PCA-8, not more. New Results subsection "Read-out and decision rule" placed right after the primary contrast. |
| CE/80 called "matched" | Methods and Results state that CE/80 uses fixed `[-z, z]` logits and a zero threshold, so it equalises loss and epochs only. |
| "Pre-specified" / confirmatory vs exploratory | Study design now states PROTOCOL.md fixed the confirmatory arms before the run, was not externally registered, and every follow-up is exploratory. |
| Anti-conservative statistics | Nadeau–Bengio corrected resampled t-test + 95% CI for every contrast (`scripts/analysis_revision_stats.py` → `results/revision_stats.json`). Table 5 shows corrected and naive p. Repeated-CV hybrid p changes 0.007 → 0.096. IQP/Z vs linear-PCA4 "trailed" claims softened (CIs include 0). |
| Seed variance | Run-to-run variation (0.840 vs 0.865/0.871 for the same recipe) reported; differences < 0.03 not interpreted. Multi-seed scripts added (below). |
| "Capacity-matched" | Replaced with "parameter-matched", defined, and its limits stated. Title changed accordingly. Param column added to Table 4. |
| Untuned optimisation / kernels | No-tuning statement; kernel bandwidth caveat (Shaydulin & Wild 2022); test-selected LR grid reported as an optimistic bound (VQC 0.851 vs MLP 0.967); convergence caveat on gradient norms. |
| QFEAT-SVM uninformative | Shown to be insensitive by construction (extras = 1.3% of RBF distance variance); extras alone 0.798; standardised concat 0.972. Expectations shown to be classically computable. |
| KTA overclaimed | Recomputed with exact statevectors; **bug fixed**: diagnosis-script IQP lacked data-dependent ZZ phases and duplicated the angle kernel. Pooled ρ = 0.73, within-kernel ρ = 0–0.2; text now says KTA separates map families only. |
| SD inconsistencies | All SDs are sample SD (ddof = 1). Boost, morphology, diagnosis numbers converted. Scripts that used ddof = 0 fixed. |
| Hardware inconsistency | Implementation names both machines; locked recipe reproduces exactly on both. |
| Word count, p ≥ 0.11 claim, tanh error, "six-qubit" E2E error | Fixed (5,424 words; classical extras are sines + products, not tanh; E2E uses eight qubits). |
| Missing tables / blank paths in DOCX | New `scripts/build_frontiers_docx.py` (pandoc 3 via pypandoc) keeps all 5 tables, PNG figures, code paths and resolved numbering. |
| Missing literature; unsupported claims about prior work | Added Bowles et al. 2024 (arXiv), Schuld & Killoran 2022, Nadeau & Bengio 2003, Shaydulin & Wild 2022. Claims about Jose/Yadav/Vashisth softened to "as we read these reports" and "we did not re-run their pipelines". |
| Unreported arm | HYBRID MLP→QSVM-ANGLE (0.919) added to Table 4; investigation runs (entanglement, sample efficiency, random labels, LR grid) now reported. |
| AI statement | No longer claims AI checked "every" number; states AI checks are not independent verification. |

## Code changes

- `train_wbcd_cv.py`, `train_wbcd_vqc_boost.py`: new `--init-seed` (folds stay fixed by `--seed`; default reproduces the deposited numbers). Boost summary now uses ddof = 1 and always writes `rows.csv`.
- `train_wbcd_quantum_diagnosis.py`: IQP encoder now matches the sweep; ddof = 1.
- `train_wbcd_gpu_investigation.py`, `train_wbcd_morph_kta.py`: ddof = 1.
- New: `analysis_revision_stats.py`, `run_multiseed.sh`, `analysis_multiseed.py`, `build_frontiers_docx.py`.

## Still to do (authors)

1. **Run the multi-seed battery** on the campus workstation (from the `mbc` root):
   ```bash
   source .venv/bin/activate
   QUICK=1 bash papers/paper_a/scripts/run_multiseed.sh          # smoke test first
   bash papers/paper_a/scripts/run_multiseed.sh                  # 5 seeds: 42 101 202 303 404
   ```
   Output: `papers/paper_a/results/multiseed/multiseed_summary.{json,md}`.
   Seed 42 must reproduce the deposited confirmatory and boost numbers.
   Send the summary back and the manuscript will be updated (seed SDs, paired
   tests and TOST equivalence for the read-out arms, margin ±0.02).
2. Verify the four new references against Crossref (Bowles 2024 is an arXiv
   preprint; it breaks the journal-only rule in AGENTS.md — keep or drop).
3. Confirm the claims about what Jose 2024 / Yadav 2026 / Vashisth 2021
   compared against (ideally a supplementary table).
4. Commit, tag `paper-a-v1.2`, and upload a new Zenodo version; update the DOI
   in the manuscript if Zenodo mints a version DOI.
