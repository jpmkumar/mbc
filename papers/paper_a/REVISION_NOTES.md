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
| Missing literature; unsupported claims about prior work | Added Schuld & Killoran 2022, Nadeau & Bengio 2003, Shaydulin & Wild 2022 (Bowles et al. 2024 considered but dropped: arXiv only). Claims about Jose/Yadav/Vashisth softened to "as we read these reports" and "we did not re-run their pipelines". |
| Unreported arm | HYBRID MLP→QSVM-ANGLE (0.919) added to Table 4; investigation runs (entanglement, sample efficiency, random labels, LR grid) now reported. |
| AI statement | No longer claims AI checked "every" number; states AI checks are not independent verification. |

## Code changes

- `train_wbcd_cv.py`, `train_wbcd_vqc_boost.py`: new `--init-seed` (folds stay fixed by `--seed`; default reproduces the deposited numbers). Boost summary now uses ddof = 1 and always writes `rows.csv`.
- `train_wbcd_quantum_diagnosis.py`: IQP encoder now matches the sweep; ddof = 1.
- `train_wbcd_gpu_investigation.py`, `train_wbcd_morph_kta.py`: ddof = 1.
- New: `analysis_revision_stats.py`, `run_multiseed.sh`, `analysis_multiseed.py`, `build_frontiers_docx.py`.

## Multi-seed battery (done 2026-11-24)

Campus run finished (ntfy 13:35 IST). Seed 42 reproduced confirmatory VQC-8
0.840 / MLP-PCA8 0.957 / SVM-RBF 0.975 and boost L4 0.971. Seed-averaged
primary Δ = −0.113 (95% CI −0.143 to −0.084; NB p = 4.6e-4). All-Z L2 ≡
MLP-PCA8 within ±0.02 (TOST); L4 vs linear PCA-8 Δ = +0.005 (p = 0.71).
Summaries in `results/multiseed/`; manuscript Frontiers draft updated.

## Multi-seed integration check (2026-11-24 evening)

- All multi-seed numbers in main.tex match `results/multiseed/multiseed_summary.json`;
  seed 42 reproduces the deposited confirmatory and boost fold scores exactly.
- Added: multi-seed hybrid result (gap to MLP-H6 shrinks −0.065 → −0.028,
  p = 0.10; seed 42 was MLP-H6's best seed; vs MLP-30D −0.033, p = 0.014),
  five-seed CE/80 result, multi-seed rows in Table 5, seed-42 note in the
  Table 4 caption, word counts (5,563 / 316). Rebuilt PDF, DOCX and LaTeX zip.

## Still to do (authors)

0. DONE (18:31): all raw multi-seed outputs copied; recomputing the summary from them reproduces multiseed_summary.json exactly (max diff 5e-15).
1. DONE: references checked on Crossref. SchuldKilloran2022 (PRX Quantum 3(3)
   030101), Nadeau2003 (Mach. Learn. 52(3) 239-281) and Shaydulin2022 (Phys. Rev.
   A 106(4) 042407) are journal-article records matching the bib. Bowles 2024
   has no journal version (arXiv v2 only), so it was removed from the text and
   bibliography per the journal-only rule.
2. DONE: prior-work claims checked against the published articles (Vashisth:
   De Gruyter full text; Yadav: PeerJ full text; Jose: author copy on
   inspirehep.net, Wiley page blocked by a bot check). Corrections made:
   - Vashisth's classical polynomial SVM on the same 2 PCs beat the QSVM
     (0.95 vs 0.92); their claimed advantage is theoretical run time.
   - Yadav's own SVM/logistic regression under the same preprocessing beat the
     hybrid (97.4% vs 95.6%); their reported recall 0.972 is the benign recall
     implied by their confusion matrix [[38,4],[2,70]] (malignant recall 0.905).
   - Jose reports RBF/linear/poly/sigmoid SVMs but no stated input; 100% QSVM
     accuracy on 10 random 300/50 splits; "remarkably accurate" quote confirmed.
   - Claim "none uses an RBF SVM on all 30 features" softened to "none states".
   - New Table 1 summarises the three studies; old Tables 1–5 are now 2–6.
   Jose: the inspirehep.net file is the Wiley-typeset article (IET Quant.
   Comm. 5:384-398, CC BY, Wiley download stamp), so the blocked Wiley page is
   not needed. Its Table 3 reports classical SVMs at 2-10 inputs indexed by
   qubit count (RBF 0.85-0.95 vs quantum kernel 0.87-0.97); Table 1 updated.
   Other prior work, checked against full texts:
   - Xiang 2024 (nature.com): claims correct; added single 70/30 split and
     accuracy-only evaluation.
   - Azevedo 2022 (Springer): correct; added BCDR mammograms, frozen ResNet18.
   - Zorlu 2026 (MDPI): correct; added external WBC validation, VQC close to
     RBF-SVM (AUC 0.992 vs 0.998), QSVM with fixed hyperparameters.
   - Daga 2026 (nature.com): "leakage-free" applies to the out-of-fold
     stacking only; PCA was fitted globally (they report a 0.0018 accuracy
     effect). Ensemble matched, did not beat, RBF-SVM on WDBC; standalone VQC
     failed. Text corrected accordingly.
3. Commit remaining scripts if needed, tag `paper-a-v1.2`, and upload a new
   Zenodo version; update the DOI in the manuscript if Zenodo mints a version
   DOI.

## Language and figure pass (2026-11-24, 18:50)

Full read of abstract, body, captions and all three figures. Edits made for
accuracy and clarity (not to target any AI detector; the Generative AI
statement is unchanged and remains the disclosure that matters):
- One acronym throughout: WBCD → WDBC (matches UCI and the cited literature);
  Figure 1 box relabelled; running head updated.
- Figure 2 y-axis "mean ± std" → "mean ± SD" to match the text.
- Factual fix: morphology paragraph wrongly said "ten sklearn features";
  the 30 features are 10 attributes × (mean, SE, worst).
- Removed an unsupported claim ("ceiling ... bounded by the classical models")
  from Limitations; fixed a dangling "that uncertainty".
- Clarified the hybrid fold count (MLP-H6 higher in 5/5 folds, MLP-30D 4/5),
  the seed-vs-script variation sentence, and trimmed repeated "therefore".
- Shortened the abstract's read-out sentence; removed a duplicated class
  count from Study design.
Figures regenerated with scripts/make_figures.py (backups in
figures/before_voice_2026-11-24/ and frontiers_upload/before_voice_2026-11-24/).

## Reviewer-style fixes (2026-11-24, 19:10)

Backup: `manuscript_frontiers/main_before_reviewfix_2026-11-24.tex`.

1. Entangler ablation: the "no entangler" arm has no trainable layer (output = cos of PC1 at <Z0>). Methods 2.11 now describes it correctly. Results 3.5 adds its ROC-AUC (0.971 ± 0.012, from `results/gpu_investigation/summary.json`, same as trained VQC-8 0.972) and states that the ablation does not isolate entanglement, because only qubit 0 is read. Discussion no longer claims that "entanglement helped the trained circuit".
2. Generative AI statement: "Claude Opus 5.5" changed to "Claude (Anthropic; Opus 5.5 and Sonnet 5.5 models)". AUTHOR TO CONFIRM the full list of models used in all sessions.
3. Limitations p-range: now 0.10–0.24 vs MLP-H6 (confirmatory, repeated, CE/80) and 0.34 vs MLP-30D, matching Table 6 and `revision_stats.json`.
4. Table citation order: Section 2.1 cites "Tables 2–4"; Section 2.14 points to Section 3.4 instead of the sweep table; the sensitivity table now floats before the exploratory table (sens = Table 5, sweep = Table 6), so first citations run 1, 2–4, 5, 6.
5. Figure 2 caption: notes that the y-axis starts at 0.55.
6. `build_frontiers_docx.py`: resolves `Section~\ref{sec:...}` from main.aux.
7. Main-text word count updated to 5,800 (+94). Rebuilt PDF, DOCX, LaTeX zip; 0 undefined refs.
