# Paper writing plan — Histopath hybrid QML (IEEE Access)

**Writing standards:** See [`PAPER_WRITING_GUIDELINES.md`](PAPER_WRITING_GUIDELINES.md) (14 points, **Q1/Q2 targeting**).

**Status:** E2 / E2b / E3 new-bundle 5-fold complete (null quantum advantage vs matched MLP).  
**E4:** optional (running / pending).  
**Primary venue:** IEEE Access (`paper/VENUE.md`). Backup: Quantum Machine Intelligence.  
**Working title (proposed):**  
*Rigorous Evaluation of Hybrid Classical–Quantum Heads for IDC Histopathology Classification: Patient-Level Cross-Validation and Parameter-Matched Controls*

> **Important:** Current `paper/main.tex` still frames multimodal mammo/ultrasound/thermo. Empirically strongest story **now** is **histopathology IDC** with E2/E2b/E3. Plan below retargets the manuscript around that evidence (keep multimodal as future work / framework note).

---

## 1. Narrative (what the paper claims)

### Defensible contributions
1. **Leakage-safe patient-level 5-fold CV** on large-scale IDC patches (~279 patients, ~277k patches).
2. **Three-way head comparison:** linear (E2) vs **parameter-matched classical MLP (E2b)** vs VQC (E3).
3. **Honest metrics** under imbalance: AUPRC, F-β thresholding, recall/precision, not accuracy-only.
4. **Statistical test:** Friedman across folds — **no significant difference** (F1 p≈0.27).
5. **Finding:** under staged training + this VQC design, quantum head does **not** beat matched classical control; all E3 folds selected `best_stage=stage_a`.

### What NOT to claim
- “Novel quantum architecture that outperforms SOTA”
- Clinical deployment readiness
- Quantum advantage (results do not support it)

### Framing for reviewers
> Prior hybrid QML breast papers often report accuracy gains without matched-capacity classical controls or patient-safe CV. We close that gap and report a **null result** with rigorous controls — scientifically valuable and harder to reject as overclaiming.

---

## 2. Manuscript structure (IEEE Access, ~10–12 pages)

| § | Section | Content | Est. pages |
|---|---------|---------|------------|
| 0 | Title / Abstract / Keywords | Histopath + hybrid QML + matched controls + null finding | 0.5 |
| I | Introduction | IDC screening need; QML hype vs rigor; contributions | 1 |
| II | Related Work | Classical histopath DL; hybrid QML breast; evaluation pitfalls | 1.5 |
| III | Methods | Architecture (E2/E2b/E3[/E4]); dataset & splits; training bundle; metrics; stats | 3 |
| IV | Experiments | Setup; main 5-fold table; per-fold plots; Friedman; efficiency | 2.5 |
| V | Discussion | Why null? bolt-on VQC; stage_a selection; limitations; implications | 1.5 |
| VI | Conclusion | Summary + future (E4, ablations, external set) | 0.5 |
| — | References | ~45–50 from `proposed_model_core_references.bib` | — |
| App | Supplementary | Hyperparams; full per-fold; optional ablations/E4 | — |

---

## 3. Figures required

| ID | Figure | Type | Source / how to make | Priority |
|----|--------|------|----------------------|----------|
| **Fig.1** | System architecture | Diagram | Mermaid→TikZ/draw.io/PPT: EfficientNet→Transformer→Compression→{E2,E2b,E3[,E4]} | **P0** |
| **Fig.2** | VQC circuit | Diagram | Angle encode → RY/RZ + CNOT × L → Pauli-Z → linear | **P0** |
| **Fig.3** | Training stages A/B/C | Diagram | Freeze/train blocks per stage (+ fusion for E4) | **P0** |
| **Fig.4** | Patient-level CV scheme | Diagram | StratifiedGroupKFold by patient / IDC-ratio bins | **P1** |
| **Fig.5** | Main results bar chart | Chart | Mean±std E2/E2b/E3 for bal_acc, F1, AUPRC | **P0** |
| **Fig.6** | Per-fold metric lines | Chart | F1 (or AUPRC) vs fold for three arms | **P0** |
| **Fig.7** | PR / ROC curves | Chart | One representative fold (e.g. fold 0) overlay E2/E2b/E3 | **P1** |
| **Fig.8** | Confusion matrices | Chart | 3× panels fold-mean or fold 0 | **P2** |
| **Fig.9** | Efficiency | Chart/table-fig | Params, wall-time/fold (E2 ~2–4h vs E3 ~3–5h) | **P1** |
| Fig.10 | (Optional) E4 fusion α | Chart | If E4 completes | P2 |
| Fig.11 | (Optional) Grad-CAM examples | Images | Success + failure cases | P2 |
| Fig.S1 | Ablation | Chart | If entanglement/qubits run | P2 |

**Assets location:** `figures/` (repo already uses `\graphicspath{{../figures/}}`).  
**Export:** PNG/PDF ≥300 DPI for IEEE Access.

### Fig.1 sketch (architecture)

```
Patch 224×224
    → EfficientNet-B0 (1280-d)
    → Modality Transformer (2048-d)
    → Compression 2048→128→32→8
    → Head:
         E2  : Linear(8→2)
         E2b : LayerNorm→Linear→GELU→Linear
         E3  : VQC → Linear
         E4  : α·MLP + (1−α)·VQC   [optional]
    → Softmax + val-tuned F-β threshold (+ TTA)
```

---

## 4. Tables required

| ID | Table | Columns | Priority |
|----|-------|---------|----------|
| **Tab.I** | Dataset summary | Patients, patches, IDC ratio, split policy | **P0** |
| **Tab.II** | Model configs | Params (backbone + head), qubits, layers, entanglement | **P0** |
| **Tab.III** | Main 5-fold results | E2/E2b/E3: bal_acc, F1, AUC, AUPRC, Rec, Prec (mean±std) | **P0** |
| **Tab.IV** | Per-fold F1 (or AUPRC) | Fold 0–4 × arms + Friedman χ², p | **P0** |
| **Tab.V** | Training protocol | Loss, aug, TTA, threshold, stages, early stop, batch | **P1** |
| **Tab.VI** | Related hybrid QML comparison | Paper, data, CV?, matched control?, metrics claimed | **P1** |
| **Tab.VII** | Compute | GPU, time/fold, software (PennyLane, PyTorch) | **P1** |
| Tab.S1 | Full per-fold all metrics | All arms | P1 |
| Tab.S2 | Hyperparameters | Full YAML | P1 |
| Tab.S3 | E4 / ablations | If available | P2 |

### Tab.III numbers (ready to paste)

| Arm | bal_acc | F1 | AUC | AUPRC | Recall | Precision |
|-----|---------|-----|-----|-------|--------|-----------|
| E2 | 0.883±0.006 | 0.803±0.013 | 0.951±0.004 | 0.888±0.018 | 0.903±0.025 | 0.724±0.018 |
| E2b | 0.884±0.007 | 0.806±0.019 | 0.951±0.005 | 0.887±0.022 | 0.902±0.019 | 0.729±0.031 |
| E3 | 0.883±0.009 | 0.800±0.026 | 0.950±0.005 | 0.887±0.013 | 0.908±0.022 | 0.717±0.047 |

Friedman: F1 χ²=2.63, p=0.27; bal_acc χ²=2.80, p=0.25; AUPRC χ²=1.37, p=0.50.

---

## 5. References

| Source | Role |
|--------|------|
| `/Users/muthu/Research/ConferencePaper/proposed_model_core_references.bib` | **Core ~45** — merge into `paper/references.bib` |
| Keep: Mari2020, Schuld2020, PerezSalinas2020, Azevedo2022, Xiang2024, CruzRoa2017, Spanhol2016, Saito2015, Varoquaux2022, Demsar2006, Mahmood2025MFFHistoNet, Voon2022IDCEfficientNet | Must-cite |
| Drop/de-emphasize bulk XAI-only mammo papers from old survey | Dilute focus |

**IEEE cite style** already in `main.tex` via `\cite{}`.

---

## 6. Writing work packages (order)

| # | Task | Owner / tool | Depends on |
|---|------|--------------|------------|
| W1 | Retarget `main.tex` title/abstract/intro to histopath + matched-control story | Writing | — |
| W2 | Draw Fig.1–3 (architecture, VQC, stages) | Diagram tool / TikZ | — |
| W3 | Generate Tab.III–IV + Fig.5–6 from `results/` JSONs | Script | Done data |
| W4 | Methods: dataset, splits, losses, TTA, threshold, stages | Writing | W1 |
| W5 | Related work rewrite using core bib | Writing | merge bib |
| W6 | Results + Discussion (null finding, limitations) | Writing | W3 |
| W7 | PR/ROC Fig.7 from saved probs if available; else fold-level bars only | Script | checkpoints |
| W8 | Ethics, data availability, code availability | Existing drafts | — |
| W9 | Merge E4 if useful; else “future work” | After E4 zip | E4 run |
| W10 | IEEE polish: captions, 300 DPI, checklist | `paper/SUBMISSION_CHECKLIST.md` | W1–W8 |

---

## 7. Scripts / deliverables to add in repo

1. `scripts/generate_histopath_paper_tables.py` — CSV/LaTeX from fold backups  
2. `scripts/plot_histopath_cv_comparison.py` — Fig.5–6  
3. `figures/fig1_architecture.{pdf,png}` etc.  
4. Update `paper/references.bib` from core bib  
5. Update `publication/PUBLICATION_RESULTS.md` with E2/E2b/E3 means  

*(Do not commit AI drafting notes, `HISTOPATH_QML_REPORT.md`, or `Co-authored-by` trailers.)*

---

## 8. Suggested abstract skeleton (~200 words)

Background → IDC patch classification; hybrid QML often lacks matched controls.  
Methods → EfficientNet-B0 + transformer + compression; heads E2/E2b/E3; patient-level 5-fold; focal loss, strong aug, TTA, F-β.  
Results → Means (table); Friedman non-significant; stage_a best on all E3 folds.  
Conclusion → No evidence of VQC advantage over matched MLP under this protocol; evaluation methodology is the contribution.

---

## 9. Parallel while E4 runs

- [ ] Draft abstract + contributions bullets  
- [ ] Fig.1 architecture diagram  
- [ ] Merge core references into `paper/references.bib`  
- [ ] Build Tab.III / Tab.IV LaTeX  
- [ ] Rewrite Related Work (histopath + hybrid QML + evaluation rigor)  

---

## 10. Decision checkpoint after E4 fold 0

| E4 outcome | Paper action |
|------------|--------------|
| Beats E2b significantly | Add as main arm; update claims carefully |
| Ties / loses | Mention briefly or supplementary; keep null-result narrative |
| Incomplete | Leave as future work |

---

*Plan aligned with local log: `results/histopath_experiment_log.md` (gitignored).*
