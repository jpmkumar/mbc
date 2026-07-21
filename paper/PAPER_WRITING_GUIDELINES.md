# Paper writing guidelines — histopath hybrid QML manuscript

Apply to all DOCX/LaTeX drafting. Target: **Q1/Q2 journal standard** (see §14).

---

## 1. Human-like academic writing

Write in the style of an experienced researcher, not AI-generated text.

Avoid generic expressions such as:

- "Nowadays..."
- "It is worth mentioning..."
- "In today's world..."
- "This paper aims to..."

Use natural academic transitions and logical reasoning.

---

## 2. No repetitive content

Never repeat the same explanation across sections.

If a concept is introduced once, later sections build upon it instead of restating it.

Every paragraph must contribute new information.

---

## 3. Minimize similarity (plagiarism)

Every section will be written from scratch.

- Avoid copying wording from published papers.
- Use original sentence structures.
- Present ideas through synthesis rather than paraphrasing.
- Integrate multiple sources into a cohesive discussion instead of describing papers one by one.

Proper citations remain mandatory; originality is in expression and synthesis.

---

## 4. Smooth story flow

The paper will read as a continuous narrative, not disconnected sections.

**Flow:**

Problem → Current research → Existing limitations → Research gap → Motivation → Proposed framework → Analysis → Discussion → Future directions → Conclusion

Each paragraph leads naturally into the next.

---

## 5. Strong paragraph structure

Every paragraph follows:

1. Topic sentence  
2. Supporting evidence  
3. Critical analysis  
4. Connection to the next idea  

---

## 6. Proper figure references

Every figure is introduced **before** it appears, then discussed in detail.

Example (adapt to this study):

> Figure 1 depicts the shared EfficientNet–transformer backbone and the three classification heads (linear, matched MLP, and VQC). The diagram highlights that only the final head differs across experimental arms, isolating the effect of the quantum module.

---

## 7. Proper table references

Every table is cited and interpreted in the text—not merely pointed to.

Example:

> Table I summarises five-fold test performance under patient-level cross-validation. The matched classical MLP (E2b) attains the highest mean F1, whereas the VQC head (E3) shows greater fold-wise variance without a statistically significant advantage (Friedman p > 0.25).

---

## 8. Proper chart references

Charts are introduced and interpreted, with trends linked to the argument.

Example:

> Figure 5 compares mean balanced accuracy and AUPRC across arms. The overlapping confidence intervals corroborate the Friedman test: performance differences are modest relative to fold-to-fold variability.

---

## 9. No floating figures or tables

Each figure/table requires:

- Introduction before placement  
- Analysis immediately after  
- Clear linkage to surrounding discussion  

---

## 10. Professional academic tone

Writing will be:

- Formal  
- Objective  
- Evidence-based  
- Analytical  
- Critical rather than merely descriptive  

---

## 11. Consistent formatting

Throughout the manuscript:

- Headings: black, consistent hierarchy  
- Body text: black  
- Figure/table captions: consistent style  
- Cross-references: consistent numbering (Fig. / Table / Eq.)  
- Citations: uniform style per target journal (IEEE numbered until venue fixed)  

---

## 12. High-level critical analysis

Literature sections compare methodologies, note strengths and weaknesses, identify unresolved challenges, and explain how gaps motivate **this** study—not a generic EQML label.

Prior hybrid QML breast work is evaluated on: patient-level CV, matched classical capacity, imbalance-aware metrics, and reproducibility.

---

## 13. Journal-quality standards

Each section satisfies expectations of strong peer-reviewed work:

- Logical flow  
- Concise yet comprehensive explanations  
- Original academic language  
- Clear motivation for every section  
- Appropriate transitions  
- Well-integrated figures, tables, and citations  
- Minimal redundancy and low textual similarity  

---

## 14. Q1 / Q2 journal targeting

The manuscript is written to meet **first- or second-quartile (Q1/Q2)** expectations, not conference-extended or low-tier filler.

### What Q1/Q2 reviewers expect

| Dimension | Requirement for this paper |
|-----------|----------------------------|
| **Contribution clarity** | Lead with **rigorous evaluation + matched control + null result** as a substantive finding—not as a failed experiment |
| **Novelty** | Methodological: E2b control, patient StratifiedGroupKFold, Friedman + AUPRC; not “new quantum SOTA” |
| **Rigor** | Full protocol transparency; 5-fold aggregates; honest limitations; optional ablations (entanglement, qubits) in supplementary |
| **Related work** | Critical synthesis of classical histopath DL **and** hybrid QML breast papers; gap table comparing CV / matched control / metrics |
| **Significance** | Explain why null results matter for the field (reproducibility crisis, overclaiming in QML medical imaging) |
| **Presentation** | Publication-grade figures (≥300 DPI); no placeholder captions in submission draft |
| **Reproducibility** | Code/data availability statement; hyperparameters in supplement |
| **Claims discipline** | No clinical deployment, quantum advantage, or SOTA language unsupported by Table I |

### Venue alignment (Q1/Q2 realistic targets)

| Tier | Examples | Fit for this manuscript |
|------|----------|-------------------------|
| **Primary** | IEEE Access (often Q1/Q2 by field), Computers in Biology and Medicine, Quantum Machine Intelligence | Strong fit |
| **Stretch Q1** | Medical Image Analysis, Pattern Recognition, npj Digital Medicine | Needs exceptional framing + external validation or ablation depth |
| **Avoid mismatch** | Venues expecting quantum speedup or new architecture only | Rejection likely despite solid science |

### Writing bar check (before submission)

- [ ] Abstract states **finding** (null advantage) in sentence 3, not buried  
- [ ] Introduction ends with numbered contributions matched to Results  
- [ ] Methods reproducible from text + supplement alone  
- [ ] Every figure/table referenced, interpreted, and tied to a claim  
- [ ] Discussion addresses **why** image bolt-on VQC differs from tabular hybrid success in literature  
- [ ] Limitations section is explicit (single dataset, simulation, 8-D bottleneck)  
- [ ] Supplementary: per-fold table, Friedman inputs, ablation (entnone/q4/q12) when complete  

---

*Guidelines apply to `paper/manuscript.docx` (local) and any future LaTeX port. See `PAPER_WRITING_PLAN.md` for structure and numbers.*
