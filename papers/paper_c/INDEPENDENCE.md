# Paper C — independence from Paper B

**Concern:** Paper B is not yet submitted and acceptance is not guaranteed. Can
Paper C rely on it?

**Answer: no, and it must not need to.** Paper C is designed to stand alone. This
document records how that is enforced and what to do in each acceptance scenario.

---

## 1. The rule

**Paper C's manuscript contains no load-bearing citation to Paper B.**

A citation is *load-bearing* if removing it forces a claim to be weakened,
qualified, or deleted. Paper C may mention Paper B as a courtesy once Paper B is
accepted; it may never depend on it.

Two structural facts make this achievable:

**Quantum never appears in Paper C's text.** The rationale for the exclusion is
a project decision record, not manuscript content. No reviewer asks a classical computational-
pathology paper to justify omitting quantum computing. A one-line scope statement
covers it, and that line cites nothing.

**The shared artifacts are data, not findings.** Paper C reuses the frozen case-ID
folds in `data/splits/histopath_kaggle/folds/fold_*/`. Those are a repository
artifact citable as the code/data release, not a result of Paper B. Reusing a
split file creates no citation dependency.

## 2. Why the dependency direction matters

The natural intellectual dependency runs **B → C**, not C → B:

- Paper B makes an *equivalence* claim, and an equivalence claim is only as strong
  as the classical reference it is measured against. A leakage-free, calibrated,
  patient-cluster-interval classical baseline is exactly what strengthens it.
- Paper C makes a *methodological* claim about evaluation validity and
  representation transfer. Nothing in it requires knowing whether a variational
  circuit helps.

This is the safe direction. **C never waits on B.** If B is delayed, rejected, or
abandoned, C is unaffected.

The reverse would have been dangerous: had Paper C been framed as "we exclude
quantum because our companion paper showed it does not help," an unpublished
companion would have left a hole in C's argument. That framing is now explicitly
removed from the decision record.

## 3. Overlap management

B and C share a dataset, a cohort, and fold assignments. That is legitimate and
common, but it must be handled deliberately.

| Dimension | Paper B | Paper C | Overlap risk |
|---|---|---|---|
| Research question | Does a hybrid quantum stage help? | Is the classical evaluation valid, and do foundation models transfer? | none |
| Method | VQC / hybrid schedule, Stage-A/B | Foundation-model probes, calibration, conformal | none |
| Primary endpoint | Paired AUPRC gap, TOST equivalence | Protocol/context AUPRC contrasts, BCSS replication | low |
| Split protocol | outer case-ID folds, two-way | same outer folds, **three-way** inner splits plus random-patch arm | numbers differ |
| Dataset & folds | IDC shared | IDC shared; BCSS unique to C | **disclose** |

Three safeguards:

1. **Do not re-report Paper B's numbers as new results in Paper C.** Paper C's
   three-way inner-split protocol produces genuinely different figures from Paper
   B's two-way protocol, so this is mostly automatic — but any classical baseline
   that does coincide must be recomputed under Paper C's protocol rather than
   copied.
2. **Disclose the related manuscript** in Paper C's cover letter and in any
   "related submissions" field. Most publishers ask directly whether related work
   is under consideration elsewhere; answering plainly costs nothing and an
   undisclosed overlap discovered later is serious.
3. **Write an overlap statement** for the submission package naming the shared
   cohort and folds and stating the distinct research questions.

## 4. Contingency by scenario

| Scenario at Paper C submission | Action |
|---|---|
| Paper B accepted, DOI available | Add one courtesy sentence and citation. Verify the DOI through `verify_crossref.py` like any other entry. |
| Paper B under review | Cite nothing. Disclose in the cover letter as a related manuscript under consideration. |
| Paper B rejected or withdrawn | Cite nothing. No change to Paper C — nothing in it depended on B. |
| Paper B repeatedly unplaceable | Consider repositioning B for a methods or negative-results venue, using Paper C's published baseline as its classical reference. Do **not** merge B into C (§5). |

Note that the journal-only reference policy already forbids citing Paper B as a
preprint, so "cite the arXiv version" is not an available shortcut. That
constraint pushes in the same direction as this document.

## 5. Why not merge B into C

If Paper B struggles, folding it into Paper C is the tempting move. It is the
wrong one:

- **Overstuffed scope.** Paper C carries a focused protocol/context benchmark
  with external replication. A quantum arm on top
  reads as bolted on, and reviewers cut the weakest limb — which would be the
  quantum section.
- **The equivalence argument needs room.** Paper B's TOST framing, preregistered
  margins, ten-seed extensions, and the disclosed v2/v3 protocol ordering are a
  careful statistical narrative. Compressed into a subsection it would lose the
  precision that makes it defensible.
- **It contradicts Paper C's own thesis.** Paper C isolates representation
  quality under a common frozen-probe protocol. Adding a different trainable
  head family would destroy that controlled comparison.
- **It forfeits a publication.** Two well-scoped papers beat one unfocused one.

If B genuinely cannot be placed, the better move is repositioning it, not
dissolving it into C.

## 6. Pre-submission check

Before Paper C is submitted, audit the manuscript:

```bash
# No citation to Paper B should remain unless it is accepted and courtesy-only.
rg -in "paper b|companion|quantum|variational|qubit|VQC" papers/paper_c/manuscript/
```

Every hit must be either absent, or a deliberate courtesy citation with a verified
DOI. Any sentence that would change if Paper B vanished is a bug.
