# Stage-B secondary analysis v3: nested per-fold learning-rate selection

Declared **before** any v3 number was computed, and recorded in git so the
declaration timestamp precedes the first v3 run. This document adds a secondary
analysis. It does not amend protocol v2, which remains primary.

## Disclosure of the trigger, stated first

This analysis was added **after** seeing an unexpected v2 result on Fold 3, and
that ordering is the single most important thing a reader needs to know about
it. Folds 0 through 3 had already been scored on their held-out splits under
protocol v2 when the problem below was identified. Protocol v2 explicitly warns
against outcome-triggered protocol changes. This addition is outcome-triggered.
That is precisely why it is declared as a **secondary** analysis whose result
cannot replace the pre-registered one, rather than as an amendment that would
quietly re-define the primary comparison.

## The problem it addresses

Protocol v2 locks each head's learning rate to the value selected on Fold 0:
MLP at 1e-3, VQC at 1e-2. On Fold 3 that lock produced a mean gap of +0.01553
test AUPRC in the VQC's favour, with a ten-seed median of +0.01365 that did not
collapse toward zero as it had on Folds 1 and 2.

The two heads nevertheless share a ceiling on Fold 3. The MLP's best seed
reaches 0.88927 test AUPRC against the VQC's 0.89008, a difference of 0.00081
that is an order of magnitude inside the practical margin. What differs is how
often each head reaches its own ceiling: the MLP lands within 0.005 of it on 3
of 10 seeds, the VQC on 9 of 10.

A validation-only diagnostic identifies the cause as the locked rate rather
than the head. On Fold 3 the MLP at 1e-2 reaches 0.948939, 0.948939 and
0.948938 validation AUPRC across seeds 42 to 44 — stable to six decimal places
and slightly above the VQC's own locked-1e-2 values of 0.948824, 0.946960 and
0.945866. The MLP at the locked 1e-3 instead ranges from 0.92628 to 0.94719.
The pre-registered rate, chosen on one fold, therefore fails to transfer to
Fold 3, and the primary comparison on that fold comes down to a badly tuned
MLP against a well tuned VQC.

The direction of this bias is recorded here so it cannot be re-interpreted
later: it inflates the VQC, so it pushes the cross-fold result *away* from the
equivalence conclusion this project expects. The four-fold provisional decision
moved from practical equivalence at three folds, TOST p = 0.011, to
inconclusive, TOST p = 0.179, on Fold 3 alone.

## The selection rule, fixed now

For each fold and each head **independently**, the learning rate is chosen from
{1e-3, 1e-2} as the one with the higher mean best validation AUPRC across the
ten declared seeds 42 to 51. Ties beyond 1e-6 resolve to the smaller rate. The
choice is computed by `scripts/select_vqc_stage_b_nested.py` from the
validation columns of the confirmatory run directories, so it is mechanical,
deterministic and reproducible from the committed artifacts.

No test data of any kind enters the selection. The rule is applied to all five
folds, including the folds whose v2 result was unremarkable and including
Fold 0, rather than only to Fold 3. Applying it selectively would reproduce the
defect this document exists to disclose.

## What does not change

Raw features. Two thousand optimizer steps, batch 64, 4096 train and 1024
validation patches per class. Eight qubits, two layers, angle-Y encoding,
linear entanglement. Thresholds from each checkpoint's validation subset. Ten
seeds, 42 through 51. The practical margin of 0.01 AUPRC. The fold as the unit
of analysis, the Nadeau-Bengio corrected interval, and the TOST decision rules
of protocol v2 section "Decision rules, fixed in advance", applied verbatim.

## What this costs, stated plainly

Each fold's held-out split is scored a **second** time, once under v2 and once
under v3. That is a real reduction in the strength of the held-out guarantee
and it is disclosed in the manuscript rather than absorbed. Two mitigations are
fixed now: the v3 selection cannot see test data, and both results are reported
in full, in the same table, with this document cited as the reason the second
evaluation exists. No third evaluation of any fold is permitted under any
protocol version.

Training the two additional cells, MLP at 1e-2 and VQC at 1e-3, at seeds 42 to
51 is required so that both rates exist for both heads on every fold. Seeds 42
to 44 already exist for both cells on every fold and are reused.

## Reporting rules if v2 and v3 disagree

- Protocol v2 governs the pre-registered claim and is reported as primary.
- If v3 yields practical equivalence while v2 is inconclusive, the reported
  conclusion is that equivalence holds when each head is tuned fairly within
  its own fold, and that the pre-registered locked-rate comparison is
  inconclusive because the fold-0 rate does not transfer. Both numbers appear
  in the abstract. The inconclusive primary result is not omitted.
- If v3 also yields inconclusive, the study reports inconclusive.
- If v3 yields a difference in either direction, that is reported as a
  difference and the equivalence framing is abandoned.

## Additional secondary outcome, declared here

**Convergence reliability.** For each head, fold and learning rate, the
fraction of the ten seeds whose test AUPRC falls within 0.005 of that head's
own best seed on that fold, plus each head's seed spread. The Fold 3
observation motivated reporting it, which is why it is secondary. It is
computed for every fold and both rates.

**Learning-rate sensitivity.** For each head and fold, the difference in mean
validation AUPRC between 1e-3 and 1e-2. This quantifies the transfer failure
using validation data only and is reported for all folds regardless of outcome.

## Known limitations, acknowledged in advance

The candidate set holds two learning rates, so the nested selection is coarse;
it corrects a transfer failure between two known-reasonable rates rather than
searching for either head's optimum. The step budget is held fixed, so a head
that would converge at 1e-3 given more steps is still recorded as failing to
converge at 1e-3 within this budget. Five folds give four degrees of freedom in
both analyses, so both corrected intervals stay wide.
