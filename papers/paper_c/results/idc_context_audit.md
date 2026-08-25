# IDC context-geometry audit

**Run:** 21 August 2026, before foundation-model extraction.  
**Script:** `papers/paper_c/scripts/audit_idc_context.py`.

## Result

- 277,524 patches under 279 public case identifiers were parsed.
- No duplicate `(case_id, x, y)` coordinates were found.
- The modal positive coordinate increment is 50 pixels in both axes, supporting
  the intended regular 50-pixel neighbourhood lookup.
- Complete-neighbourhood centre counts:
  - `K=1`: 277,524
  - `K=3`: 162,018
  - `K=5`: 108,196
  - `K=9`: 49,798
- The 49,798 complete `K=9` centres comprise 21,706 non-IDC and 28,092
  IDC-positive centres from 216 case identifiers.
- The complete-`K=9` eligibility-index SHA-256 is
  `e970ae0b03b4c1f9fbbd8bebcbe6082b309c610a0844f4c55cfd45a4df20f1b7`.
- The canonical filepath-keyed protocol manifest SHA-256 is
  `b28e4acc2c3482256c17971cd422e39713d5ca4df860bc2929a31b3104caa266`.
  Random outer folds were assigned within each label by sorting
  `SHA256(42|filepath)` and taking rank modulo five.

## Design consequence

Only 17.9% of archive patches and 216/279 identifiers have complete `K=9`
neighbourhoods. Moreover, complete-centre prevalence (56.4% IDC-positive)
differs substantially from archive prevalence (28.4%). The primary context
contrast must therefore:

1. compare `K=1` and `K=9` on exactly these same complete-neighbourhood centres;
2. use case-balanced AUPRC and whole-case bootstrap inference;
3. avoid comparing a 277,524-row `K=1` score against a 49,798-row `K=9` score;
4. report padded all-centre analysis only as sensitivity, with padding fraction
   available to detect border shortcuts;
5. state that 63 identifiers contribute no complete `K=9` centre.

The audit supports coordinate reconstruction but also identifies selection as a
material limitation of the complete-neighbourhood estimand.
