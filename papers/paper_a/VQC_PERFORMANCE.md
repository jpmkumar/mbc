# Why VQC underperforms on WBCD — and what to try next

Campus GPU investigation (`results/gpu_investigation/`, 2026-11-24)
plus `quantum_diagnosis.json` and confirmatory `VQC_8` (~0.84 bal.acc).

## What is *not* broken

| Check | Result |
|-------|--------|
| Vanishing gradients | No — grad L2 rises ~19× by epoch 30 |
| Expressivity / random labels | Train bal.acc ~0.51 — neither VQC nor MLP memorizes noise |
| Entanglement useless | No — strong SEL **0.84** ≫ none **0.63** ≫ basic **0.48** |
| Sample size only | MLP already strong at 25% data; gap persists at 100% |

So this is not a “forgot to train” bug. It is a **model / measurement / objective** gap.

## What we are doing that caps VQC

1. **Single-qubit readout (`⟨Z₀⟩` only)**  
   Eight qubits are prepared; seven are thrown away at measurement. MLP emits a
   2-D logit. This is the largest structural handicap.

2. **MSE on ±1 vs CE on 2-class logits**  
   Confirmatory default matches older QML papers; `--vqc-loss ce` is fairer and
   usually lifts recall. Investigation / diagnosis still often used MSE.

3. **Shallow circuit (L=2, 48 params) vs richer classical heads**  
   Depth ablation: L1→L4 raises VQC ~0.53→~0.91, but MLP stays ~0.95 at every
   depth. Depth helps; it does not close the gap alone.

4. **PCA-8 angle embedding**  
   SVM-RBF on **30-D** wins (~0.975). VQC never sees the native table except in
   morphology / side-branch arms (morph was *worse*). Compression is shared with
   `MLP_PCA8`, so it does not explain VQC≪MLP on the *same* 8-D — but it does
   explain VQC≪SVM_RBF.

5. **Malignant errors**  
   ~12 FN/fold (VQC) vs ~3 (MLP) vs ~2 (SVM). Brier 0.12 vs 0.03 — calibration
   and decision surface, not just ranking (AUC already high ~0.97).

6. **BasicEntanglerLayers collapse**  
   Worse than encoding-only — wrong inductive bias / weight layout for this task,
   not “more entanglement always helps.”

## Honest ceiling

Locked narrative for Frontiers: **matched classical wins**. Chasing a VQC that
beats SVM-RBF on WBCD is possible with heavy hybridisation (quantum feature map
+ big classical head), but then the claim becomes “quantum features,” not
“standalone VQC classifier.” Keep that distinction in the manuscript.

Practical target for a **boost** arm: close on `MLP_PCA8` (~0.96), not necessarily
beat SVM-RBF (~0.975).

## Ranked next experiments

| Priority | Change | Why |
|----------|--------|-----|
| P0 | Multi-Z readout + linear CE head | Fixes measurement bottleneck |
| P0 | CE loss, 80 epochs, LR grid | Align training with MLP |
| P1 | L=4 SEL (or re-uploading ×2) | Depth already helped in diagnosis |
| P1 | Threshold / temperature on malignant | FN-heavy errors |
| P2 | Amplitude / richer encoding on 30-D | Only if P0/P1 stall |
| Avoid | Blind morph-10 (already failed KTA) | Measured worse |

Script: `scripts/train_wbcd_vqc_boost.py` (multi-Z + CE + depth/reupload sweep).

### Local smoke (`--quick`, 2 folds, 2026-11-24)

| Model | bal.acc mean |
|-------|--------------|
| **vqc_allz_ce_L2** (multi-Z + linear CE) | **0.949** |
| linear_pca8 | 0.916 |
| mlp_pca8 | 0.893 |
| baseline_z0_mse_L2 (old recipe) | 0.854 |

Multi-Z readout alone closed most of the gap in this smoke. Run full 5-fold on
the campus box before claiming numbers in the manuscript.
