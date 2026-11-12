# Paper C curated results

`curated_metrics.json` holds the locked co-primary and ordered-secondary
estimates used in the manuscript. Raw prediction caches and full inference
artifacts remain on the GPU server under `~/mbc-primary/results/paper-c/`
(and a local sync mirror in `locked_inference/`, gitignored).

## Co-primary (DINOv2 ViT-L/14)

| Contrast | Δ AUPRC | 95% CI | Holm \(p\) | Reject @ 0.05 |
|---|---|---|---|---|
| random−grouped (K=1, case-balanced) | +0.0235 | [0.0152, 0.0332] | 0.00020 | yes |
| mosaic9−upsample (complete K=9 pop.) | −0.0057 | [−0.0197, 0.0074] | 0.405 | no |

## Ordered secondaries (summary)

- Bundled random−grouped: Δ=+0.0192, bootstrap \(p\)≈0.0003
- BCSS DINOv2 k9−k1 (site-stratified): Δ=+0.0051, \(p\)=0.56
- BCSS ResNet-50 k9−k1: Δ=+0.0422, \(p\)=0.0025
- Model rank (DINOv2 ≻ ResNet) stable across protocols (Kendall τ=1)
