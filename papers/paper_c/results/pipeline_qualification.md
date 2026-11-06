# Paper C pipeline qualification status

**Date:** 21 August 2026; amended 6 November 2026.

Completed before GPU access:

- DOI verification: 81/81 candidate journal articles accepted by Crossref.
- IDC archive/context audit: 277,524 rows, 279 public case identifiers, no
  duplicate coordinates; complete `K=9` population fixed at 49,798 centres.
- Canonical filepath-keyed IDC protocol manifest built and hash-locked.
- Five deterministic inner train/validation/calibration case-ID partitions
  generated and hash-recorded.
- Paper C unit/integration tests: later expanded and passing in the qualified
  image.
- Revision-pinned ResNet path historically smoked from a local snapshot; the
  2026-11-06 redesign loads ResNet-50 via `torchvision` and DINOv2 via
  `torch.hub` (no Hugging Face token).

**2026-11-06 redesign (before labelled comparisons):**

- Co-primary reference encoder is **DINOv2 ViT-L/14**.
- Confirmatory panel: DINOv2 ViT-L/14 + ImageNet ResNet-50.
- UNI2-h, Virchow2 and Phikon-v2 are **unavailable** (institutional gated
  Hugging Face access cannot be completed). They are not replaced after
  outcomes.

Completed on the RTX A4000 (2026-11-06):

1. Synced ungated registry / loaders onto the server;
2. Smoke-qualified `dinov2_vitl14` and `resnet50` (`PASS`);
3. fp16-versus-fp32 equivalence for both (`PASS` under cosine +
   L2-normalised absolute gates).

Completed on the RTX A4000 (2026-11-06, continued):

4. BCSS centre manifest rebuilt at **stride 150**: 68,784 centres
   (27,690 positive); SHA-256
   `2e47a2cc874966703228c3b5bed2c5621b785ef7e2f9efb22b0295cda9ec25a8`.
   Prior stride-50 file archived as `bcss_centres.stride50.bak.csv`.

Still required before labelled comparisons:

1. Record GPU/container reports and lock the preregistration;
2. Full IDC / BCSS embedding extraction under the locked protocol.

No labelled model comparison has been inspected.
