# Paper C pipeline qualification status

**Date:** 21 August 2026.

Completed before GPU access:

- DOI verification: 81/81 candidate journal articles accepted by Crossref.
- IDC archive/context audit: 277,524 rows, 279 public case identifiers, no
  duplicate coordinates; complete `K=9` population fixed at 49,798 centres.
- Canonical filepath-keyed IDC protocol manifest built and hash-locked.
- Five deterministic inner train/validation/calibration case-ID partitions
  generated and hash-recorded.
- Paper C unit/integration tests: 12 passed.
- Revision-pinned `timm/resnet50.a1_in1k` snapshot loaded from a local
  Hugging Face snapshot at commit
  `767268603ca0cb0bfe326fa87277f19c419566ef`.
- End-to-end two-image CPU extraction smoke:
  2,048-dimensional output, finite cache, complete index/provenance and artifact
  hashes.

Still required on the qualified RTX A4000:

- resolve `docker/requirements-paperc.lock` against the pinned NGC image and
  hard-code its SHA-256;
- qualify UNI2-h, Virchow2, Phikon-v2, DINOv2 ViT-L/14 and ResNet-50 model
  dimensions and per-model batch sizes;
- confirm Virchow2 token pooling at runtime;
- pass fp16-versus-fp32 equivalence for every encoder;
- record GPU/container reports and lock the preregistration.

No labelled model comparison has been inspected.
