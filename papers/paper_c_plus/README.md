# Paper C+ — multi-cohort classical replication

Follow-up to [Paper C](../paper_c/). **Does not modify** Paper C’s locked
IDC/BCSS analysis.

| Item | Choice |
|---|---|
| Models | DINOv2 ViT-L/14 + ResNet-50 (ungated), frozen **F-MLP** confirmatory |
| Protocol contrast | random vs case-grouped (same head both arms) |
| Secondaries | linear probe (Paper C bridge); optional T-MLP fine-tune |
| First new cohort | BreakHis (campus archive already present) |
| Second cohort | BACH (planned) |
| Overlap with Paper D | Same BreakHis files allowed; separate splits/lock/estimands |

Start at [`PLAN.md`](PLAN.md) and
[`../../preregistration/paper_c_plus_protocol.md`](../../preregistration/paper_c_plus_protocol.md).
