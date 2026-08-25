#!/usr/bin/env python3
"""Export curated Paper-A (tabular WBCD) metrics snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "papers/paper_a/results"
PUBLICATION = ROOT / "publication/publication_metrics.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics = {}
    missing = []
    if PUBLICATION.exists():
        payload = json.loads(PUBLICATION.read_text())
        if "pilot_wbcd_tabular" in payload:
            metrics["reference_holdout_wbcd"] = payload["pilot_wbcd_tabular"]
        else:
            missing.append("pilot_wbcd_tabular")
    else:
        missing.append(str(PUBLICATION.relative_to(ROOT)))

    confirmatory = OUT_DIR / "confirmatory_cv.json"
    if confirmatory.exists():
        metrics["confirmatory_cv"] = json.loads(confirmatory.read_text())
    else:
        missing.append("confirmatory_cv")

    snapshot = {
        "exported_utc": stamp,
        "paper": "paper_a_tabular_wbcd",
        "sources_present": sorted(metrics),
        "sources_missing": missing,
        "metrics": metrics,
    }
    out_json = OUT_DIR / "curated_metrics.json"
    out_json.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({"output": str(out_json), "present": len(metrics)}, indent=2))


if __name__ == "__main__":
    main()
