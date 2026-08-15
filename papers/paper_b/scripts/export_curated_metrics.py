#!/usr/bin/env python3
"""Export a curated Paper-A metrics snapshot from local experiment results."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "papers/paper_b/results"
HISTOPATH = ROOT / "results/histopath"

SOURCES = {
    "cv_summary": HISTOPATH / "cv_summary.json",
    "stage_b_v2_final": HISTOPATH / "vqc_stage_b_crossfold_v2_final.json",
    "stage_b_v3_final": HISTOPATH / "vqc_stage_b_crossfold_v3_final.json",
    "stage_b_v2_sensitivity": HISTOPATH / "vqc_stage_b_crossfold_v2_sensitivity_folds1_4.json",
    "stage_b_v3_sensitivity": HISTOPATH / "vqc_stage_b_crossfold_v3_sensitivity_folds1_4.json",
}


def load_optional(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    present = {}
    missing = []
    for name, path in SOURCES.items():
        payload = load_optional(path)
        if payload is None:
            missing.append(str(path.relative_to(ROOT)))
        else:
            present[name] = payload

    snapshot = {
        "exported_utc": stamp,
        "paper": "paper_b_histopath_equivalence",
        "sources_present": sorted(present),
        "sources_missing": missing,
        "metrics": present,
    }
    out_json = OUT_DIR / "curated_metrics.json"
    out_json.write_text(json.dumps(snapshot, indent=2) + "\n")

    log_lines = [
        f"export_utc={stamp}",
        f"output={out_json.relative_to(ROOT)}",
        "present:",
        *[f"  - {name}: {SOURCES[name].relative_to(ROOT)}" for name in present],
        "missing:",
        *[f"  - {path}" for path in missing],
    ]
    (OUT_DIR / "export_log.txt").write_text("\n".join(log_lines) + "\n")

    print(json.dumps({"output": str(out_json), "present": len(present), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
