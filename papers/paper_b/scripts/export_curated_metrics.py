#!/usr/bin/env python3
"""Export a curated Paper-B metrics snapshot from local experiment results.

The snapshot covers every summary artifact that the reported tables and figures
are generated from, so a reader who has only the released repository can check
the published numbers without the full run directories.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "papers/paper_b/results"
HISTOPATH = ROOT / "results/histopath"

# results/histopath/cv_summary.json is a stale single-arm artifact and cannot
# reproduce Table 3, so the deployed arms are resolved per fold from the run
# directories instead, using the same lookup as
# papers/paper_b/scripts/plot_paper_figures.py:load_arm. That script has no
# __main__ guard, so the logic is duplicated here rather than imported; the two
# must be kept in step.
DEPLOYED_ARMS = {
    "E2": "e2_v2",
    "E2b": "e2b",
    "E3": "e3_v2",
}

SOURCES = {
    "stage_b_v2_final": HISTOPATH / "vqc_stage_b_crossfold_v2_final.json",
    "stage_b_v3_final": HISTOPATH / "vqc_stage_b_crossfold_v3_final.json",
    "stage_b_v2_sensitivity": HISTOPATH / "vqc_stage_b_crossfold_v2_sensitivity_folds1_4.json",
    "stage_b_v3_sensitivity": HISTOPATH / "vqc_stage_b_crossfold_v3_sensitivity_folds1_4.json",
    "width_matrix_analysis": HISTOPATH / "width_matrix_analysis.json",
    "server_width_matrix": HISTOPATH / "server_width_matrix.json",
}


def load_optional(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_arm_fold(arm_key: str, exp_key: str, fold: int) -> tuple[dict, str] | None:
    """Return the deployed test metrics for one arm and fold, with its source."""
    candidates = sorted(
        (ROOT / "results").glob(f"histopath_kaggle_fold{fold}_{arm_key}*")
    )
    if arm_key in ("e2_v2", "e3_v2"):
        candidates = [d for d in candidates if arm_key in d.name]
    elif arm_key == "e2b":
        candidates = [d for d in candidates if "_e2b" in d.name]

    for directory in candidates:
        for path in sorted(directory.rglob("cv_summary.json")):
            summary = json.loads(path.read_text())
            results = summary.get("results") or {}
            entries = results.get(exp_key)
            if not entries and results:
                entries = next(iter(results.values()))
            if not entries:
                continue
            metrics = entries[0].get("test_metrics") or entries[0].get("train_metrics")
            if metrics and "f1" in metrics:
                return metrics, str(path.relative_to(ROOT))
    return None


def collect_deployed_arms() -> tuple[dict, list[str]]:
    """Per-fold deployed test metrics for E2, E2b and E3 (Table 3, Figure 5)."""
    arms: dict[str, dict] = {}
    missing: list[str] = []
    for exp_key, arm_key in DEPLOYED_ARMS.items():
        folds = {}
        for fold in range(5):
            resolved = resolve_arm_fold(arm_key, exp_key, fold)
            if resolved is None:
                missing.append(f"{exp_key} fold {fold}")
                continue
            metrics, source = resolved
            folds[f"fold_{fold}"] = {"source": source, "test_metrics": metrics}
        arms[exp_key] = folds
    return arms, missing


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

    deployed, deployed_missing = collect_deployed_arms()
    missing.extend(deployed_missing)
    present["deployed_arms"] = deployed

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
        *[
            f"  - {name}: {SOURCES[name].relative_to(ROOT)}"
            for name in present
            if name in SOURCES
        ],
        *[
            f"  - deployed_arms/{arm}/{fold}: {record['source']}"
            for arm, folds in deployed.items()
            for fold, record in folds.items()
        ],
        "missing:",
        *[f"  - {path}" for path in missing],
    ]
    (OUT_DIR / "export_log.txt").write_text("\n".join(log_lines) + "\n")

    print(json.dumps({"output": str(out_json), "present": len(present), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
