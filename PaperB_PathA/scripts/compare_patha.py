#!/usr/bin/env python3
"""Compare the Path A arms against each other and against the published runs.

Reports, per fold and arm: the selected stage, per-stage best validation
scores, the selection margin, whether Stage C collapsed, and the test metrics
of the selected checkpoint.

Only fair-versus-control supports a claim. The published Kaggle runs are shown
as an environment-replication observation, because they were produced in a
different execution environment.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_GLOB = "results/histopath_kaggle_fold*_e3_v2/**/*progress.json"

# Containerised runs land under the server's primary root; the bare-Python
# fallback writes inside the repository. Both are searched so the analysis works
# whichever launcher produced the cells.
PRIMARY_ROOT = Path(os.environ.get("MBC_PRIMARY_ROOT", Path.home() / "mbc-primary"))
PATHA_DIRS = [PRIMARY_ROOT / "results" / "path-a", ROOT / "PaperB_PathA" / "results"]

ARMS = {"termwarm": "control (terminal weights)", "fairwarm": "fair (best checkpoint)"}


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def stage_segments(history: dict, progress: dict) -> dict[str, list[float]]:
    """Split the validation curve into its per-stage segments."""
    done = progress.get("stage_epochs_done", {})
    na = int(done.get("stage_a", 0))
    nb = int(done.get("stage_b", 0))
    nc = int(done.get("stage_c", 0))
    vals = [m.get("balanced_accuracy") for m in history.get("val_metrics", [])]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return {
        "stage_a": vals[:na],
        "stage_b": vals[na : na + nb],
        "stage_c": vals[na + nb : na + nb + nc],
    }


def summarise(progress_path: Path) -> dict | None:
    progress = load_json(progress_path)
    history = load_json(Path(str(progress_path).replace("_progress.json", "_history.json")))
    if not progress or not history:
        return None
    seg = stage_segments(history, progress)
    a, b, c = seg["stage_a"], seg["stage_b"], seg["stage_c"]
    if not a:
        return None
    rec = {
        "selected_stage": progress.get("best_stage"),
        "selected_score": progress.get("best_score"),
        "epochs": progress.get("stage_epochs_done"),
        "a_best": max(a),
        "a_last": a[-1],
        "b_best": max(b) if b else None,
        "c_best": max(c) if c else None,
    }
    rec["b_minus_a_best"] = (rec["b_best"] - rec["a_best"]) if b else None
    rec["b_minus_a_last"] = (rec["b_best"] - rec["a_last"]) if b else None
    rec["stage_c_collapsed"] = bool(c) and max(c) <= 0.5 + 1e-9
    rec["stage_init_from_best"] = progress.get("stage_init_from_best")
    rec["stage_transitions"] = progress.get("stage_transitions")
    return rec


def collect_patha() -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {tag: {} for tag in ARMS}
    for tag in ARMS:
        for fold in range(5):
            for base in PATHA_DIRS:
                if fold in out[tag]:
                    break
                if not base.exists():
                    continue
                for hit in sorted(base.glob(f"fold{fold}_{tag}/**/*progress.json")):
                    rec = summarise(hit)
                    if rec:
                        rec["source"] = str(hit)
                        out[tag][fold] = rec
                        break
    return out


def collect_published() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(glob.glob(str(ROOT / PUBLISHED_GLOB), recursive=True)):
        path = Path(p)
        try:
            fold = int(str(path).split("fold")[1][0])
        except (IndexError, ValueError):
            continue
        rec = summarise(path)
        if rec:
            rec["source"] = str(path.relative_to(ROOT))
            out[fold] = rec
    return out


def fmt(x, nd=4):
    return "     -" if x is None else f"{x:.{nd}f}"


def print_arm(title: str, recs: dict[int, dict]) -> None:
    print(f"\n=== {title} ===")
    if not recs:
        print("  no completed folds yet")
        return
    print(f"  {'fold':>4} {'selected':>9} {'A best':>8} {'A last':>8} {'B best':>8} "
          f"{'B-Abest':>9} {'B-Alast':>9} {'C collapsed':>12}")
    for fold in sorted(recs):
        r = recs[fold]
        print(f"  {fold:>4} {str(r['selected_stage']):>9} {fmt(r['a_best'])} "
              f"{fmt(r['a_last'])} {fmt(r['b_best'])} {fmt(r['b_minus_a_best'])} "
              f"{fmt(r['b_minus_a_last'])} {str(r['stage_c_collapsed']):>12}")
    quantum = [f for f, r in recs.items() if r["selected_stage"] not in (None, "stage_a")]
    print(f"  quantum-selected folds: {len(quantum)}/{len(recs)}"
          + (f"  -> {sorted(quantum)}" if quantum else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    patha = collect_patha()
    published = collect_published()

    if args.json:
        print(json.dumps({"path_a": patha, "published_kaggle": published}, indent=2))
        return

    print("PaperB Path A — fair-warmup comparison")
    print("Primary endpoint: folds in which validation selects a quantum stage (fair arm).")

    print_arm("Published five-fold E3 (Kaggle; reference only)", published)
    print_arm(f"Arm A0 — {ARMS['termwarm']}", patha["termwarm"])
    print_arm(f"Arm A1 — {ARMS['fairwarm']}", patha["fairwarm"])

    ctrl, fair = patha["termwarm"], patha["fairwarm"]
    paired = sorted(set(ctrl) & set(fair))
    print("\n=== A1 vs A0, paired within environment (the only comparison that counts) ===")
    if not paired:
        print("  no fold has both arms complete yet")
    else:
        print(f"  {'fold':>4} {'A0 selected':>12} {'A1 selected':>12} {'A0 margin':>10} {'A1 margin':>10}")
        for fold in paired:
            print(f"  {fold:>4} {str(ctrl[fold]['selected_stage']):>12} "
                  f"{str(fair[fold]['selected_stage']):>12} "
                  f"{fmt(ctrl[fold]['b_minus_a_best'])} {fmt(fair[fold]['b_minus_a_best'])}")
        flipped = [f for f in paired
                   if ctrl[f]["selected_stage"] == "stage_a"
                   and fair[f]["selected_stage"] != "stage_a"]
        print()
        if flipped:
            print(f"  DECISION: a quantum stage wins in fold(s) {flipped} under the fair schedule.")
            print("  The published selection claim is conditional on the terminal-weight")
            print("  schedule and the manuscript's central claim must be rewritten.")
        elif len(paired) == 5:
            print("  DECISION: Stage A wins in all five folds under the fair schedule.")
            print("  The published conclusion is strengthened; replace the schedule caveat")
            print("  in the Limitations with this result.")
        else:
            print(f"  {len(paired)}/5 folds paired so far; decision rule needs all five.")

    print("\n=== environment replication check (A0 vs published Kaggle) ===")
    both = sorted(set(ctrl) & set(published))
    if not both:
        print("  no control fold complete yet")
    else:
        for fold in both:
            d = ctrl[fold]["a_best"] - published[fold]["a_best"]
            print(f"  fold {fold}: Stage-A best server {fmt(ctrl[fold]['a_best'])} "
                  f"vs Kaggle {fmt(published[fold]['a_best'])}  (delta {d:+.4f})")
        print("  Large deltas mean the environments are not interchangeable; report them")
        print("  rather than pooling the two.")


if __name__ == "__main__":
    main()
