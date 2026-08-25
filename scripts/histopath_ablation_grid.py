#!/usr/bin/env python3
"""Fold-0 E3 expressivity grid: encoding × depth × entanglement.

Purpose
-------
Test whether the paper's null result (E3 ≈ E2b; best_stage=stage_a) depends on
the default VQC knobs (angle-Y, 2 layers, linear CNOT). Width ablations
(q4/q12) and entanglement=none are already finished; this grid adds the
remaining axes the manuscript limitations call out.

Default = 8 qubits, same training bundle as the paper (focal γ=2, strong aug,
TTA, Fβ β=1.5). Run on Kaggle GPU (T4); Mac has no CUDA for full folds.

Usage
-----
  # List cells (default)
  python scripts/histopath_ablation_grid.py

  # Print one Kaggle train command
  python scripts/histopath_ablation_grid.py --cell enc_ax --print-cmd

  # Run locally / on Kaggle (requires splits + archive)
  python scripts/histopath_ablation_grid.py --cell L1 --run --archive-path "$ARCHIVE"

  # Sequential batch (careful: multi-hour)
  python scripts/histopath_ablation_grid.py --run-pending --archive-path "$ARCHIVE"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fixed: fold 0, E3, 8 qubits (width already ablated separately).
FOLD = 0
EXPERIMENT = "E3"

# (cell_id, backup_label, CLI extras, status, note)
# status: done = already in results/; pending = needs Kaggle run
GRID: list[tuple[str, str, list[str], str, str]] = [
    (
        "baseline",
        "fold0_e3_v2",
        [],
        "done",
        "Default paper E3: angle_y, L=2, linear (bal_acc 0.886 / F1 0.816)",
    ),
    (
        "entnone",
        "fold0_e3_entnone",
        ["--entanglement", "none"],
        "done",
        "No CNOT — already reported in Table 6",
    ),
    (
        "circular",
        "fold0_e3_entcircular",
        ["--entanglement", "circular"],
        "pending",
        "Ring CNOT vs linear chain",
    ),
    (
        "L1",
        "fold0_e3_L1",
        ["--n-layers", "1"],
        "pending",
        "Shallower ansatz",
    ),
    (
        "L4",
        "fold0_e3_L4",
        ["--n-layers", "4"],
        "pending",
        "Deeper ansatz",
    ),
    (
        "enc_ax",
        "fold0_e3_encax",
        ["--encoding", "angle_x"],
        "pending",
        "Angle embedding on X instead of Y",
    ),
    (
        "enc_az",
        "fold0_e3_encaz",
        ["--encoding", "angle_z"],
        "pending",
        "Angle embedding on Z instead of Y",
    ),
    (
        "reup",
        "fold0_e3_reup",
        ["--data-reuploading"],
        "pending",
        "Encode before each layer (Pérez-Salinas-style)",
    ),
    (
        "L1_none",
        "fold0_e3_L1_entnone",
        ["--n-layers", "1", "--entanglement", "none"],
        "pending",
        "Corner: shallow + no entanglement",
    ),
    (
        "ax_L4",
        "fold0_e3_encax_L4",
        ["--encoding", "angle_x", "--n-layers", "4"],
        "pending",
        "Corner: alternate encoding + deeper circuit",
    ),
]


def _cell_map() -> dict[str, tuple[str, list[str], str, str]]:
    return {cid: (label, flags, status, note) for cid, label, flags, status, note in GRID}


def build_cmd(flags: list[str], archive_path: str | None) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_histopath_cv.py"),
        "--fold",
        str(FOLD),
        "--experiment",
        EXPERIMENT,
    ]
    if archive_path:
        cmd.extend(["--archive-path", archive_path])
    cmd.extend(flags)
    return cmd


def print_table() -> None:
    print(
        "Fold-0 E3 ablation grid (8 qubits fixed; width q4/q12 already done separately)\n"
    )
    print(f"{'cell':<12} {'status':<8} {'backup_label':<28} flags / note")
    print("-" * 88)
    for cid, label, flags, status, note in GRID:
        flag_s = " ".join(flags) if flags else "(default)"
        print(f"{cid:<12} {status:<8} {label:<28} {flag_s}")
        print(f"{'':12} {'':8} {'':28} {note}")
    pending = [c for c, *_ , st, _ in GRID if st == "pending"]
    print("-" * 88)
    print(f"Pending cells: {len(pending)} → {', '.join(pending)}")
    print("Expected time: ~3–5 h per cell on Kaggle T4 (Save & Run All, one cell per version).")
    print("\nWhat it answers: does changing encoding / depth / entanglement make")
    print("validation prefer Stage B/C, or beat E2b? If best_stage stays stage_a,")
    print("the null result is not an artefact of the default circuit knobs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", help="Grid cell id (e.g. L1, enc_ax)")
    parser.add_argument("--print-cmd", action="store_true", help="Print train command")
    parser.add_argument("--run", action="store_true", help="Run one --cell")
    parser.add_argument(
        "--run-pending",
        action="store_true",
        help="Run all pending cells sequentially (multi-hour)",
    )
    parser.add_argument("--archive-path", default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump grid definition as JSON",
    )
    args = parser.parse_args()
    cells = _cell_map()

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "cell": cid,
                        "backup_label": label,
                        "flags": flags,
                        "status": status,
                        "note": note,
                    }
                    for cid, label, flags, status, note in GRID
                ],
                indent=2,
            )
        )
        return

    if args.cell:
        if args.cell not in cells:
            raise SystemExit(f"Unknown cell {args.cell!r}. Choose: {', '.join(cells)}")
        label, flags, status, note = cells[args.cell]
        cmd = build_cmd(flags, args.archive_path)
        if args.print_cmd or not args.run:
            print("#", args.cell, "|", label, "|", status)
            print("#", note)
            print(" ".join(cmd))
        if args.run:
            if status == "done":
                print(f"WARNING: {args.cell} marked done; re-running anyway.", file=sys.stderr)
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        return

    if args.run_pending:
        if not args.archive_path:
            raise SystemExit("--run-pending requires --archive-path")
        for cid, label, flags, status, note in GRID:
            if status != "pending":
                continue
            cmd = build_cmd(flags, args.archive_path)
            print("\n" + "=" * 60)
            print("START", cid, label, flags)
            print(note)
            subprocess.run(cmd, cwd=ROOT, check=True)
            print("DONE", cid)
        return

    print_table()


if __name__ == "__main__":
    main()
