#!/usr/bin/env python3
"""Update paper/main.tex results table from experiment JSON."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "experiment_matrix_summary.json"
TEX = ROOT / "paper" / "main.tex"


def _fmt(m: dict, key: str) -> str:
    if key in m:
        return f"{m[key]:.3f}"
    return "---"


def main():
    if not SUMMARY.exists():
        print(f"No summary at {SUMMARY}; skipping tex update")
        return

    with open(SUMMARY) as f:
        s = json.load(f)

    e2 = s.get("E2", {})
    e3 = s.get("E3", {})
    rows = [
        ("Classical Unified (E2)", f"{e2.get('f1_mean', 0):.3f}"),
        ("Hybrid Full (E3)", f"{e3.get('f1_mean', 0):.3f}"),
    ]
    for mod in ("mammo", "ultrasound", "thermo"):
        key = f"E4_{mod}"
        if key in s and "test_metrics" in s[key]:
            tm = s[key]["test_metrics"]
            rows.append((f"LOMO {mod.capitalize()}", _fmt(tm, "f1")))

    tex = TEX.read_text()
    new_rows = "\n".join(
        f"{name} & --- & {f1} & --- \\\\" for name, f1 in rows
    )
    tex = re.sub(
        r"Classical Unified \(E2\) & --- & --- & --- \\\\.*?LOMO Thermography & --- & --- & --- \\\\",
        new_rows,
        tex,
        flags=re.DOTALL,
    )
    TEX.write_text(tex)
    print(f"Updated {TEX}")


if __name__ == "__main__":
    main()
