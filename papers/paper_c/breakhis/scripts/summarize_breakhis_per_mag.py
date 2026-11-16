#!/usr/bin/env python3
"""Per-magnification protocol gaps from seed-mean BreakHis OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(ROOT / "papers" / "paper_c" / "scripts"))
from run_idc_probe_cv import case_weights  # noqa: E402


def case_balanced_auprc(frame: pd.DataFrame) -> float:
    y = frame["label"].to_numpy()
    p = frame["probability"].to_numpy()
    w = case_weights(frame["case_id"].astype(str).to_numpy())
    return float(average_precision_score(y, p, sample_weight=w))


def mag_bootstrap_delta(
    random_f: pd.DataFrame,
    grouped_f: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict:
    """Case bootstrap within one magnification slice (cases that have that mag)."""
    paired = random_f[["filepath", "case_id", "label", "probability"]].merge(
        grouped_f[["filepath", "case_id", "label", "probability"]],
        on="filepath",
        validate="one_to_one",
        suffixes=("_r", "_g"),
    )
    if not (
        (paired["case_id_r"].astype(str) == paired["case_id_g"].astype(str)).all()
        and (paired["label_r"] == paired["label_g"]).all()
    ):
        raise SystemExit("per-mag paired metadata disagree")
    cases = paired["case_id_r"].astype(str).to_numpy()
    y = paired["label_r"].to_numpy()
    pr = paired["probability_r"].to_numpy()
    pg = paired["probability_g"].to_numpy()
    unique = np.unique(cases)

    def weighted(prob: np.ndarray, multiplicity: dict[str, int]) -> float:
        rows = {c: n for c, n in zip(*np.unique(cases, return_counts=True))}
        w = np.asarray([multiplicity.get(c, 0) / rows[c] for c in cases], dtype=np.float64)
        return float(average_precision_score(y, prob, sample_weight=w))

    base = {c: 1 for c in unique}
    est = weighted(pr, base) - weighted(pg, base)
    rng = np.random.default_rng(seed)
    diffs = np.empty(replicates)
    for i in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sel, counts = np.unique(sampled, return_counts=True)
        mult = dict(zip(sel, counts))
        diffs[i] = weighted(pr, mult) - weighted(pg, mult)
    extreme = int(np.sum(np.abs(diffs - est) >= abs(est)))
    return {
        "difference": est,
        "random_auprc": weighted(pr, base),
        "grouped_auprc": weighted(pg, base),
        "ci_95": [
            float(np.quantile(diffs, 0.025)),
            float(np.quantile(diffs, 0.975)),
        ],
        "two_sided_bootstrap_p": (extreme + 1) / (replicates + 1),
        "n_cases": int(len(unique)),
        "n_images": int(len(paired)),
        "replicates": replicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path.home() / "mbc-primary" / "results" / "paper-c-plus",
    )
    parser.add_argument("--encoder", default="dinov2_vitl14")
    parser.add_argument("--head", default="fmlp", choices=("fmlp", "linear"))
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=53026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random_path = (
        args.results_root / args.encoder / f"{args.head}-random" / "seed_mean_predictions.csv"
    )
    grouped_path = (
        args.results_root / args.encoder / f"{args.head}-grouped" / "seed_mean_predictions.csv"
    )
    random_f = pd.read_csv(random_path, dtype={"filepath": str, "case_id": str})
    grouped_f = pd.read_csv(grouped_path, dtype={"filepath": str, "case_id": str})
    if "magnification" not in random_f.columns or "magnification" not in grouped_f.columns:
        raise SystemExit("seed-mean files need magnification column")

    def mag_key(value: object) -> tuple:
        text = str(value)
        digits = "".join(ch for ch in text if ch.isdigit())
        return (int(digits) if digits else 0, text)

    mags = sorted(
        set(random_f["magnification"].astype(str))
        & set(grouped_f["magnification"].astype(str)),
        key=mag_key,
    )
    rows = []
    for offset, mag in enumerate(mags):
        r = random_f[random_f["magnification"].astype(str) == mag]
        g = grouped_f[grouped_f["magnification"].astype(str) == mag]
        boot = mag_bootstrap_delta(
            r, g, replicates=args.replicates, seed=args.seed + offset
        )
        rows.append({"magnification": mag, **boot})
        print(json.dumps(rows[-1]), flush=True)

    report = {
        "encoder": args.encoder,
        "head": args.head,
        "contrast": "random_minus_grouped case_balanced_AUPRC",
        "per_magnification": rows,
        "note": (
            "Ordered secondary. Bootstrap resamples cases that contribute "
            "images at that magnification only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
