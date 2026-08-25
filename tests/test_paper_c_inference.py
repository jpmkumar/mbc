from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from papers.paper_c.scripts.average_seed_predictions import main as average_main
from papers.paper_c.scripts.holm_coprimary import main as holm_main
from papers.paper_c.scripts.paired_case_bootstrap import main as bootstrap_main


def _prediction_frame(rng: np.random.Generator, better: bool) -> pd.DataFrame:
    rows = []
    for case_index in range(20):
        for patch_index in range(10):
            label = (case_index + patch_index) % 2
            signal = 0.75 if better else 0.55
            probability = signal if label else 1 - signal
            probability += rng.normal(0, 0.18 if better else 0.25)
            rows.append({
                "filepath": f"case-{case_index}/patch-{patch_index}.png",
                "case_id": f"case-{case_index}",
                "label": label,
                "probability": float(np.clip(probability, 0.01, 0.99)),
            })
    return pd.DataFrame(rows)


def test_bootstrap_and_holm_outputs(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(9)
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _prediction_frame(rng, better=True).to_csv(left, index=False)
    _prediction_frame(rng, better=False).to_csv(right, index=False)

    protocol = tmp_path / "protocol.json"
    monkeypatch.setattr(sys, "argv", [
        "paired_case_bootstrap.py",
        "--left", str(left),
        "--right", str(right),
        "--left-label", "better",
        "--right-label", "worse",
        "--replicates", "200",
        "--output", str(protocol),
    ])
    assert bootstrap_main() == 0
    report = json.loads(protocol.read_text())
    assert report["difference"] > 0
    assert report["case_identifiers"] == 20

    context = tmp_path / "context.json"
    context.write_text(protocol.read_text())
    holm = tmp_path / "holm.json"
    monkeypatch.setattr(sys, "argv", [
        "holm_coprimary.py",
        "--protocol-report", str(protocol),
        "--context-report", str(context),
        "--output", str(holm),
    ])
    assert holm_main() == 0
    adjusted = json.loads(holm.read_text())
    assert set(adjusted["results"]) == {"protocol_optimism", "context_gain"}


def test_seed_averaging_joins_on_filepath(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(19)
    paths = []
    for seed in (42, 43, 44):
        frame = _prediction_frame(rng, better=True)
        frame["fold"] = [
            int(case_id.split("-")[1]) % 5 for case_id in frame["case_id"]
        ]
        frame["logit"] = np.log(frame["probability"] / (1 - frame["probability"]))
        frame["threshold"] = 0.5 + (seed - 43) * 0.01
        path = tmp_path / f"seed-{seed}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    output = tmp_path / "averaged.csv"
    monkeypatch.setattr(sys, "argv", [
        "average_seed_predictions.py",
        "--prediction", str(paths[0]), "--seed", "42",
        "--prediction", str(paths[1]), "--seed", "43",
        "--prediction", str(paths[2]), "--seed", "44",
        "--output", str(output),
    ])
    assert average_main() == 0
    averaged = pd.read_csv(output)
    assert len(averaged) == 200
    assert averaged["filepath"].is_unique
    assert np.allclose(averaged["threshold"], 0.5)
