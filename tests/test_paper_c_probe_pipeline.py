from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from papers.paper_c.scripts.run_idc_probe_cv import main, sha256


@pytest.mark.filterwarnings("ignore:Maximum number of iteration reached")
def test_random_probe_pipeline_produces_complete_oof_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(7)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    features = []
    for label in (0, 1):
        for index in range(50):
            rows.append({
                "filepath": f"case-{index:03d}/{label}/patch-{label}-{index:03d}.png",
                "case_id": f"case-{index:03d}",
                "label": label,
            })
            features.append(
                rng.normal(loc=label * 1.5, scale=1.0, size=8).astype(np.float16)
            )
    pd.DataFrame(rows).to_csv(cache / "index.csv", index=False)
    np.save(cache / "embeddings.npy", np.stack(features))
    (cache / "provenance.json").write_text(
        json.dumps({
            "complete": True,
            "context_k": 1,
            "n_centres": 100,
            "embedding_dim": 8,
            "dtype": "float16",
            "index_sha256": sha256(cache / "index.csv"),
            "embeddings_sha256": sha256(cache / "embeddings.npy"),
        }) + "\n"
    )

    manifest = pd.DataFrame(rows)
    manifest["grouped_outer_fold"] = [
        index % 5 for _label in (0, 1) for index in range(50)
    ]
    manifest["random_outer_fold"] = manifest["grouped_outer_fold"]
    manifest["k9_complete"] = 1
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    output = tmp_path / "output"

    monkeypatch.setattr(sys, "argv", [
        "run_idc_probe_cv.py",
        "--cache", str(cache),
        "--patch-manifest", str(manifest_path),
        "--protocol", "random",
        "--output-dir", str(output),
        "--max-iter", "20",
    ])
    assert main() == 0

    predictions = pd.read_csv(output / "oof_predictions.csv")
    summary = json.loads((output / "summary.json").read_text())
    assert len(predictions) == 100
    assert predictions["filepath"].is_unique
    assert set(predictions["fold"]) == set(range(5))
    assert summary["complete"] is True
    assert summary["rows"] == 100
