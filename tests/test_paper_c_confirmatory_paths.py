"""End-to-end coverage for the confirmatory Paper C paths.

The random IDC probe is covered in ``test_paper_c_probe_pipeline``. These tests
exercise the paths that carry the co-primary and external-replication claims:
grouped nested cross-validation, the harmonised/bundled weighting regimes, and
the BCSS institution-held-out probe including its provenance guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from papers.paper_c.scripts.run_idc_probe_cv import main as idc_main, sha256

SCRIPTS = Path(__file__).resolve().parents[1] / "papers" / "paper_c" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_bcss_probe import main as bcss_main  # noqa: E402

CASES = 20
FOLDS = 5


def _idc_fixture(tmp_path: Path, seed: int = 3) -> tuple[Path, Path, Path]:
    """Build an IDC cache, patch manifest and frozen grouped inner splits."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    for case in range(CASES):
        case_id = f"case-{case:03d}"
        # Deliberately uneven case sizes and class imbalance so that
        # case-balanced and class-balanced fitting cannot coincide.
        positives = 2 + case % 3
        negatives = 8 + case % 5
        for index in range(positives + negatives):
            label = 1 if index < positives else 0
            rows.append({
                "filepath": f"{case_id}/{label}/patch-{index:03d}.png",
                "case_id": case_id,
                "label": label,
                "grouped_outer_fold": case % FOLDS,
                "k9_complete": 1,
            })
            features.append(rng.normal(label * 1.6, 1.0, size=8).astype(np.float16))

    manifest = pd.DataFrame(rows)
    # Random outer folds are label-stratified across the pooled patches.
    manifest["random_outer_fold"] = 0
    for label in (0, 1):
        members = manifest.index[manifest["label"] == label].to_numpy()
        manifest.loc[members, "random_outer_fold"] = np.arange(len(members)) % FOLDS

    cache = tmp_path / "cache"
    cache.mkdir()
    index_frame = manifest[["filepath", "case_id", "label"]]
    index_frame.to_csv(cache / "index.csv", index=False)
    embeddings = np.stack(features)
    np.save(cache / "embeddings.npy", embeddings)
    (cache / "provenance.json").write_text(
        json.dumps({
            "complete": True,
            "context_k": 1,
            "n_centres": len(manifest),
            "embedding_dim": int(embeddings.shape[1]),
            "dtype": "float16",
            "index_sha256": sha256(cache / "index.csv"),
            "embeddings_sha256": sha256(cache / "embeddings.npy"),
        }) + "\n"
    )

    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Frozen inner splits. The path must contain paper_c/idc: the runner
    # refuses to borrow Paper B's partition.
    inner_root = tmp_path / "data" / "splits" / "paper_c" / "idc"
    case_ids = sorted({str(row["case_id"]) for row in rows})
    summary: dict[str, object] = {"folds": []}
    for fold in range(FOLDS):
        development = [
            case_id for case_id in case_ids
            if int(case_id.split("-")[1]) % FOLDS != fold
        ]
        train, val, cal = development[:10], development[10:13], development[13:]
        fold_dir = inner_root / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, ids in (
            ("inner_train", train), ("inner_val", val), ("inner_cal", cal)
        ):
            path = fold_dir / f"{name}_case_ids.csv"
            path.write_text("case_id\n" + "".join(f"{item}\n" for item in ids))
        summary["folds"].append({"fold": fold, "counts": {
            "inner_train": len(train), "inner_val": len(val), "inner_cal": len(cal),
        }})
    summary_path = inner_root / "inner_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return cache, manifest_path, inner_root


def _run_idc(
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    manifest_path: Path,
    inner_root: Path,
    output: Path,
    *,
    protocol: str,
    weighting: str | None = None,
) -> dict:
    argv = [
        "run_idc_probe_cv.py",
        "--cache", str(cache),
        "--patch-manifest", str(manifest_path),
        "--protocol", protocol,
        "--inner-case-splits", str(inner_root),
        "--inner-split-summary", str(inner_root / "inner_split_summary.json"),
        "--output-dir", str(output),
        "--max-iter", "25",
    ]
    if weighting:
        argv += ["--weighting", weighting]
    monkeypatch.setattr(sys, "argv", argv)
    assert idc_main() == 0
    return json.loads((output / "summary.json").read_text())


@pytest.mark.filterwarnings("ignore:Maximum number of iteration reached")
def test_grouped_probe_covers_every_patch_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, manifest_path, inner_root = _idc_fixture(tmp_path)
    output = tmp_path / "grouped"
    summary = _run_idc(
        monkeypatch, cache, manifest_path, inner_root, output, protocol="grouped"
    )

    predictions = pd.read_csv(output / "oof_predictions.csv", dtype={"case_id": str})
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    assert set(predictions["filepath"]) == set(manifest["filepath"])
    assert predictions["filepath"].is_unique
    assert set(predictions["fold"]) == set(range(FOLDS))
    assert summary["complete"] is True
    assert summary["case_weighted_fitting"] is True

    # A case may never be split across grouped folds.
    per_case_folds = predictions.groupby("case_id")["fold"].nunique()
    assert (per_case_folds == 1).all()


@pytest.mark.filterwarnings("ignore:Maximum number of iteration reached")
def test_weighting_regime_changes_only_the_random_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, manifest_path, inner_root = _idc_fixture(tmp_path)

    harmonised = _run_idc(
        monkeypatch, cache, manifest_path, inner_root, tmp_path / "random-harm",
        protocol="random", weighting="case-balanced",
    )
    bundled = _run_idc(
        monkeypatch, cache, manifest_path, inner_root, tmp_path / "random-bundled",
        protocol="random", weighting="protocol-native",
    )

    # The co-primary regime case-weights both arms; the bundled secondary
    # restores each arm's conventional weighting.
    assert harmonised["case_weighted_fitting"] is True
    assert bundled["case_weighted_fitting"] is False
    assert harmonised["weighting"] == "case-balanced"
    assert bundled["weighting"] == "protocol-native"

    left = pd.read_csv(tmp_path / "random-harm" / "oof_predictions.csv")
    right = pd.read_csv(tmp_path / "random-bundled" / "oof_predictions.csv")
    assert not np.allclose(left["probability"], right["probability"])

    # Grouped is case-weighted under either regime, so the co-primary contrast
    # differs from the bundled one only through the random arm.
    for weighting in ("case-balanced", "protocol-native"):
        summary = _run_idc(
            monkeypatch, cache, manifest_path, inner_root,
            tmp_path / f"grouped-{weighting}",
            protocol="grouped", weighting=weighting,
        )
        assert summary["case_weighted_fitting"] is True


DEFAULT_PLAN = [
    ("TCGA-A2-0001", "A2", "train"),
    ("TCGA-A2-0002", "A2", "train"),
    ("TCGA-A2-0003", "A2", "train"),
    ("TCGA-A2-0004", "A2", "train"),
    ("TCGA-BH-0005", "BH", "val"),
    ("TCGA-BH-0006", "BH", "val"),
    ("TCGA-C8-0007", "C8", "cal"),
    ("TCGA-C8-0008", "C8", "cal"),
    ("TCGA-OL-0009", "OL", "test"),
    ("TCGA-OL-0010", "OL", "test"),
    ("TCGA-LL-0011", "LL", "test"),
    ("TCGA-LL-0012", "LL", "test"),
]


def _bcss_fixture(
    tmp_path: Path,
    plan: list[tuple[str, str, str]] | None = None,
    seed: int = 5,
) -> tuple[Path, Path]:
    """Build a BCSS cache and centre manifest whose provenance is consistent."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    for patient, site, split in (plan or DEFAULT_PLAN):
        roi = f"{patient}-01Z-00-DX1"
        for index in range(8):
            label = index % 2
            rows.append({
                "filepath": f"{roi}:x{index * 50}:y0",
                "image_filename": f"{roi}.png",
                "mask_filename": f"{roi}.png",
                "roi_id": roi,
                "patient_id": patient,
                "site": site,
                "split": split,
                "x": index * 50,
                "y": 0,
                "label": label,
            })
            features.append(rng.normal(label * 1.8, 1.0, size=8).astype(np.float16))

    frame = pd.DataFrame(rows)
    manifest_path = tmp_path / "bcss_centres.csv"
    frame.to_csv(manifest_path, index=False)

    cache = tmp_path / "bcss_cache"
    cache.mkdir()
    frame.to_csv(cache / "index.csv", index=False)
    embeddings = np.stack(features)
    np.save(cache / "embeddings.npy", embeddings)
    (cache / "provenance.json").write_text(
        json.dumps({
            "complete": True,
            "cohort": "BCSS",
            "manifest_sha256": sha256(manifest_path),
            "n_centres": len(frame),
            "embedding_dim": int(embeddings.shape[1]),
            "dtype": "float16",
            "index_sha256": sha256(cache / "index.csv"),
            "embeddings_sha256": sha256(cache / "embeddings.npy"),
        }) + "\n"
    )
    return cache, manifest_path


def _run_bcss(
    monkeypatch: pytest.MonkeyPatch, cache: Path, manifest_path: Path, output: Path
) -> int:
    monkeypatch.setattr(sys, "argv", [
        "run_bcss_probe.py",
        "--cache", str(cache),
        "--centre-manifest", str(manifest_path),
        "--output-dir", str(output),
        "--max-iter", "25",
    ])
    return bcss_main()


@pytest.mark.filterwarnings("ignore:Maximum number of iteration reached")
def test_bcss_probe_scores_only_held_out_institutions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, manifest_path = _bcss_fixture(tmp_path)
    output = tmp_path / "bcss"
    assert _run_bcss(monkeypatch, cache, manifest_path, output) == 0

    summary = json.loads((output / "summary.json").read_text())
    predictions = pd.read_csv(output / "test_predictions.csv", dtype={"case_id": str})

    assert summary["cohort"] == "BCSS"
    assert sorted(summary["test_sites"]) == ["LL", "OL"]
    # Only held-out institutions may be scored.
    assert set(predictions["site"]) == {"LL", "OL"}
    assert set(predictions["case_id"]) == {
        patient for patient, _, split in DEFAULT_PLAN if split == "test"
    }
    assert {"sensitivity", "specificity"} <= set(summary["test_metrics"])


def test_bcss_probe_rejects_a_manifest_edited_after_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, manifest_path = _bcss_fixture(tmp_path)
    frame = pd.read_csv(manifest_path)
    frame.loc[0, "label"] = 1 - int(frame.loc[0, "label"])
    frame.to_csv(manifest_path, index=False)

    with pytest.raises(SystemExit, match="manifest differs from extraction"):
        _run_bcss(monkeypatch, cache, manifest_path, tmp_path / "bcss")


TEST_SITE_CODES = ["OL", "LL", "E2", "EW", "GM", "S3"]
DEV_SITE_CODES = ["A2", "BH", "C8", "A1", "E9"]


def _bcss_mirror(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Materialise a miniature 151-ROI BCSS mirror with RGB/mask pairs."""
    from PIL import Image

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    records = []
    roi_total = 0
    for patient_index in range(75):
        site = (
            TEST_SITE_CODES[patient_index]
            if patient_index < len(TEST_SITE_CODES)
            else DEV_SITE_CODES[patient_index % len(DEV_SITE_CODES)]
        )
        patient = f"TCGA-{site}-{patient_index:04d}"
        # Three ROIs for the first patient brings the mirror to exactly 151.
        for roi_index in range(3 if patient_index == 0 else 2):
            stem = f"{patient}-roi{roi_index}"
            # Every patient carries both a tumour and a non-tumour ROI, so any
            # non-empty partition has both classes.
            code = 1 if roi_index == 0 else 2
            Image.new("RGB", (60, 60), (200, 150, 200)).save(images / f"{stem}.png")
            Image.new("L", (60, 60), code).save(masks / f"{stem}.png")
            records.append({
                "image_id": stem,
                "patient_id": patient,
                "xmin": 0,
                "ymin": 0,
            })
            roi_total += 1

    assert roi_total == 151
    metadata = tmp_path / "mirror_rows.json"
    metadata.write_text(json.dumps(records, indent=2) + "\n")
    return images, masks, metadata


@pytest.mark.filterwarnings("ignore:Maximum number of iteration reached")
def test_bcss_centre_builder_reports_frozen_institutions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the centre builder, which must survive its own provenance report."""
    from papers.paper_c.scripts.build_bcss_centres import main as centres_main
    from papers.paper_c.scripts.build_bcss_patient_splits import (
        main as splits_main,
    )

    images, masks, metadata = _bcss_mirror(tmp_path)
    splits_path = tmp_path / "bcss_patient_splits.csv"
    monkeypatch.setattr(sys, "argv", [
        "build_bcss_patient_splits.py",
        "--metadata", str(metadata),
        "--output", str(splits_path),
    ])
    assert splits_main() == 0

    manifest_path = tmp_path / "bcss_centres.csv"
    monkeypatch.setattr(sys, "argv", [
        "build_bcss_centres.py",
        "--images", str(images),
        "--masks", str(masks),
        "--metadata", str(metadata),
        "--patient-splits", str(splits_path),
        "--output", str(manifest_path),
        "--stride", "20",
        "--centre-size", "10",
        "--context-size", "50",
    ])
    assert centres_main() == 0

    report = json.loads(manifest_path.with_suffix(".summary.json").read_text())
    assert report["images"] == 151
    assert sorted(report["test_sites"]) == sorted(TEST_SITE_CODES)
    assert not set(report["test_sites"]) & set(report["development_sites"])
    assert report["patient_split_sha256"] == sha256(splits_path)

    centres = pd.read_csv(manifest_path, dtype={"patient_id": str})
    assert set(centres["split"]) == {"train", "val", "cal", "test"}
    for split in ("train", "val", "cal", "test"):
        assert set(centres.loc[centres["split"] == split, "label"]) == {0, 1}
    # A patient may never appear in two partitions.
    assert (centres.groupby("patient_id")["split"].nunique() == 1).all()


def test_bcss_probe_rejects_institution_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaky = [
        (patient, site, "train" if patient == "TCGA-OL-0009" else split)
        for patient, site, split in DEFAULT_PLAN
    ]
    cache, manifest_path = _bcss_fixture(tmp_path, plan=leaky)

    with pytest.raises(SystemExit, match="test institutions leak"):
        _run_bcss(monkeypatch, cache, manifest_path, tmp_path / "bcss")
