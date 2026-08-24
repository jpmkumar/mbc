from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from papers.paper_c.scripts.build_bcss_patient_splits import (
    site_from_patient,
    split_for,
)
from papers.paper_c.scripts.extract_embeddings import (
    PatchDataset,
    add_context_metadata,
    assert_existing_index,
    build_patch_index,
    parse_transform,
    pool_virchow2_tokens,
    write_index,
)


def _archive(tmp_path: Path, k: int = 3) -> Path:
    archive = tmp_path / "archive"
    for y in range(k):
        for x in range(k):
            label = "1" if (x, y) == (k // 2, k // 2) else "0"
            directory = archive / "case-a" / label
            directory.mkdir(parents=True, exist_ok=True)
            color = (x * 50, y * 50, 25)
            image = Image.new("RGB", (50, 50), color)
            image.save(
                directory
                / f"case-a_idx5_x{x * 50}_y{y * 50}_class{label}.png"
            )
    return archive


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    return torch.from_numpy(array.copy())


def test_context_metadata_and_mosaic_geometry(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    rows = build_patch_index(archive)
    lookup = {
        (row["case_id"], row["x"], row["y"]): row["filepath"] for row in rows
    }
    add_context_metadata(rows, lookup, 3)

    centre_index = next(
        idx for idx, row in enumerate(rows) if row["x"] == row["y"] == 50
    )
    assert rows[centre_index]["context_complete"] == 1
    assert sum(row["context_complete"] for row in rows) == 1

    dataset = PatchDataset(rows, archive, _to_tensor, 3, lookup)
    mosaic, returned_index = dataset[centre_index]
    assert returned_index == centre_index
    assert tuple(mosaic.shape) == (3, 150, 150)
    assert torch.allclose(
        mosaic[:, 50:100, 50:100].mean(dim=(1, 2)),
        torch.tensor([50 / 255, 50 / 255, 25 / 255]),
        atol=1e-5,
    )


def test_index_join_contract_detects_order_change(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    rows = build_patch_index(archive)
    lookup = {
        (row["case_id"], row["x"], row["y"]): row["filepath"] for row in rows
    }
    add_context_metadata(rows, lookup, 1)
    index_path = tmp_path / "index.csv"
    write_index(index_path, rows)
    assert assert_existing_index(index_path, rows)

    with pytest.raises(RuntimeError, match="differs"):
        assert_existing_index(index_path, list(reversed(rows)))

    with index_path.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert [row["filepath"] for row in saved] == [row["filepath"] for row in rows]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("upsample224", 1), ("mosaic3", 3), ("mosaic5", 5), ("mosaic9", 9)],
)
def test_transform_parser(name: str, expected: int) -> None:
    assert parse_transform(name) == expected


def test_transform_parser_rejects_unregistered_context() -> None:
    with pytest.raises(SystemExit):
        parse_transform("mosaic7")


def test_virchow2_pooling_skips_register_tokens() -> None:
    tokens = torch.zeros(2, 261, 1280)
    tokens[:, 0] = 3
    tokens[:, 1:5] = 100
    tokens[:, 5:] = 7
    pooled = pool_virchow2_tokens(tokens)
    assert pooled.shape == (2, 2560)
    assert torch.all(pooled[:, :1280] == 3)
    assert torch.all(pooled[:, 1280:] == 7)
    with pytest.raises(RuntimeError):
        pool_virchow2_tokens(torch.zeros(1, 260, 1280))


def test_bcss_patient_and_site_partition_is_deterministic() -> None:
    patient_id = "TCGA-OL-A5D6"
    site = site_from_patient(patient_id)
    assert site == "OL"
    assert split_for(patient_id, site) == "test"

    observed = {
        split_for(f"TCGA-A2-{index:04d}", "A2")
        for index in range(500)
    }
    assert observed == {"train", "val", "cal"}
    assert split_for("TCGA-A2-0042", "A2") == split_for(
        "TCGA-A2-0042", "A2"
    )
