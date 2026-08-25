from __future__ import annotations

import pandas as pd
import pytest

from papers.paper_c.scripts.build_idc_inner_splits import split_groups
from src.data.histopath_splits import split_train_val_cal_groups


def _groups(n_per_bin: int = 12) -> pd.DataFrame:
    rows = []
    for ratio_bin in range(4):
        for idx in range(n_per_bin):
            rows.append({
                "patient_id": f"{ratio_bin}-{idx:02d}",
                "ratio_bin": ratio_bin,
                "idc_ratio": (ratio_bin + 0.5) / 4,
            })
    return pd.DataFrame(rows)


def test_three_way_group_split_is_disjoint_complete_and_deterministic() -> None:
    frame = _groups()
    group_ids = frame["patient_id"].tolist()

    first = split_train_val_cal_groups(frame, group_ids, seed=1042)
    second = split_train_val_cal_groups(frame, group_ids, seed=1042)

    assert first == second
    train, val, cal = map(set, first)
    assert not train & val
    assert not train & cal
    assert not val & cal
    assert train | val | cal == set(group_ids)

    for ratio_bin in range(4):
        bin_ids = set(frame.loc[frame["ratio_bin"] == ratio_bin, "patient_id"])
        assert train & bin_ids
        assert val & bin_ids
        assert cal & bin_ids


def test_three_way_group_split_changes_with_seed() -> None:
    frame = _groups()
    group_ids = frame["patient_id"].tolist()
    assert split_train_val_cal_groups(
        frame, group_ids, seed=1042
    ) != split_train_val_cal_groups(frame, group_ids, seed=1043)


def test_generator_split_matches_library_implementation() -> None:
    """The frozen-artifact generator is stdlib-only, so guard against drift."""
    frame = _groups(n_per_bin=17)
    group_ids = frame["patient_id"].tolist()
    stats_by_id = {
        row["patient_id"]: {"ratio_bin": str(row["ratio_bin"])}
        for row in frame.to_dict("records")
    }

    for seed in (1042, 1043, 1044, 1045, 1046):
        assert split_groups(
            stats_by_id,
            group_ids,
            val_ratio=0.125,
            cal_ratio=0.125,
            seed=seed,
        ) == split_train_val_cal_groups(
            frame, group_ids, val_ratio=0.125, cal_ratio=0.125, seed=seed
        )


def test_three_way_group_split_rejects_missing_bins_and_unknown_ids() -> None:
    frame = _groups()
    with pytest.raises(ValueError, match="ratio_bin"):
        split_train_val_cal_groups(
            frame.drop(columns=["ratio_bin"]), frame["patient_id"].tolist()
        )
    with pytest.raises(ValueError, match="Unknown group"):
        split_train_val_cal_groups(frame, ["not-present"])
