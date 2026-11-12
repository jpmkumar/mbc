#!/usr/bin/env python3
"""Paper C+ BreakHis confirmatory F-MLP cross-validation (random vs grouped)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT / "papers" / "paper_c" / "scripts"))
from run_idc_probe_cv import (  # noqa: E402
    case_weights,
    choose_threshold,
    fit_temperature,
    metric_bundle,
    partition_random_inner,
    sha256,
)


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def read_case_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["case_id"] for row in csv.DictReader(stream)}


def grouped_inner(
    frame: pd.DataFrame, inner_root: Path, outer_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_dir = inner_root / "folds" / f"fold_{outer_fold}"
    train_ids = read_case_ids(fold_dir / "inner_train_case_ids.csv")
    val_ids = read_case_ids(fold_dir / "inner_val_case_ids.csv")
    cal_ids = read_case_ids(fold_dir / "inner_cal_case_ids.csv")
    sets = [train_ids, val_ids, cal_ids]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"fold {outer_fold}: inner case lists overlap")
    train = frame.index[frame["case_id"].isin(train_ids)].to_numpy()
    val = frame.index[frame["case_id"].isin(val_ids)].to_numpy()
    cal = frame.index[frame["case_id"].isin(cal_ids)].to_numpy()
    return train, val, cal


def weighted_ap(y: np.ndarray, p: np.ndarray, cases: np.ndarray) -> float:
    return float(average_precision_score(y, p, sample_weight=case_weights(cases)))


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cases_val: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> ProbeMLP:
    torch.manual_seed(seed)
    model = ProbeMLP(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train.astype(np.float32)),
            torch.from_numpy(w_train.astype(np.float32)),
        ),
        batch_size=256,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    best_state = None
    best_ap = -math.inf
    bad = 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            val_prob = torch.sigmoid(model(x_val_t)).cpu().numpy()
        ap = weighted_ap(y_val, val_prob, cases_val)
        if ap > best_ap + 1e-6:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=("grouped", "random"))
    parser.add_argument(
        "--weighting",
        choices=("case-balanced", "protocol-native"),
        default="case-balanced",
    )
    parser.add_argument(
        "--inner-case-splits",
        type=Path,
        default=ROOT / "data" / "splits" / "paper_c_plus" / "breakhis",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    provenance = json.loads((args.cache / "provenance.json").read_text())
    if not provenance.get("complete") or provenance.get("cohort") != "BreakHis":
        raise SystemExit("BreakHis embedding cache incomplete or misidentified")
    embeddings = np.load(args.cache / "embeddings.npy", mmap_mode="r")
    cache_index = pd.read_csv(args.cache / "index.csv", dtype={"filepath": str, "case_id": str})
    manifest = pd.read_csv(args.manifest, dtype={"filepath": str, "case_id": str})
    if len(cache_index) != len(embeddings) or len(cache_index) != len(manifest):
        raise SystemExit("cache/manifest row counts differ")
    if set(cache_index["filepath"]) != set(manifest["filepath"]):
        raise SystemExit("cache and manifest filepath populations differ")
    if sha256(args.cache / "index.csv") != provenance["index_sha256"]:
        raise SystemExit("index sha mismatch")
    if sha256(args.cache / "embeddings.npy") != provenance["embeddings_sha256"]:
        raise SystemExit("embeddings sha mismatch")

    # Extraction writes the C+ manifest into index.csv; re-join to the locked
    # manifest so filepath population and folds stay digest-checked.
    need = [
        "filepath",
        "case_id",
        "y",
        "magnification",
        "grouped_outer_fold",
        "random_outer_fold",
    ]
    frame = cache_index[["filepath"]].merge(manifest[need], on="filepath", how="left")
    if frame[need].isna().any().any():
        raise SystemExit("manifest join incomplete")
    if list(frame["filepath"]) != list(cache_index["filepath"]):
        raise SystemExit("cache index order must match manifest join order")
    frame["_embedding_row"] = np.arange(len(frame))
    frame["label"] = frame["y"].astype(int)
    frame = frame.reset_index(drop=True)

    fold_column = (
        "grouped_outer_fold" if args.protocol == "grouped" else "random_outer_fold"
    )
    case_weighted = args.weighting == "case-balanced" or args.protocol == "grouped"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_parts: list[pd.DataFrame] = []
    fold_reports: list[dict] = []

    for fold in range(5):
        outer_test = frame.index[frame[fold_column] == fold].to_numpy()
        development = frame[frame[fold_column] != fold]
        if args.protocol == "grouped":
            train, val, cal = grouped_inner(frame, args.inner_case_splits, fold)
            allowed = set(development.index)
            if (set(train) | set(val) | set(cal)) != allowed:
                raise RuntimeError(f"fold {fold}: inner sets do not cover development")
        else:
            # partition_random_inner expects label column and filepath.
            train, val, cal = partition_random_inner(development, fold)

        row_train = frame.loc[train, "_embedding_row"].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            np.asarray(embeddings[row_train], dtype=np.float32)
        ).astype(np.float32)
        x_val = scaler.transform(
            np.asarray(
                embeddings[frame.loc[val, "_embedding_row"].to_numpy()], dtype=np.float32
            )
        ).astype(np.float32)
        x_cal = scaler.transform(
            np.asarray(
                embeddings[frame.loc[cal, "_embedding_row"].to_numpy()], dtype=np.float32
            )
        ).astype(np.float32)
        x_test = scaler.transform(
            np.asarray(
                embeddings[frame.loc[outer_test, "_embedding_row"].to_numpy()],
                dtype=np.float32,
            )
        ).astype(np.float32)

        y_train = frame.loc[train, "label"].to_numpy()
        y_val = frame.loc[val, "label"].to_numpy()
        y_cal = frame.loc[cal, "label"].to_numpy()
        if case_weighted:
            w_train = case_weights(frame.loc[train, "case_id"].to_numpy())
            w_cal = case_weights(frame.loc[cal, "case_id"].to_numpy())
            w_val_cases = frame.loc[val, "case_id"].to_numpy()
        else:
            from sklearn.utils.class_weight import compute_sample_weight

            w_train = compute_sample_weight("balanced", y_train)
            w_cal = np.ones(len(cal))
            w_val_cases = frame.loc[val, "case_id"].to_numpy()

        model = train_mlp(
            x_train,
            y_train,
            w_train,
            x_val,
            y_val,
            w_val_cases,
            seed=args.seed + fold,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
        )
        with torch.inference_mode():
            cal_logits = (
                model(torch.from_numpy(x_cal).to(device)).cpu().numpy()
            )
            val_logits = (
                model(torch.from_numpy(x_val).to(device)).cpu().numpy()
            )
            test_logits = (
                model(torch.from_numpy(x_test).to(device)).cpu().numpy()
            )
        temperature = fit_temperature(cal_logits, y_cal, w_cal)
        val_probability = expit(val_logits / temperature)
        threshold = choose_threshold(
            y_val,
            val_probability,
            case_weights(frame.loc[val, "case_id"].to_numpy()) if case_weighted else None,
        )
        test_probability = expit(test_logits / temperature)
        test_frame = frame.loc[
            outer_test, ["filepath", "case_id", "label", "magnification"]
        ].copy()
        test_frame["fold"] = fold
        test_frame["logit"] = test_logits
        test_frame["probability"] = test_probability
        test_frame["threshold"] = threshold
        oof_parts.append(test_frame)
        fold_reports.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_val": len(val),
                "n_cal": len(cal),
                "n_test": len(outer_test),
                "temperature": temperature,
                "threshold": threshold,
                "test_metrics": metric_bundle(test_frame),
            }
        )
        print(json.dumps(fold_reports[-1]), flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != len(frame) or oof["filepath"].duplicated().any():
        raise RuntimeError("OOF predictions incomplete")
    oof = oof.sort_values("filepath").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / "oof_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    if pred_path.exists() or summary_path.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    oof.to_csv(pred_path, index=False)
    summary = {
        "complete": True,
        "paper": "C+",
        "head": "F-MLP",
        "protocol": args.protocol,
        "weighting": args.weighting,
        "cache": str(args.cache),
        "seed": args.seed,
        "rows": len(oof),
        "case_identifiers": int(oof["case_id"].nunique()),
        "metrics": metric_bundle(oof),
        "folds": fold_reports,
        "predictions_sha256": sha256(pred_path),
        "cache_provenance_sha256": sha256(args.cache / "provenance.json"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"metrics": summary["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
