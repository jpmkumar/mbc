#!/usr/bin/env python3
"""Dump out-of-fold malignant scores for Paper A ROC / PR curves.

Reuses the confirmatory trainer (same folds, seed 42, MSE/30 VQC). Writes
`results/oof_scores.json` only — does not overwrite confirmatory_cv.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_wbcd_cv import (  # noqa: E402
    MALIGNANT,
    last_hidden_relu,
    prepare_fold,
    set_seeds,
    train_mlp_torch,
    train_qsvm,
    train_vqc,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "oof_scores.json"
SEED = 42
N_SPLITS = 5


def _sklearn_p_mal(model, x_te) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(x_te)[:, classes.index(MALIGNANT)]
    return 1.0 / (1.0 + np.exp(model.decision_function(x_te)))


def main() -> None:
    set_seeds(SEED)
    data = load_breast_cancer()
    x, y = data.data, data.target
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)
    rows: list[dict] = []

    for fold, (tr, te) in enumerate(cv.split(x, y)):
        prep = prepare_fold(x[tr], x[te], y[tr], y[te], 8, n_qsvm=4)
        y_te = prep.y_te.astype(int).tolist()

        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED + fold)
        svm.fit(prep.x_tr_s, prep.y_tr)
        rows.append(
            {
                "model": "SVM_RBF",
                "fold": fold,
                "y_true": y_te,
                "p_mal": _sklearn_p_mal(svm, prep.x_te_s).astype(float).tolist(),
            }
        )

        _, p_mlp8, _ = train_mlp_torch(
            prep.x_tr_q,
            prep.y_tr,
            prep.x_te_q,
            hidden=8,
            epochs=80,
            batch_size=32,
            lr=1e-3,
            seed=SEED + fold,
        )
        rows.append(
            {
                "model": "MLP_PCA8",
                "fold": fold,
                "y_true": y_te,
                "p_mal": np.asarray(p_mlp8, dtype=float).tolist(),
            }
        )

        _, p_vqc, _ = train_vqc(
            prep.x_tr_q,
            prep.y_tr,
            prep.x_te_q,
            n_qubits=8,
            n_layers=2,
            epochs=30,
            batch_size=32,
            lr=0.05,
            seed=SEED + fold,
            loss_name="mse",
        )
        rows.append(
            {
                "model": "VQC_8",
                "fold": fold,
                "y_true": y_te,
                "p_mal": np.asarray(p_vqc, dtype=float).tolist(),
            }
        )

        _, p_qsvm, _ = train_qsvm(
            prep.x_tr_q4,
            prep.y_tr,
            prep.x_te_q4,
            n_qubits=4,
            reps=2,
            c=10.0,
            seed=SEED + fold,
        )
        rows.append(
            {
                "model": "QSVM_ZZ4",
                "fold": fold,
                "y_true": y_te,
                "p_mal": np.asarray(p_qsvm, dtype=float).tolist(),
            }
        )

        extractor = MLPClassifier(
            hidden_layer_sizes=(32, 6),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=SEED + fold,
            early_stopping=True,
            validation_fraction=0.15,
        )
        extractor.fit(prep.x_tr_s, prep.y_tr)
        h_tr = last_hidden_relu(extractor, prep.x_tr_s)
        h_te = last_hidden_relu(extractor, prep.x_te_s)
        mm6 = MinMaxScaler(feature_range=(0.0, np.pi)).fit(h_tr)
        _, p_hyb, _ = train_vqc(
            mm6.transform(h_tr),
            prep.y_tr,
            mm6.transform(h_te),
            n_qubits=6,
            n_layers=2,
            epochs=30,
            batch_size=32,
            lr=0.05,
            seed=SEED + fold,
            loss_name="mse",
        )
        rows.append(
            {
                "model": "HYBRID_MLP_VQC",
                "fold": fold,
                "y_true": y_te,
                "p_mal": np.asarray(p_hyb, dtype=float).tolist(),
            }
        )
        print(f"fold {fold} done  n_te={len(te)}")

    payload = {
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper": "paper_a_tabular_wbcd",
        "protocol": "same folds as confirmatory (seed 42, 1x5)",
        "positive_class": "malignant (label 0)",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
