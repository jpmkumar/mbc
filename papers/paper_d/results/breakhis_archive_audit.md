# BreakHis archive audit (Paper D)

- Archive: `papers/paper_d/data/BreaKHis_v1.tar.gz`
- SHA-256: `d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640`
- Images: **7909**
- Cases: **82**
- Labels (images): {'malignant': 5429, 'benign': 2480}
- Magnifications: {'100X': 2081, '200X': 2013, '400X': 1820, '40X': 1995}
- Splitter: StratifiedGroupKFold(n_splits=5, seed=42)

| Fold | Test cases | Benign cases | Malignant cases | Test images |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | 5 | 12 | 1799 |
| 1 | 17 | 5 | 12 | 1631 |
| 2 | 16 | 5 | 11 | 1454 |
| 3 | 16 | 5 | 11 | 1528 |
| 4 | 16 | 4 | 12 | 1497 |

Case lists written under `data/splits/breakhis/fold_*/`.
