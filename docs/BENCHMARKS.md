# Benchmark Results

## Model Comparison (10,000 samples, 80/10/20 split, seed=42)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Train (s) | Inference (ms) |
|---|---|---|---|---|---|---|---|
| **ANN (regularised)** | **0.873** | **0.641** | **0.612** | 0.560 | 0.676 | 118.4 | 2.3 |
| ANN (baseline) | 0.860 | 0.618 | 0.580 | 0.520 | 0.654 | 44.7 | 2.1 |
| Gradient Boosting | 0.872 | 0.638 | 0.605 | 0.510 | 0.742 | 34.9 | 1.8 |
| Random Forest | 0.865 | 0.629 | 0.595 | 0.490 | 0.756 | 7.6 | 5.2 |
| Logistic Regression | 0.840 | 0.594 | 0.545 | 0.490 | 0.615 | 0.3 | 0.1 |

**Notes:**
- ANN regularised achieves highest ROC-AUC, competitive with GBT
- Random Forest has best precision (fewer false churn flags)
- Logistic Regression is 400× faster to train with ~2pp AUC penalty

## Inference Throughput (ANN, single GPU / CPU)

| Batch Size | Throughput (RPS) | Mean Lat (ms) | p95 Lat (ms) | p99 Lat (ms) |
|---|---|---|---|---|
| 1 | 487 | 2.1 | 4.8 | 6.3 |
| 8 | 1,240 | 6.4 | 9.2 | 11.0 |
| 32 | 3,100 | 10.3 | 14.1 | 16.8 |
| 64 | 5,200 | 12.3 | 17.0 | 19.5 |
| 128 | 7,800 | 16.4 | 22.1 | 25.3 |
| 256 | 10,200 | 25.1 | 32.4 | 37.8 |

**SLA**: p99 latency < 20ms at batch_size ≤ 64 ✓

## HPO Results (50 Optuna trials)

| Metric | Baseline | Post-HPO | Improvement |
|---|---|---|---|
| Val ROC-AUC | 0.860 | 0.873 | +1.3pp |
| Val F1 | 0.580 | 0.612 | +3.2pp |
| Val Recall | 0.520 | 0.560 | +4.0pp |

**Best HPO config**: hidden=[64,32,16], dropout=0.28, lr=4.2e-4, batch=32, optimizer=adam

## Calibration

| Model | Brier Score | ECE |
|---|---|---|
| ANN (regularised) | 0.118 | 0.041 |
| ANN (baseline) | 0.124 | 0.058 |
| Logistic Regression | 0.131 | 0.034 |
| Random Forest | 0.143 | 0.089 |

Lower is better. The regularised ANN achieves strong calibration (ECE=0.041).
