# Drift Detector Baseline Evaluation


- **Dataset:** v1_synthetic
- **Detector:** drift-detector v0 (default config)
- **Generated at:** 2026-05-04T11:52:49.926306

## Overview

- **Conversations:** 200
- **Turns:** 1776
- **Drift rate (positive turns):** 31.5%

## Headline Metrics

At threshold **0.300**:

| Metric | Value |
| --- | --- |
| Precision | 0.315 |
| Recall | 1.000 |
| F1 | 0.480 |
| Accuracy | 0.316 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 560 | FN = 0 |
| **Actual negative** | FP = 1215 | TN = 1 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.294 |
| ROC-AUC | 0.693 |
| PR-AUC | 0.478 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 1.000 |
| constraint | 1.000 |
| consistency | 1.000 |

## Severity Confusion Matrix

Accuracy: 0.160

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 0 | 1 | 228 | 987 |
| **low** | 0 | 0 | 4 | 140 |
| **medium** | 0 | 0 | 1 | 128 |
| **high** | 0 | 0 | 4 | 283 |

## Time-to-Detection

- **Detected:** 144
- **Missed:** 0
- **Early (pre-onset positive):** 144

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.000 |
| 90th-percentile delay | 0.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 144
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.244 | 0.648 |
| constraint | -0.032 | 0.492 |
| consistency | 0.273 | 0.656 |
| total | 0.317 | 0.693 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.315 | 1.000 | 0.479 |
| 0.100 | 0.315 | 1.000 | 0.479 |
| 0.150 | 0.315 | 1.000 | 0.479 |
| 0.200 | 0.315 | 1.000 | 0.479 |
| 0.250 | 0.315 | 1.000 | 0.479 |
| 0.300 | 0.315 | 1.000 | 0.480 |
| 0.350 | 0.316 | 1.000 | 0.480 |
| 0.400 | 0.320 | 1.000 | 0.485 |
| 0.450 | 0.332 | 0.995 | 0.498 |
| 0.500 | 0.358 | 0.984 | 0.525 |
| 0.550 | 0.380 | 0.946 | 0.542 |
| **0.600** | **0.410** | **0.848** | **0.553** |
| 0.650 | 0.451 | 0.604 | 0.516 |
| 0.700 | 0.455 | 0.366 | 0.406 |
| 0.750 | 0.510 | 0.184 | 0.270 |
| 0.800 | 0.621 | 0.073 | 0.131 |
| 0.850 | 0.900 | 0.016 | 0.032 |
| 0.900 | 0.000 | 0.000 | 0.000 |
| 0.950 | 0.000 | 0.000 | 0.000 |

## Notes

Threshold: 0.3
Detector: drift-detector v0 (default weights)
Dataset: /Users/tbrady/code/Stable-Agent/Drift-Evaluator/datasets/v1_synthetic
