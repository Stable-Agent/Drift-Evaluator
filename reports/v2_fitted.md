# Drift Detector v2 — Fitted Weights & Threshold


- **Dataset:** v1_synthetic
- **Detector:** DriftDetectorV2 (fitted)
- **Generated at:** 2026-05-05T20:51:04.190261

## Overview

- **Conversations:** 200
- **Turns:** 1776
- **Drift rate (positive turns):** 25.8%

## Headline Metrics

At threshold **0.450**:

| Metric | Value |
| --- | --- |
| Precision | 0.335 |
| Recall | 0.821 |
| F1 | 0.475 |
| Accuracy | 0.532 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 377 | FN = 82 |
| **Actual negative** | FP = 750 | TN = 567 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.235 |
| ROC-AUC | 0.682 |
| PR-AUC | 0.421 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.801 |
| constraint | 0.801 |
| consistency | 0.877 |

## Severity Confusion Matrix

Accuracy: 0.560

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 867 | 109 | 3 | 338 |
| **low** | 51 | 12 | 0 | 54 |
| **medium** | 40 | 16 | 0 | 51 |
| **high** | 91 | 28 | 1 | 115 |

## Time-to-Detection

- **Detected:** 131
- **Missed:** 13
- **Early (pre-onset positive):** 125

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.191 |
| 90th-percentile delay | 1.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 113
1: ████ 13
2: █ 3
3: █ 2
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.211 | 0.641 |
| constraint | 0.097 | 0.536 |
| consistency | 0.191 | 0.644 |
| total | 0.286 | 0.682 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.258 | 1.000 | 0.411 |
| 0.100 | 0.258 | 1.000 | 0.411 |
| 0.150 | 0.259 | 1.000 | 0.411 |
| 0.200 | 0.259 | 1.000 | 0.411 |
| 0.250 | 0.261 | 0.996 | 0.413 |
| 0.300 | 0.262 | 0.993 | 0.415 |
| 0.350 | 0.267 | 0.991 | 0.420 |
| 0.400 | 0.305 | 0.932 | 0.460 |
| **0.450** | **0.335** | **0.821** | **0.475** |
| 0.500 | 0.366 | 0.651 | 0.469 |
| 0.550 | 0.394 | 0.479 | 0.433 |
| 0.600 | 0.409 | 0.309 | 0.352 |
| 0.650 | 0.481 | 0.190 | 0.272 |
| 0.700 | 0.651 | 0.089 | 0.157 |
| 0.750 | 0.727 | 0.087 | 0.156 |
| 0.800 | 0.689 | 0.068 | 0.123 |
| 0.850 | 0.684 | 0.057 | 0.105 |
| 0.900 | 0.654 | 0.037 | 0.070 |
| 0.950 | 0.600 | 0.013 | 0.026 |

## Notes

Fitted threshold: 0.450
Fitted weights: {'goal_alignment': 0.2833305961534779, 'constraint_adherence': 0.3700389374125166, 'consistency': 0.3466304664340055}
Fitted severity_thresholds: {'none': 0.5144509128504171, 'low': 0.549449023388252, 'medium': 0.5505817329474165}
