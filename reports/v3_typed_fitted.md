# Drift Detector v3 — Typed Validators (fitted)


- **Dataset:** v1_synthetic
- **Detector:** DriftDetectorV2(constraint_mode=typed) — fitted
- **Generated at:** 2026-05-06T07:21:49.422310

## Overview

- **Conversations:** 200
- **Turns:** 1776
- **Drift rate (positive turns):** 25.8%

## Headline Metrics

At threshold **0.150**:

| Metric | Value |
| --- | --- |
| Precision | 0.324 |
| Recall | 0.813 |
| F1 | 0.463 |
| Accuracy | 0.512 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 373 | FN = 86 |
| **Actual negative** | FP = 780 | TN = 537 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.182 |
| ROC-AUC | 0.666 |
| PR-AUC | 0.414 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.754 |
| constraint | 0.815 |
| consistency | 0.902 |

## Severity Confusion Matrix

Accuracy: 0.543

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 836 | 134 | 7 | 340 |
| **low** | 46 | 17 | 2 | 52 |
| **medium** | 41 | 15 | 1 | 50 |
| **high** | 95 | 27 | 2 | 111 |

## Time-to-Detection

- **Detected:** 133
- **Missed:** 11
- **Early (pre-onset positive):** 122

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.271 |
| 90th-percentile delay | 1.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 112
1: ████ 13
2: █ 5
3: █ 2
4:  0
5:  0
6:  0
7: █ 1
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.211 | 0.641 |
| constraint | 0.042 | 0.532 |
| consistency | 0.191 | 0.644 |
| total | 0.269 | 0.666 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.291 | 0.941 | 0.444 |
| 0.100 | 0.303 | 0.878 | 0.451 |
| **0.150** | **0.324** | **0.813** | **0.463** |
| 0.200 | 0.340 | 0.704 | 0.459 |
| 0.250 | 0.367 | 0.601 | 0.455 |
| 0.300 | 0.382 | 0.481 | 0.426 |
| 0.350 | 0.389 | 0.370 | 0.379 |
| 0.400 | 0.422 | 0.281 | 0.337 |
| 0.450 | 0.472 | 0.205 | 0.286 |
| 0.500 | 0.652 | 0.094 | 0.164 |
| 0.550 | 0.667 | 0.092 | 0.161 |
| 0.600 | 0.714 | 0.087 | 0.155 |
| 0.650 | 0.706 | 0.078 | 0.141 |
| 0.700 | 0.688 | 0.072 | 0.130 |
| 0.750 | 0.700 | 0.061 | 0.112 |
| 0.800 | 0.710 | 0.048 | 0.090 |
| 0.850 | 0.654 | 0.037 | 0.070 |
| 0.900 | 0.625 | 0.022 | 0.042 |
| 0.950 | 0.636 | 0.015 | 0.030 |

## Notes

Fitted weights: {'goal_alignment': 0.46265875377365484, 'constraint_adherence': 0.0, 'consistency': 0.5373412462263452}
Fitted threshold: 0.150
