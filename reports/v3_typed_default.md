# Drift Detector v3 — Typed Validators (default weights)


- **Dataset:** v1_synthetic
- **Detector:** DriftDetectorV2(constraint_mode=typed)
- **Generated at:** 2026-05-06T07:21:49.382862

## Overview

- **Conversations:** 200
- **Turns:** 1776
- **Drift rate (positive turns):** 25.8%

## Headline Metrics

At threshold **0.400**:

| Metric | Value |
| --- | --- |
| Precision | 0.354 |
| Recall | 0.479 |
| F1 | 0.407 |
| Accuracy | 0.640 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 220 | FN = 239 |
| **Actual negative** | FP = 401 | TN = 916 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.203 |
| ROC-AUC | 0.641 |
| PR-AUC | 0.342 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.340 |
| constraint | 0.548 |
| consistency | 0.615 |

## Severity Confusion Matrix

Accuracy: 0.286

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 387 | 529 | 270 | 131 |
| **low** | 15 | 50 | 37 | 15 |
| **medium** | 16 | 38 | 34 | 19 |
| **high** | 25 | 95 | 78 | 37 |

## Time-to-Detection

- **Detected:** 97
- **Missed:** 47
- **Early (pre-onset positive):** 76

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.660 |
| 90th-percentile delay | 2.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 63
1: ██████████ 17
2: ████████ 13
3: █ 1
4: █ 1
5: █ 1
6:  0
7:  0
8:  0
9: █ 1
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.211 | 0.641 |
| constraint | 0.042 | 0.532 |
| consistency | 0.191 | 0.644 |
| total | 0.196 | 0.641 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.267 | 0.991 | 0.420 |
| 0.100 | 0.268 | 0.976 | 0.420 |
| 0.150 | 0.285 | 0.928 | 0.436 |
| 0.200 | 0.302 | 0.878 | 0.450 |
| **0.250** | **0.325** | **0.813** | **0.464** |
| 0.300 | 0.339 | 0.708 | 0.458 |
| 0.350 | 0.355 | 0.595 | 0.445 |
| 0.400 | 0.354 | 0.479 | 0.407 |
| 0.450 | 0.352 | 0.377 | 0.364 |
| 0.500 | 0.368 | 0.320 | 0.343 |
| 0.550 | 0.355 | 0.190 | 0.247 |
| 0.600 | 0.351 | 0.155 | 0.215 |
| 0.650 | 0.346 | 0.120 | 0.178 |
| 0.700 | 0.340 | 0.076 | 0.125 |
| 0.750 | 0.286 | 0.035 | 0.062 |
| 0.800 | 0.273 | 0.013 | 0.025 |
| 0.850 | 0.500 | 0.004 | 0.009 |
| 0.900 | 0.000 | 0.000 | 0.000 |
| 0.950 | 0.000 | 0.000 | 0.000 |

## Notes

Threshold: 0.4
Per-template typed validators routed by template_validators.py
