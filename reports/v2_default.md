# Drift Detector v2 — Default Weights


- **Dataset:** v1_synthetic
- **Detector:** DriftDetectorV2 (default weights)
- **Generated at:** 2026-05-05T20:51:04.147726

## Overview

- **Conversations:** 200
- **Turns:** 1776
- **Drift rate (positive turns):** 25.8%

## Headline Metrics

At threshold **0.500**:

| Metric | Value |
| --- | --- |
| Precision | 0.326 |
| Recall | 0.850 |
| F1 | 0.471 |
| Accuracy | 0.507 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 390 | FN = 69 |
| **Actual negative** | FP = 806 | TN = 511 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.282 |
| ROC-AUC | 0.676 |
| PR-AUC | 0.399 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.822 |
| constraint | 0.842 |
| consistency | 0.902 |

## Severity Confusion Matrix

Accuracy: 0.107

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 4 | 54 | 794 | 465 |
| **low** | 0 | 0 | 52 | 65 |
| **medium** | 0 | 1 | 39 | 67 |
| **high** | 0 | 1 | 87 | 147 |

## Time-to-Detection

- **Detected:** 136
- **Missed:** 8
- **Early (pre-onset positive):** 131

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.176 |
| 90th-percentile delay | 1.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 118
1: ████ 14
2: █ 2
3: █ 2
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.211 | 0.641 |
| constraint | 0.097 | 0.536 |
| consistency | 0.191 | 0.644 |
| total | 0.265 | 0.676 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.258 | 1.000 | 0.411 |
| 0.100 | 0.258 | 1.000 | 0.411 |
| 0.150 | 0.259 | 1.000 | 0.411 |
| 0.200 | 0.259 | 1.000 | 0.411 |
| 0.250 | 0.259 | 1.000 | 0.412 |
| 0.300 | 0.261 | 0.996 | 0.414 |
| 0.350 | 0.263 | 0.996 | 0.416 |
| 0.400 | 0.266 | 0.996 | 0.420 |
| 0.450 | 0.305 | 0.930 | 0.460 |
| 0.500 | 0.326 | 0.850 | 0.471 |
| **0.550** | **0.346** | **0.743** | **0.472** |
| 0.600 | 0.375 | 0.608 | 0.464 |
| 0.650 | 0.387 | 0.481 | 0.429 |
| 0.700 | 0.395 | 0.349 | 0.370 |
| 0.750 | 0.420 | 0.242 | 0.307 |
| 0.800 | 0.453 | 0.146 | 0.221 |
| 0.850 | 0.690 | 0.044 | 0.082 |
| 0.900 | 0.591 | 0.028 | 0.054 |
| 0.950 | 0.455 | 0.011 | 0.021 |

## Notes

Threshold: 0.5
Dataset: /Users/tbrady/code/Stable-Agent/Drift-Evaluator/datasets/v1_synthetic
