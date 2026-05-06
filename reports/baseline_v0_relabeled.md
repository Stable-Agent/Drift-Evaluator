# Drift Detector Baseline Evaluation


- **Dataset:** v1_synthetic
- **Detector:** drift-detector v0 (default config)
- **Generated at:** 2026-05-05T20:41:57.633943

## Overview

- **Conversations:** 200
- **Turns:** 1776
- **Drift rate (positive turns):** 25.8%

## Headline Metrics

At threshold **0.300**:

| Metric | Value |
| --- | --- |
| Precision | 0.259 |
| Recall | 1.000 |
| F1 | 0.411 |
| Accuracy | 0.259 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 459 | FN = 0 |
| **Actual negative** | FP = 1316 | TN = 1 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.313 |
| ROC-AUC | 0.686 |
| PR-AUC | 0.393 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 1.000 |
| constraint | 1.000 |
| consistency | 1.000 |

## Severity Confusion Matrix

Accuracy: 0.132

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 0 | 1 | 230 | 1086 |
| **low** | 0 | 0 | 2 | 115 |
| **medium** | 0 | 0 | 1 | 106 |
| **high** | 0 | 0 | 1 | 234 |

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
| goal | 0.218 | 0.642 |
| constraint | -0.024 | 0.493 |
| consistency | 0.260 | 0.663 |
| total | 0.287 | 0.686 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.258 | 1.000 | 0.411 |
| 0.100 | 0.258 | 1.000 | 0.411 |
| 0.150 | 0.258 | 1.000 | 0.411 |
| 0.200 | 0.258 | 1.000 | 0.411 |
| 0.250 | 0.258 | 1.000 | 0.411 |
| 0.300 | 0.259 | 1.000 | 0.411 |
| 0.350 | 0.259 | 1.000 | 0.411 |
| 0.400 | 0.262 | 1.000 | 0.416 |
| 0.450 | 0.273 | 0.998 | 0.428 |
| 0.500 | 0.295 | 0.991 | 0.455 |
| 0.550 | 0.315 | 0.959 | 0.474 |
| **0.600** | **0.341** | **0.858** | **0.488** |
| 0.650 | 0.376 | 0.619 | 0.467 |
| 0.700 | 0.373 | 0.373 | 0.373 |
| 0.750 | 0.407 | 0.181 | 0.250 |
| 0.800 | 0.458 | 0.059 | 0.104 |
| 0.850 | 0.909 | 0.022 | 0.043 |
| 0.900 | 0.000 | 0.000 | 0.000 |
| 0.950 | 0.000 | 0.000 | 0.000 |

## Notes

Threshold: 0.3
Detector: drift-detector v0 (default weights)
Dataset: /Users/tbrady/code/Stable-Agent/Drift-Evaluator/datasets/v1_synthetic
