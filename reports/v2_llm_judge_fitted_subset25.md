# Drift Detector v2 — LLM-judge constraints (fitted)


- **Dataset:** _subset_25
- **Detector:** DriftDetectorV2 (LLM-judge, fitted)
- **Generated at:** 2026-05-04T17:12:56.900577

## Overview

- **Conversations:** 25
- **Turns:** 233
- **Drift rate (positive turns):** 29.2%

## Headline Metrics

At threshold **0.050**:

| Metric | Value |
| --- | --- |
| Precision | 0.330 |
| Recall | 0.941 |
| F1 | 0.489 |
| Accuracy | 0.425 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 64 | FN = 4 |
| **Actual negative** | FP = 130 | TN = 35 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.208 |
| ROC-AUC | 0.637 |
| PR-AUC | 0.431 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.943 |
| constraint | 0.889 |
| consistency | 1.000 |

## Severity Confusion Matrix

Accuracy: 0.536

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 102 | 1 | 3 | 59 |
| **low** | 6 | 0 | 0 | 8 |
| **medium** | 5 | 0 | 0 | 8 |
| **high** | 18 | 0 | 0 | 23 |

## Time-to-Detection

- **Detected:** 14
- **Missed:** 0
- **Early (pre-onset positive):** 13

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.071 |
| 90th-percentile delay | 0.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 13
1: ███ 1
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.189 | 0.624 |
| constraint | 0.151 | 0.580 |
| consistency | 0.003 | 0.670 |
| total | 0.232 | 0.637 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| **0.050** | **0.330** | **0.941** | **0.489** |
| 0.100 | 0.328 | 0.868 | 0.476 |
| 0.150 | 0.325 | 0.794 | 0.462 |
| 0.200 | 0.343 | 0.721 | 0.464 |
| 0.250 | 0.365 | 0.676 | 0.474 |
| 0.300 | 0.368 | 0.618 | 0.462 |
| 0.350 | 0.398 | 0.574 | 0.470 |
| 0.400 | 0.412 | 0.515 | 0.458 |
| 0.450 | 0.443 | 0.397 | 0.419 |
| 0.500 | 0.395 | 0.250 | 0.306 |
| 0.550 | 0.474 | 0.132 | 0.207 |
| 0.600 | 0.600 | 0.132 | 0.217 |
| 0.650 | 0.538 | 0.103 | 0.173 |
| 0.700 | 0.750 | 0.088 | 0.158 |
| 0.750 | 0.857 | 0.088 | 0.160 |
| 0.800 | 0.800 | 0.059 | 0.110 |
| 0.850 | 0.500 | 0.015 | 0.029 |
| 0.900 | 0.500 | 0.015 | 0.029 |
| 0.950 | 1.000 | 0.015 | 0.029 |

## Notes

Fitted threshold: 0.050
Fitted weights: {'goal_alignment': 0.5491161550485848, 'constraint_adherence': 0.4508838449514151, 'consistency': 0.0}
Fitted severity_thresholds: {'none': 0.32978049808858895, 'low': 0.33541846803850167, 'medium': 0.3559925946987622}
