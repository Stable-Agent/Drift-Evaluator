# Drift Detector v2 — LLM-judge constraints (fitted)


- **Dataset:** _subset_117
- **Detector:** DriftDetectorV2 (LLM-judge, fitted)
- **Generated at:** 2026-05-04T21:31:02.239010

## Overview

- **Conversations:** 117
- **Turns:** 1030
- **Drift rate (positive turns):** 32.8%

## Headline Metrics

At threshold **0.200**:

| Metric | Value |
| --- | --- |
| Precision | 0.417 |
| Recall | 0.737 |
| F1 | 0.533 |
| Accuracy | 0.576 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 249 | FN = 89 |
| **Actual negative** | FP = 348 | TN = 344 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.213 |
| ROC-AUC | 0.667 |
| PR-AUC | 0.488 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.721 |
| constraint | 0.625 |
| consistency | 0.842 |

## Severity Confusion Matrix

Accuracy: 0.519

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 435 | 58 | 9 | 190 |
| **low** | 34 | 9 | 0 | 39 |
| **medium** | 30 | 9 | 0 | 35 |
| **high** | 72 | 18 | 1 | 91 |

## Time-to-Detection

- **Detected:** 72
- **Missed:** 10
- **Early (pre-onset positive):** 63

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 0.000 |
| Mean delay | 0.264 |
| 90th-percentile delay | 1.000 |

### Delay distribution

```
0: ████████████████████████████████████████ 62
1: ██ 4
2: ██ 4
3: █ 1
4: █ 1
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.232 | 0.645 |
| constraint | 0.092 | 0.549 |
| consistency | 0.141 | 0.692 |
| total | 0.272 | 0.667 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.364 | 0.950 | 0.526 |
| 0.100 | 0.373 | 0.888 | 0.525 |
| 0.150 | 0.394 | 0.808 | 0.530 |
| **0.200** | **0.417** | **0.737** | **0.533** |
| 0.250 | 0.438 | 0.621 | 0.513 |
| 0.300 | 0.453 | 0.494 | 0.472 |
| 0.350 | 0.474 | 0.399 | 0.433 |
| 0.400 | 0.515 | 0.308 | 0.385 |
| 0.450 | 0.578 | 0.251 | 0.351 |
| 0.500 | 0.653 | 0.139 | 0.229 |
| 0.550 | 0.702 | 0.098 | 0.171 |
| 0.600 | 0.692 | 0.053 | 0.099 |
| 0.650 | 0.588 | 0.030 | 0.056 |
| 0.700 | 0.500 | 0.021 | 0.040 |
| 0.750 | 0.444 | 0.012 | 0.023 |
| 0.800 | 0.429 | 0.009 | 0.017 |
| 0.850 | 0.000 | 0.000 | 0.000 |
| 0.900 | 0.000 | 0.000 | 0.000 |
| 0.950 | 0.000 | 0.000 | 0.000 |

## Notes

Fitted threshold: 0.200
Fitted weights: {'goal_alignment': 0.4557726425713513, 'constraint_adherence': 0.17883412172459606, 'consistency': 0.36539323570405274}
Fitted severity_thresholds: {'none': 0.2605104127989405, 'low': 0.30402628595394654, 'medium': 0.3081944262809577}
