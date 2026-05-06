# Drift Detector v2 — LLM-judge constraints (default weights)


- **Dataset:** _subset_117
- **Detector:** DriftDetectorV2 (constraint_mode=llm_judge, model=llama3.1:8b)
- **Generated at:** 2026-05-04T21:31:02.195805

## Overview

- **Conversations:** 117
- **Turns:** 1030
- **Drift rate (positive turns):** 32.8%

## Headline Metrics

At threshold **0.500**:

| Metric | Value |
| --- | --- |
| Precision | 0.571 |
| Recall | 0.166 |
| F1 | 0.257 |
| Accuracy | 0.685 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 56 | FN = 282 |
| **Actual negative** | FP = 42 | TN = 650 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.216 |
| ROC-AUC | 0.651 |
| PR-AUC | 0.472 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.103 |
| constraint | 0.182 |
| consistency | 0.228 |

## Severity Confusion Matrix

Accuracy: 0.414

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 354 | 231 | 98 | 9 |
| **low** | 23 | 31 | 23 | 5 |
| **medium** | 26 | 21 | 22 | 5 |
| **high** | 48 | 81 | 34 | 19 |

## Time-to-Detection

- **Detected:** 32
- **Missed:** 50
- **Early (pre-onset positive):** 9

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 1.000 |
| Mean delay | 1.312 |
| 90th-percentile delay | 3.900 |

### Delay distribution

```
0: ████████████████████████████████████████ 14
1: ████████████████████ 7
2: ██████████████ 5
3: █████ 2
4: ████████ 3
5:  0
6:  0
7: ██ 1
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.232 | 0.645 |
| constraint | 0.092 | 0.549 |
| consistency | 0.141 | 0.692 |
| total | 0.250 | 0.651 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.050 | 0.368 | 0.950 | 0.530 |
| 0.100 | 0.374 | 0.888 | 0.526 |
| **0.150** | **0.396** | **0.808** | **0.532** |
| 0.200 | 0.416 | 0.713 | 0.526 |
| 0.250 | 0.407 | 0.592 | 0.483 |
| 0.300 | 0.419 | 0.491 | 0.452 |
| 0.350 | 0.447 | 0.402 | 0.424 |
| 0.400 | 0.502 | 0.320 | 0.391 |
| 0.450 | 0.530 | 0.207 | 0.298 |
| 0.500 | 0.571 | 0.166 | 0.257 |
| 0.550 | 0.639 | 0.115 | 0.195 |
| 0.600 | 0.763 | 0.086 | 0.154 |
| 0.650 | 0.722 | 0.038 | 0.073 |
| 0.700 | 0.571 | 0.012 | 0.023 |
| 0.750 | 0.800 | 0.012 | 0.023 |
| 0.800 | 0.667 | 0.006 | 0.012 |
| 0.850 | 0.000 | 0.000 | 0.000 |
| 0.900 | 0.000 | 0.000 | 0.000 |
| 0.950 | 0.000 | 0.000 | 0.000 |

## Notes

Threshold: 0.5
Dataset: /Users/tbrady/code/Stable-Agent/Drift-Evaluator/reports/_subset_117
