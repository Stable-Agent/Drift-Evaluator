# Drift Detector v2 — LLM-judge constraints (default weights)


- **Dataset:** _subset_25
- **Detector:** DriftDetectorV2 (constraint_mode=llm_judge, model=llama3.1:8b)
- **Generated at:** 2026-05-04T17:12:56.887878

## Overview

- **Conversations:** 25
- **Turns:** 233
- **Drift rate (positive turns):** 29.2%

## Headline Metrics

At threshold **0.500**:

| Metric | Value |
| --- | --- |
| Precision | 0.471 |
| Recall | 0.118 |
| F1 | 0.188 |
| Accuracy | 0.704 |

Confusion-matrix counts:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP = 8 | FN = 60 |
| **Actual negative** | FP = 9 | TN = 156 |

## Calibration

| Metric | Value |
| --- | --- |
| Brier score | 0.205 |
| ROC-AUC | 0.637 |
| PR-AUC | 0.429 |

## Per-Drift-Type Recall

| Drift type | Recall |
| --- | --- |
| goal | 0.114 |
| constraint | 0.111 |
| consistency | 0.133 |

## Severity Confusion Matrix

Accuracy: 0.433

| true \\ pred | none | low | medium | high |
| --- | --- | --- | --- | --- |
| **none** | 89 | 57 | 17 | 2 |
| **low** | 5 | 4 | 3 | 2 |
| **medium** | 5 | 3 | 4 | 1 |
| **high** | 13 | 21 | 3 | 4 |

## Time-to-Detection

- **Detected:** 6
- **Missed:** 8
- **Early (pre-onset positive):** 2

| Statistic | Value (turns) |
| --- | --- |
| Median delay | 1.500 |
| Mean delay | 1.333 |
| 90th-percentile delay | 2.500 |

### Delay distribution

```
0: ████████████████████████████████████████ 2
1: ████████████████████ 1
2: ████████████████████████████████████████ 2
3: ████████████████████ 1
```

## Per-Signal Predictive Power

| Signal | Pearson r | ROC-AUC |
| --- | --- | --- |
| goal | 0.189 | 0.624 |
| constraint | 0.151 | 0.580 |
| consistency | 0.003 | 0.670 |
| total | 0.224 | 0.637 |

## Threshold Sweep

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| **0.050** | **0.328** | **0.926** | **0.485** |
| 0.100 | 0.320 | 0.809 | 0.458 |
| 0.150 | 0.338 | 0.721 | 0.460 |
| 0.200 | 0.372 | 0.662 | 0.476 |
| 0.250 | 0.388 | 0.588 | 0.468 |
| 0.300 | 0.417 | 0.515 | 0.461 |
| 0.350 | 0.415 | 0.397 | 0.406 |
| 0.400 | 0.472 | 0.250 | 0.327 |
| 0.450 | 0.500 | 0.162 | 0.244 |
| 0.500 | 0.471 | 0.118 | 0.188 |
| 0.550 | 0.615 | 0.118 | 0.198 |
| 0.600 | 0.778 | 0.103 | 0.182 |
| 0.650 | 0.500 | 0.029 | 0.056 |
| 0.700 | 0.500 | 0.015 | 0.029 |
| 0.750 | 1.000 | 0.015 | 0.029 |
| 0.800 | 0.000 | 0.000 | 0.000 |
| 0.850 | 0.000 | 0.000 | 0.000 |
| 0.900 | 0.000 | 0.000 | 0.000 |
| 0.950 | 0.000 | 0.000 | 0.000 |

## Notes

Threshold: 0.5
Dataset: /Users/tbrady/code/Stable-Agent/Drift-Evaluator/reports/_subset_25
