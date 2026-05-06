# Drift Detector: v0 baseline vs v2 fitted

- **Generated at:** 2026-05-06T03:51:04Z
- **A:** v0 baseline (default config)
- **B:** v2 fitted (NLI signals)
- **A coverage:** 200 conversations / 1776 turns (drift rate 0.315)
- **B coverage:** 200 conversations / 1776 turns (drift rate 0.258)
- **Notes:** v0 = original detector with embedding-cosine signals.
v2 = NLI-based constraint & contradiction signals + calibrated goal signal,
with weights/threshold/severity refit on the same dataset.

## Headline Metrics

| Metric | v0 baseline (default config) (thr=0.30) | v2 fitted (NLI signals) (thr=0.45) | Δ (B - A) |
| --- | --- | --- | --- |
| Precision | 0.315 | 0.335 | +0.019 |
| Recall | 1.000 | 0.821 | -0.179 |
| F1 | 0.480 | 0.475 | -0.004 |
| Accuracy | 0.316 | 0.532 | +0.216 |

## Calibration

| Metric | v0 baseline (default config) | v2 fitted (NLI signals) | Δ (B - A) |
| --- | --- | --- | --- |
| Brier (lower better) | 0.294 | 0.235 | -0.059 |
| ROC-AUC | 0.693 | 0.682 | -0.011 |
| PR-AUC | 0.478 | 0.421 | -0.056 |

## Per-Drift-Type Recall

| Drift Type | v0 baseline (default config) | v2 fitted (NLI signals) | Δ (B - A) |
| --- | --- | --- | --- |
| goal | 1.000 | 0.801 | -0.199 |
| constraint | 1.000 | 0.801 | -0.199 |
| consistency | 1.000 | 0.877 | -0.123 |

## Per-Signal Predictive Power

| Signal | Pearson r (v0 baseline (default config)) | Pearson r (v2 fitted (NLI signals)) | ROC-AUC (v0 baseline (default config)) | ROC-AUC (v2 fitted (NLI signals)) |
| --- | --- | --- | --- | --- |
| goal | 0.244 | 0.211 | 0.648 | 0.641 |
| constraint | -0.032 | 0.097 | 0.492 | 0.536 |
| consistency | 0.273 | 0.191 | 0.656 | 0.644 |
| total | 0.317 | 0.286 | 0.693 | 0.682 |

## Time-to-Detection

| Metric | v0 baseline (default config) | v2 fitted (NLI signals) | Δ (B - A) |
| --- | --- | --- | --- |
| Detected | 144 | 131 | -13 |
| Missed | 0 | 13 | +13 |
| Early | 144 | 125 | -19 |
| Median delay | 0.000 | 0.000 | +0.000 |
| Mean delay | 0.000 | 0.191 | +0.191 |
| P90 delay | 0.000 | 1.000 | +1.000 |

## Severity Confusion Accuracy

| Metric | v0 baseline (default config) | v2 fitted (NLI signals) | Δ (B - A) |
| --- | --- | --- | --- |
| Accuracy | 0.160 | 0.560 | +0.400 |

## Improvement Summary

**Improvements:**

- Accuracy: 0.316 → 0.532 (+0.216) ✅ improvement
- Brier: 0.294 → 0.235 (-0.059) ✅ improvement (note: lower is better)
- Severity accuracy: 0.160 → 0.560 (+0.400) ✅ improvement

**Regressions:**

- Recall: 1.000 → 0.821 (-0.179) ❌ regression
- PR-AUC: 0.478 → 0.421 (-0.056) ❌ regression
- Recall (goal): 1.000 → 0.801 (-0.199) ❌ regression
- Recall (constraint): 1.000 → 0.801 (-0.199) ❌ regression
- Recall (consistency): 1.000 → 0.877 (-0.123) ❌ regression
