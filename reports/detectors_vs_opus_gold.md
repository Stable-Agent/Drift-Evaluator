# Detector evaluation vs Opus gold (`gold_v1_opus`)

**Generated:** 2026-05-07
**Gold set:** `gold/gold_v1_opus.jsonl` — 508 turns from 179 conversations,
labeled by Opus 4.7 against `docs/gold_set_rubric.md` under the blind-view
protocol (no card label, no metadata, no prior-judge label visible at
labeling time).

This report scores the existing detector configurations against the Opus
labels and contrasts them with the same detectors' performance against the
synthetic dataset's card labels.

---

## 1. Opus label distribution (n=508)

| Quantity | Value |
|---|---:|
| Drifted | 128 (25.2%) |
| Clean | 380 (74.8%) |
| Severity: low / medium / high | 29 / 50 / 49 |
| Multi-label drift types: goal / constraint / consistency | 47 / 59 / 38 |
| Constraint indices violated: 0 / 1 / 2 | 30 / 10 / 26 |
| Contaminated | 0 |
| Parse errors | 0 |

Opus's per-turn drift rate (25.2%) is roughly half the rate the card
metadata implies for the same 508 turns (44.9%). The gap is the rubric
working as designed: clarifying-question turns where a "drift onset"
constraint hasn't actually fired yet are correctly labeled as not-drifted,
even though the card says drift was supposed to start.

---

## 2. Opus vs card labels (binary `drifted`)

|  | opus = drift | opus = clean |
|---|---:|---:|
| **card = drift** | 121 (TP) | 165 (FN) |
| **card = clean** | 7 (FP) | 215 (TN) |

- Precision (opus relative to card): 0.945
- Recall (opus relative to card): 0.423
- F1: 0.585
- Cohen's κ: 0.363

When both flag drift the type alignment is strong:
`goal_drift → goal` (34×), `constraint_violation → constraint` (27×),
`consistency_break → consistency` (25×). Most multi-type opus labels are
supersets of the card type rather than disagreements.

Per-template post-onset recall (opus agrees with card-says-drift at this turn):

| Template | Hit / Card-onset turns | Recall |
|---|---:|---:|
| Cooking (ingredients) | 27 / 39 | 69.2% |
| PR review | 18 / 32 | 56.2% |
| Email drafting | 17 / 39 | 43.6% |
| Road trip planning | 12 / 33 | 36.4% |
| Fitness routine | 12 / 34 | 35.3% |
| Children's story | 13 / 38 | 34.2% |
| SQL teaching | 11 / 34 | 32.4% |
| Quarterly financial report | 11 / 37 | 29.7% |

The templates with the largest recall gap (Financial, SQL, Story) are the
ones where the generator most often plants drift on a clarifying-question
turn rather than on the next substantive turn — exactly the cases the
rubric's gated-constraint logic deliberately excludes.

---

## 3. Detector results (binary, default thresholds)

| Detector | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_v0 (DriftDetector) | 0.30 | 0.252 | 1.000 | 0.403 | 0.647 | 0.372 | 0.339 | 128 / 380 / 0 / 0 |
| v2_default (DriftDetectorV2) | 0.30 | 0.255 | 1.000 | 0.406 | 0.682 | 0.379 | 0.302 | 128 / 374 / 0 / 6 |
| v2_fitted (saved config) | 0.45 | 0.322 | 0.922 | **0.477** | 0.697 | 0.415 | 0.246 | 118 / 249 / 10 / 131 |
| **v3_typed_default** (typed validators) | 0.40 | **0.357** | 0.578 | 0.442 | 0.661 | 0.343 | **0.206** | 74 / 133 / 54 / 247 |
| v3_typed_fitted (refit on rows) | 0.15 | 0.313 | 0.914 | 0.466 | **0.700** | 0.415 | 0.175 | 117 / 257 / 11 / 123 |

### Threshold sweep

`v2_default` score against Opus:

| threshold | precision | recall | F1 | accuracy |
|---:|---:|---:|---:|---:|
| 0.30 (default) | 0.255 | 1.000 | 0.406 | 0.264 |
| 0.40 | 0.258 | 1.000 | 0.410 | 0.274 |
| 0.50 | 0.307 | 0.922 | 0.461 | 0.457 |
| 0.60 | 0.355 | 0.711 | **0.474** | 0.602 |
| 0.70 | 0.359 | 0.406 | 0.381 | 0.667 |

`baseline_v0`:

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.30–0.40 | 0.25 | 1.00 | 0.40 |
| 0.50 | 0.263 | 0.977 | 0.414 |
| 0.60 | 0.307 | 0.898 | **0.457** |
| 0.70 | 0.348 | 0.430 | 0.385 |

`v3_typed_default`:

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.292 | 0.945 | 0.446 |
| 0.30 | 0.341 | 0.812 | **0.480** |
| 0.40 (default) | 0.357 | 0.578 | 0.442 |
| 0.50 | 0.344 | 0.414 | 0.376 |

`v3_typed_fitted`:

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.292 | 0.961 | 0.448 |
| 0.20 | 0.344 | 0.828 | **0.486** |
| 0.30 | 0.382 | 0.609 | 0.470 |

---

## 4. Findings

1. **Raw discrimination is similar across all four detectors** (ROC-AUC
   0.647 → 0.700). The headline performance gap is *operating-point
   calibration*, not signal quality. `baseline_v0` and `v2_default` are
   effectively saturated at the published threshold of 0.30: they fire on
   >98% of turns and their TP advantage is just "always say yes."

2. **`v3_typed_default` is the only detector calibrated at a sensible
   operating point out of the box.** Highest precision (0.357), 247 true
   negatives, best Brier (0.206). The per-template typed validators
   genuinely fire more selectively than rule-based string checks or
   generic NLI.

3. **The fitted configs are not overfitting the gold set indirectly.**
   `v2_fitted` and `v3_typed_fitted` were fit against the *card* labels;
   their F1 against Opus (0.477 and 0.466) is essentially the same as
   their fit-time F1 against the cards (0.475 and 0.463). Whatever signal
   the fitting found generalizes to the rubric.

4. **`v3_typed_fitted` zeroed out the constraint signal**
   (weights `goal=0.46, constraint=0.00, consistency=0.54`). Against the
   noisy card labels the typed-validator constraint signal looked
   unhelpful at fit-time. The fact that `v3_typed_default` (which keeps
   the constraint signal) has the *best precision and lowest Brier* against
   Opus suggests the typed-validator constraint signal carries real
   information that the card-label fit threw away. Refitting on Opus
   directly would test this — but that's the gold-set-as-dev-set
   experiment §10 of the rubric explicitly warns against and so should be
   reported as diagnostic only, not as a config to ship.

5. **At any sensible threshold no detector exceeds F1 ≈ 0.49 against Opus.**
   The five best-F1 operating points cluster between 0.457 and 0.486 — a
   narrow band given the spread in default-threshold F1 (0.40 → 0.48).
   Detector-vs-detector ranking on Opus is real but small; the rubric's
   guidance to attach bootstrap CIs to any "Phase X improves Y" claim
   applies.

---

## 5. Methodological notes

**Re-scoring vs the existing reports.** The existing reports in
`reports/baseline_v0.json`, `reports/v2_default.json`,
`reports/v2_fitted.json`, and `reports/v3_typed_*.json` score against
*card* labels over all 1776 turns of `v1_synthetic`. This report scores
the same detectors over the **508 Opus-labeled turns only**, with truth
columns replaced by Opus's labels. Detector predictions are unchanged;
only the truth join differs.

- `v2_default`: re-scored from the saved `reports/v2_default_rows.jsonl`.
- `v2_fitted`: re-scored by reapplying `reports/fitted_config.json` to
  the same row file.
- `baseline_v0` and `v3_typed`: detectors re-run end-to-end over
  `datasets/v1_synthetic`, predictions saved to
  `reports/baseline_v0_rows.jsonl` and `reports/v3_typed_rows.jsonl`,
  then scored against Opus.

**Why F1 differs from `comparison_v0_vs_v2_fitted.md`.** That comparison
report scores against the card labels over the full 1776-turn dataset.
The numbers there should not be compared directly to the F1 figures in
§3 of this report — different ground truth, different sample size, different
selection bias.

**On `v3_typed_fitted`'s threshold of 0.15.** The fit was run against
card labels with the v3 signal distribution. The fitted threshold sits in
a region where precision is poor on Opus (0.31). At threshold 0.20 the
same scores deliver F1 0.486, suggesting the fit-time threshold is
slightly off-target relative to the rubric. Retuning the threshold on
Opus (without retuning weights) recovers most of the available F1 — but
this is itself a gold-set tuning step and so the §10 caveat applies.
