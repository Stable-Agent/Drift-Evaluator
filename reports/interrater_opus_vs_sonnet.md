# Inter-rater agreement: Opus 4.7 vs Sonnet 4.6 (`gold_v1`)

Both raters apply `docs/gold_set_rubric.md` under the blind-view protocol — same prompt, same hidden metadata, same JSON schema. Only the model differs. The numbers below are the ceiling that any detector can reach against a single rater: a detector that matches Opus better than Sonnet does is at the inter-rater ceiling, not under-fitted.

## 1. Sample coverage

- Joined rows (both raters labeled successfully): **508**
- Labeled by Opus only (Sonnet missing or parse_error): 0
- Labeled by Sonnet only (Opus missing or parse_error): 0
- Parse errors — Opus: 0, Sonnet: 0

## 2. Binary drift agreement

- Cohen's κ: **0.724**
- Raw agreement: 449/508 = 88.4%
- Opus drift rate: 25.2%; Sonnet drift rate: 34.1%

Confusion matrix (rows=Opus, cols=Sonnet):

|  | sonnet=drift | sonnet=clean |
|---|---:|---:|
| **opus=drift** | 121 | 7 |
| **opus=clean** | 52 | 328 |

## 3. Severity agreement

- κ (unweighted, 4 levels): **0.526**
- κ (linear-weighted, ordinal): 0.655
- κ (quadratic-weighted, ordinal): **0.741**  ← preferred for ordinal severity

Confusion matrix (rows=Opus, cols=Sonnet):

| Opus \ Sonnet | none | low | medium | high |
|---|---:|---:|---:|---:|
| **none** | 328 | 23 | 22 | 7 |
| **low** | 1 | 12 | 10 | 6 |
| **medium** | 4 | 10 | 23 | 13 |
| **high** | 2 | 2 | 15 | 30 |

**Mapped to corrector strategy tiers** (none→skip, low→gentle, medium→structured, high→emergency):
- Tier-level κ: **0.526**
- Raw agreement: 393/508 = 77.4%
- Interpretation: tier disagreement bounds how stable the corrector's escalation behavior can be, regardless of detector quality.

## 4. Per-type agreement (multi-label)

Each row's drift_types is a multi-label set; we compute κ on each type independently as a binary indicator.

| Type | Opus + | Sonnet + | Both + | κ |
|---|---:|---:|---:|---:|
| goal | 47 | 58 | 39 | 0.714 |
| constraint | 59 | 112 | 54 | 0.565 |
| consistency | 38 | 55 | 36 | 0.752 |

## 5. Violated-constraint agreement

Drives the corrector's Structured-Anchor callout text — disagreement here means the two raters would point the agent at different constraints to recommit to.

- Exact-set match: 431/508 = 84.8%
- Mean Jaccard (all rows, with empty=empty=1.0): **0.857**
- Mean Jaccard (rows where ≥1 rater flagged any constraint, n=118): **0.386**

## 6. Per-template binary-drift agreement

| Template | n | Opus drift | Sonnet drift | Agree | κ |
|---|---:|---:|---:|---:|---:|
| childrens_story_writer | 66 | 14 | 18 | 90.9% | 0.754 |
| code_review_bot | 60 | 18 | 25 | 85.0% | 0.679 |
| cooking_recipe_assistant | 67 | 30 | 33 | 95.5% | 0.910 |
| email_drafter | 67 | 17 | 15 | 94.0% | 0.836 |
| financial_report_analyst | 65 | 11 | 14 | 95.4% | 0.852 |
| fitness_coach | 62 | 13 | 15 | 93.5% | 0.816 |
| sql_tutor | 62 | 13 | 40 | 53.2% | 0.199 |
| travel_planner | 59 | 12 | 13 | 98.3% | 0.949 |

## 7. How to read this against detector results

Re-score the detectors against `gold_v1_sonnet.jsonl` with the same scoring code that produced `reports/detectors_vs_opus_gold.md`. Three diagnostic patterns:

1. **detector-vs-Opus ≈ detector-vs-Sonnet ≈ Opus-vs-Sonnet** — you've hit the inter-rater ceiling. More fitting will not help; ship the simplest config that gets there.
2. **detector-vs-Opus < Opus-vs-Sonnet** — there is real headroom. A held-out refit (per the design in this conversation) is worth running.
3. **Opus-vs-Sonnet itself is low** — the rubric is the bottleneck, not the detector. Firm up the rubric (or the prompt that conveys it) before any further detector work.
