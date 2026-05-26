# Inter-rater agreement on derived label (`goal_failure OR consistency_break`)

Joined samples: **508**. Source files: `gold/gold_v1_opus.jsonl`, `gold/gold_v1_sonnet.jsonl`.

## 1. Headline κ comparison

| Signal | Opus + | Sonnet + | Raw agreement | Cohen's κ |
|---|---:|---:|---:|---:|
| Full `drifted` (goal+constraint+consistency) | 128 | 173 | 88.4% | **0.724** |
| Derived: goal OR consistency | 84 | 102 | 93.7% | **0.790** |
| Δκ (derived − full) | | | | **+0.066** |

Confusion (rows=Opus derived, cols=Sonnet derived):

|  | sonnet=drift | sonnet=clean |
|---|---:|---:|
| **opus=drift** | 77 | 7 |
| **opus=clean** | 25 | 399 |

## 2. Per-template breakdown

| Template | n | Opus + (full→derived) | Sonnet + (full→derived) | Agree (full→derived) | κ full | κ derived | Δκ |
|---|---:|---:|---:|---:|---:|---:|---:|
| childrens_story_writer | 66 | 14→10 | 18→13 | 90.9%→92.4% | 0.754 | 0.738 | -0.016 |
| code_review_bot | 60 | 18→15 | 25→22 | 85.0%→88.3% | 0.679 | 0.731 | +0.052 |
| cooking_recipe_assistant | 67 | 30→18 | 33→21 | 95.5%→95.5% | 0.910 | 0.892 | -0.019 |
| email_drafter | 67 | 17→12 | 15→12 | 94.0%→94.0% | 0.836 | 0.797 | -0.039 |
| financial_report_analyst | 65 | 11→7 | 14→4 | 95.4%→95.4% | 0.852 | 0.704 | -0.148 |
| fitness_coach | 62 | 13→11 | 15→10 | 93.5%→98.4% | 0.816 | 0.943 | +0.127 |
| sql_tutor | 62 | 13→4 | 40→7 | 53.2%→95.2% | 0.199 | 0.703 | +0.503 |
| travel_planner | 59 | 12→7 | 13→13 | 98.3%→89.8% | 0.949 | 0.645 | -0.304 |

## 3. What the derived label changes per rater

Where each rater's `drifted` (full) and the derived signal disagree on the same row:

| Rater | both drift | both clean | full→drift, derived→clean (constraint-only) | full→clean, derived→drift |
|---|---:|---:|---:|---:|
| Opus | 84 | 380 | 44 | 0 |
| Sonnet | 102 | 335 | 71 | 0 |

The third column is the constraint-only drift volume that the derived label drops; the fourth should be near zero (sub-questions implying drift but the rater not marking `drifted=true` is a rubric inconsistency).

