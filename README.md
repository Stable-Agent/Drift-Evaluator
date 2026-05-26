# Drift-Evaluator

Synthetic datasets and metrics for evaluating LLM drift detectors. Part of the
[Stable-Agent](https://github.com/Stable-Agent) ecosystem.

The Drift-Evaluator answers a single question: **how good is your drift detector?**
It generates labeled multi-turn conversations using a local LLM, runs your
detector against them, and produces a report with precision, recall, F1,
calibration, time-to-detection, per-signal predictive power, and a threshold
sweep.

Without an evaluator, every change to a drift-detection algorithm is guesswork.
With one, you can see whether a tweak helped, hurt, or didn't matter.

---

## Quickstart

```bash
# 1. Install Ollama and pull the generation model
brew install ollama
ollama serve &                      # leave running
ollama pull llama3.1:8b              # ~4.7 GB

# 2. Install the evaluator (editable)
pip install -e ./Drift-Evaluator
pip install -e ./Drift-Detector       # required for run_baseline.py

# 3. Generate a labeled dataset (~1–2 hours on M1 for 200 conversations)
python Drift-Evaluator/scripts/generate_dataset.py --n 200

# 4. Run the current detector against it and write a report
python Drift-Evaluator/scripts/run_baseline.py
```

Reports land in `Drift-Evaluator/reports/baseline_v0.md` and `.json`.

---

## What's in the box

```
Drift-Evaluator/
  src/
    schema.py         # Conversation / Turn / TurnLabel dataclasses + JSON IO
    ollama_client.py  # minimal HTTP client for local Ollama
    generator.py      # synthetic labeled-conversation generator
    metrics.py        # precision/recall/F1, ROC, PR, Brier, time-to-detection, etc.
    harness.py        # run a detector against a labeled dataset
    report.py         # markdown + JSON report rendering
  scripts/
    generate_dataset.py
    run_baseline.py
  tests/              # pytest unit tests for metrics, harness, schema
  datasets/           # generated datasets land here (gitignored)
  reports/            # baseline + future reports land here
```

---

## How the dataset is built

The generator does **not** rely on the LLM to label its own output. Each
conversation starts from a deterministic *scenario card*:

- `goal` and `constraints` (drawn from one of 8 domain templates: financial
  analyst, code reviewer, recipe assistant, travel planner, fitness coach,
  email drafter, SQL tutor, children's-story writer).
- `drift_type` ∈ {`clean`, `goal_drift`, `constraint_violation`,
  `consistency_break`}, sampled by configured mix (default 25/25/25/25).
- `drift_onset_turn` — the turn at which drift begins (random ≥ 3).
- `target_constraint_index` — for constraint violations.

Llama 3.1 8B generates each assistant turn with a system prompt conditioned on
whether the turn is pre- or post-onset and which drift type was selected.
**Labels are derived deterministically from the card**, not from interpretation
of the model's output. This means the labels are reliable even when the model
doesn't perfectly comply with drift instructions — and any non-compliance shows
up directly as detector evaluation noise, which is exactly what we want
measured.

Severity ramps post-onset: `low` at the onset turn, `medium` the next turn,
`high` thereafter.

---

## What the report tells you

Run on a 200-conversation dataset, the baseline report contains:

- **Headline binary metrics** at the configured threshold: precision, recall,
  F1, accuracy, plus TP/FP/TN/FN counts.
- **Calibration**: Brier score, ROC-AUC, PR-AUC over the continuous drift
  score.
- **Per-drift-type recall**: how well the detector catches goal drift vs.
  constraint violation vs. consistency break, in isolation.
- **Severity confusion matrix**: predicted vs. true severity over
  {none, low, medium, high}.
- **Time-to-detection**: for conversations with drift, how many turns elapsed
  between drift onset and the detector's first alarm. Distribution + text
  histogram. Also reports `early` flags (false positives before onset) and
  `missed` (drift never caught).
- **Per-signal predictive power**: Pearson r and ROC-AUC for each individual
  signal (goal, constraint, consistency, total) against ground truth — surfaces
  which signals carry their weight and which are dead.
- **Threshold sweep**: precision/recall/F1 across thresholds 0.05–0.95, with
  the F1-maximizing threshold highlighted.

The JSON report is the same data, machine-readable, suitable for CI regression
tracking.

---

## Custom evaluations

Evaluate a non-default detector configuration:

```python
from drift_evaluator.harness import evaluate_dataset, make_default_detector_factory
from drift_evaluator.report import write_report

factory = make_default_detector_factory(
    drift_threshold=0.20,
    weights={
        "goal_alignment": 0.50,
        "constraint_adherence": 0.30,
        "consistency": 0.15,
        "turn_degradation": 0.05,
    },
)
rows, report = evaluate_dataset(
    "Drift-Evaluator/datasets/v1_synthetic",
    factory,
    detector_threshold=0.20,
)
write_report(report, "Drift-Evaluator/reports", name="experiment_lower_threshold")
```

Evaluate a completely custom detector by writing your own factory — any
zero-arg callable returning an object with `setup(goal, constraints)` and
`check_response(response, turn) -> DriftResult` will work.

---

## Running the tests

```bash
cd Drift-Evaluator
pip install pytest
python -m pytest tests/ -v
```

30 unit tests cover schema validation, metrics math, and harness wiring. Tests
do not require Ollama or load any models — they run in under a second.

---

## Results — the four-phase story

This benchmark went through four iterations. Each phase produced a result, and
together they tell a story about what really limits drift detection on this
kind of dataset. Run on 200 synthetic conversations / 1776 turns.

### Phase 1: baseline v0 (cosine signals)

| Metric (thr=0.30) | Value |
|---|---:|
| Precision | 0.315 |
| Recall | 1.000 |
| F1 | 0.480 |
| Accuracy | 0.316 |

v0's recall=1.000 was a pathology: it flagged 1215 false positives out of 1216
negatives (the embedding-cosine constraint signal had ROC-AUC 0.49 — random).
At v0's F1-optimal threshold of 0.60, F1 reaches 0.553.

### Phase 2: v2 NLI-based detector

Replaced cosine constraints with NLI entailment, calibrated the goal signal,
fit weights/threshold from data.

| | v0 (default) | v2 fitted |
|---|---:|---:|
| F1 (own threshold) | 0.480 | 0.522 |
| Brier (lower better) | 0.294 | **0.215** |
| Severity confusion accuracy | 0.160 | **0.533** |
| Constraint AUC | 0.492 | **0.534** |

The pathology disappeared (severity classification +37 points), but F1 only
moved +0.04. NLI fixed the *wrong* signal but the constraint signal was still
weak (0.534 ROC-AUC).

### Phase 3: LLM-judge constraints

Added an opt-in mode that uses local Llama 3.1 8B to judge each
(response, constraint) pair on a 0-10 compliance scale. On a 73-conv
fully-cached subset, the constraint signal moved 0.534 → **0.590** (+0.056).
But the total ROC-AUC didn't improve and F1 stayed flat.

### Phase 3.5: dataset audit + LLM-judge relabel

A hand audit found **80% of "constraint_violation" labels were wrong** — Llama
(the generator) ignored "violate this rule" instructions and produced
compliant responses. The labels were the bottleneck, not the algorithms.

Fix: regenerate the dataset with template-specific violation directives, then
relabel via a strict LLM-judge with structured-output prompts (force the judge
to extract a quote/topics/contradicted-claims as evidence before verdict).

| | Original labels | After relabel |
|---|---:|---:|
| Drift turns | 560 | 459 confirmed |
| goal_drift agreement | — | 100% |
| constraint_violation agreement | — | 81% |
| consistency_break agreement | — | 65% |

### Phase 4: gated typed validators

The breakthrough insight from per-template analysis:

| Constraint | Validator | AUC | NLI AUC |
|---|---|---:|---:|
| email_drafter "under 150 words" | `len(words) > 150` | **0.829** | 0.504 |
| childrens_story "under 200 words" | `len(words) > 200` | **0.984** | 0.529 |

A 5-character word-count check beats an 8B-parameter NLI cross-encoder by
0.32–0.45 ROC-AUC. The mistake was treating all 24 constraints as one
"compliance detection" problem instead of routing each to the simplest matcher.

Phase 4 introduced **typed validators** (`WordCountValidator`,
`KeywordBlocklistValidator`, `RegexForbiddenValidator`,
`NumberedStepsValidator`, `CodeBlockBudgetValidator`, `NLIFallbackValidator`,
…) plus **gated constraints** so required-pattern rules only fire when the
response is actually attempting the relevant content (e.g., "email must
include CTA" only checks responses that look like email drafts).

| | NLI AUC | Typed (un-gated) | **Typed (gated)** |
|---|---:|---:|---:|
| Constraint AUC on `constraint_violation` convs | 0.534 | 0.590 | **0.626** |
| Drift-vs-clean separation on `constraint_violation` convs | — | +0.082 | **+0.111** |

**Bottom line: gated typed validators beat NLI by +0.092 ROC-AUC on the slice
where the constraint signal is supposed to fire**, with sub-millisecond
latency and explainable evidence strings ("trigger active, requirement
failed: word_count=237 > 150").

The headline overall F1 (~0.46-0.49 for all detector variants) is bounded by
the goal/consistency signals (0.64 AUC each — embedding/NLI ceilings) and the
fact that constraint-violation convs are only 25% of the dataset. In a
production setting where one drift type dominates or where signal quality on
each axis matters independently, the architectural improvement is large.

### What this benchmark proves

1. The *data*, not the algorithm, was the bottleneck for most of the project.
   Synthetic labels need careful regeneration and verification before they
   support fine-grained ML evaluation.
2. **Gated typed validators are the production-grade architecture** for
   constraint-style drift detection: cheap, explainable, composable.
3. NLI and LLM-judge belong as *fallbacks* for inherently fuzzy constraints
   ("be professional", "family-friendly"), not as the default.

Reports are checked into `reports/`:

- `baseline_v0_relabeled.md` — v0 on the relabeled dataset
- `v2_default.md` / `v2_fitted.md` — v2 NLI on the relabeled dataset
- `v3_typed_default.md` / `v3_typed_fitted.md` — v3 gated typed validators
- `comparison_v0_vs_v2_fitted.md` — side-by-side
- `audit_clean_turns.json` — clean-turn FN audit
- `fitted_config.json` — learned weights, threshold, severity buckets

To reproduce the full pipeline:

```bash
# 1. Regenerate dataset (~2 hr)
python Drift-Evaluator/scripts/generate_dataset.py --n 200

# 2. Relabel via strict LLM-judge (~45 min)
python Drift-Evaluator/scripts/relabel_dataset.py --reset-first --prompt-version v2

# 3. Run all detector variants
python Drift-Evaluator/scripts/run_baseline.py --name baseline_v0_relabeled    # v0
python Drift-Evaluator/scripts/run_phase2.py --force-eval                       # v2 NLI
python Drift-Evaluator/scripts/run_phase3.py                                    # v2 LLM-judge (slow)
python Drift-Evaluator/scripts/run_phase4.py                                    # v3 gated typed
```

## License

MIT. See `LICENSE`.
