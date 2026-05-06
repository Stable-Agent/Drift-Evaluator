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

## Results to date

Run on the v1 synthetic dataset (200 conversations / 1776 turns / 31.5% drift rate).

| | v0 baseline (default config, thr=0.30) | v2 fitted (NLI signals, thr=0.40) | Δ |
|---|---:|---:|---:|
| Precision | 0.315 | 0.400 | **+0.084** |
| Recall | 1.000 | 0.754 | -0.246 |
| F1 | 0.480 | 0.522 | **+0.043** |
| Accuracy | 0.316 | 0.565 | **+0.249** |
| Brier (lower better) | 0.294 | 0.215 | **-0.079** |
| Severity-confusion accuracy | 0.160 | 0.533 | **+0.373** |

**The recall drop is misleading.** v0's recall=1.000 was a side-effect of its
catastrophic false-positive rate (1215 FPs out of 1216 negatives — it flagged
nearly every turn). v2 trades that pathology for an honest classifier:
+25 points of accuracy and +37 points of severity-classification accuracy.

### Per-signal predictive power

| Signal | v0 ROC-AUC | v2 ROC-AUC |
|---|---:|---:|
| goal | 0.648 | 0.647 |
| constraint | **0.492** (random) | 0.526 |
| consistency | 0.656 | 0.678 |
| total | 0.693 | 0.678 |

The v0 constraint signal was actively useless (Pearson r=−0.03, ROC-AUC at
random). v2's NLI-based constraint signal is better but still weak. NLI
entailment of "the response complies with: {constraint}" doesn't cleanly
discriminate violations for arbitrary natural-language constraints — that's the
biggest remaining gap and the natural target for Phase 3 (likely LLM-judge mode).

Reports are checked into `reports/`:

- `baseline_v0.md` — original detector
- `v2_default.md` — v2 detector with default weights
- `v2_fitted.md` — v2 detector with refit weights/threshold
- `comparison_v0_vs_v2_fitted.md` — side-by-side
- `fitted_config.json` — learned weights, threshold, severity buckets

To reproduce:

```bash
python Drift-Evaluator/scripts/run_phase2.py
```

(uses `reports/v2_default_rows.jsonl` cache to skip the ~15-min NLI re-eval
unless `--force-eval` is passed).

## License

MIT. See `LICENSE`.
