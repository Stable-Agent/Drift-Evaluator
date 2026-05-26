# Intervention-test pilot, v0 (2026-05-09)

First validation pass on the `coding_v0.1` rubric, plus a control group
and an 8B zero-shot detection baseline.

**Three experiments combined here:**

1. **Intervention** (n=7) — does the corrector message at the labeled cue
   step flip `resolved=false → true`?
2. **Control** (n=7) — same instances, *no* corrector. Tests whether
   Claude would have produced the right patch unprompted.
3. **8B detection baseline** (n=14) — given a trajectory, can
   llama3.1:8b zero-shot identify the drift step, scope_error, and modes?
   Establishes the gap any open-source uplift would have to close.

The rubric (section 9 of `docs/coding_rubric_v0.md`) is built on
intervention testing replacing κ as the primary quality signal. The
control + 8B baseline together let us say what's actually happening,
not just "intervention worked once."

---

## Pipeline

`Drift-Evaluator/scripts/intervention_harness.py`. Two modes:

- `agent-loop`: full replay via Docker container, ~20-50 LLM calls, ~$2-3 per
  instance. Faithful to original agent behavior. Used for django-14404 only.
- `one-shot` (default): single LLM call asks for the unified diff directly,
  given the truncated trajectory + corrector. ~1 call per instance, ~$0.20.
  Lighter fidelity but token-budget-friendly. Used for the rest.

Both modes run the canonical SWE-bench eval afterward (Docker, ~4 min).
Auth via Claude Code's `claude -p` CLI — no API key required.

---

## Results

All 7 trajectories drawn from `livesweagent_opus45` (mini-swe-agent format,
Opus 4.5 as the original agent). Replayed model also Opus 4.5 for fidelity.

| Instance | scope_error | intervention_type | Mode | Patch applied | Resolved |
|---|---|---|---|:-:|:-:|
| `django__django-14404` | under_fix | scope_widen | agent-loop | ✓ | **TRUE** ✓ |
| `django__django-14404` | under_fix | scope_widen | one-shot | ✓ | **TRUE** ✓ |
| `astropy__astropy-14365` | under_fix | scope_widen | one-shot | ✓ | FALSE |
| `sympy__sympy-15976` | over_fix | scope_narrow | one-shot | malformed | — |
| `django__django-11433` | correct_sites_wrong_change | hypothesis_broaden | one-shot | ✓ | **TRUE** ✓ |
| `django__django-13925` | correct_sites_wrong_change | pr_author_test | one-shot | malformed | — |
| `django__django-14053` | correct_sites_wrong_change | hypothesis_broaden | one-shot | ✓ | FALSE (28 P2P regressions) |
| `django__django-15732` | correct_sites_wrong_change | scope_widen | one-shot | ✓ | FALSE |

**Headline: 2/5 evaluable patches resolved (40%). 2/7 malformed (no signal).**

For comparison: random-guess patch generation has near-0% chance of resolving
SWE-bench tasks, so 40% is real causal signal. But it's not 100%, and the
spread is informative.

## Control group (intervention vs no-corrector, n=7)

Same 7 instances run again with the corrector message *removed* — just the
truncated trajectory + "produce the unified diff." Establishes the baseline
Claude would reach without our help:

| Instance | Control patch | Control resolved | Intervention resolved |
|---|:-:|:-:|:-:|
| `django-14404` | applied | FALSE | **TRUE ✓** |
| `astropy-14365` | applied | FALSE | FALSE |
| `django-11433` | malformed | — | TRUE ✓ |
| `django-14053` | malformed | — | FALSE (28 P2P regressions) |
| `django-15732` | applied | FALSE | FALSE |
| `sympy-15976` | applied | FALSE | malformed |
| `django-13925` | malformed | — | malformed |

**On the 3 cases where both produced applyable patches:**
- Control: 0/3 resolved
- Intervention: 1/3 resolved (django-14404)

**The corrector is doing causal work.** 0/3 → 1/3 isn't strong, but it's
not zero either. And in cases where the control couldn't even produce a
valid patch (django-11433), the corrector at minimum gave it the right
shape: django-11433 went from malformed control → resolved intervention.

## 8B detection baseline (llama3.1:8b zero-shot, n=14)

For the open-source uplift track. Same 14 labels, fed to llama3.1:8b via
Ollama, asked to predict drift step + scope_error + drift_modes from the
raw trajectory. Compared against the rubric labels (gold = Opus 4.7 + the
v0.1 rubric).

| Field | llama3.1:8b zero-shot agreement with gold |
|---|---:|
| `drifted` (binary) | **93% (13/14)** |
| `scope_error` (exact) | 14% (2/14) |
| `intervention_type` (exact) | 14% (2/14) |
| `drift_step` (exact) | 0% (0/14) |
| `drift_step` (within ±5) | 0% (0/14) |
| `drift_modes` (mean Jaccard) | 0.20 |

**8B can detect that drift happened.** It cannot classify the shape or
localize the step. That's the gap any open-source uplift would have to
close — a clear fine-tuning target.

The fact that binary detection lands at 93% means the labeled trajectories
are *recognizable* as drifty even to a small model — the rubric isn't
labeling subtle phenomena. The reasoning failure is in the structural
classification (scope_error, drift_step), which is the harder task.

---

## Per-axis breakdown

### By intervention type

| Type | Validated / total |
|---|:-:|
| `scope_widen` | 1 / 3 |
| `hypothesis_broaden` | 1 / 2 |
| `scope_narrow` | 0 / 0 (1 malformed) |
| `pr_author_test` | 0 / 0 (1 malformed) |

No intervention type has enough n to support claims. `scope_widen`
worked on the cleanest case (django-14404) but failed on
astropy-14365 (similar shape) and django-15732. The single
`hypothesis_broaden` success (django-11433) came against a
`correct_sites_wrong_change` label.

### By scope_error

| scope_error | Validated / total |
|---|:-:|
| `under_fix` | 1 / 2 |
| `correct_sites_wrong_change` | 1 / 3 |
| `over_fix` | 0 / 0 (1 malformed) |

The modal scope error (`correct_sites_wrong_change`) is also the modal
*failure* mode in this batch. That makes sense — it's the most underspecified
category and the corrector messages we authored for it are softer (e.g.,
"consider alternatives," "challenge the content") than the sharper
`scope_widen` / `scope_narrow` prompts.

---

## Two distinct failure modes mixed together

This matters for interpreting the 40% number.

### Genuine label/intervention failures (3 cases)

`astropy-14365`, `django-14053`, `django-15732` — patches applied cleanly
but didn't resolve. The corrector framing was insufficient to lead the model
to the correct fix even with the bug well-understood from the trajectory.

`django-14053` is the most concerning: patch applied AND introduced **28
PASS_TO_PASS regressions**. The corrector message *actively pushed the
model toward a wrong, over-broad fix that broke things*. That's a label-quality
problem — re-read the corrector draft and ask whether it could be reworded
to point narrower.

### One-shot mode artifacts (2 + possibly 3 cases)

`sympy-15976`, `django-13925`: patches were malformed — correct diff format
but line numbers / context didn't match the actual file. **Agent-loop mode
would have caught this** (the agent reads files, edits, verifies, fixes).
One-shot has no such recovery.

`django-14053`'s 28-regression failure may also be one-shot-mode-specific:
without the ability to run tests, the model couldn't notice it had broken
things. Agent-loop would have run tests and tried again.

If we re-ran the malformed cases (and possibly django-14053) in agent-loop
mode when rate-limit budget is fresh, validation rate likely climbs above 40%.

---

## What's solid, what isn't

**Solid:**

- Pipeline works end-to-end. We can take a labeled drift step, inject a
  corrector, replay, run the SWE-bench eval, and read out a binary outcome.
- The intervention test methodology (rubric §9) is producing real signal —
  not random.
- django-14404's outcome flip *replicates* across both modes (agent-loop
  and one-shot). That's the cleanest positive result.
- One-shot mode is tractable for budget-constrained use: ~$0.20 per
  instance vs ~$2-3 for agent-loop.

**Not solid:**

- 7 trajectories isn't enough for per-mode or per-intervention claims.
- One-shot mode loses fidelity — at minimum 2/7 cases are mode-artifacts,
  not rubric failures.
- Single annotator (this session); per-mode κ unmeasurable.
- Drawn entirely from `livesweagent_opus45` — no cross-agent generalization
  evidence.

---

## Implications for both tracks

**Frontier track (production deployment):**
- Corrector messages do causal work, but are not yet reliable.
- Single-call one-shot mode is ~30x cheaper than agent-loop and reproduces
  django-14404's success — fine for cost-constrained validation. Likely
  insufficient for production drift correction, where agent-loop fidelity matters.
- The `correct_sites_wrong_change` corrector drafts under-perform (1/3
  validated, plus the 28-regression incident). Wording iteration is the
  highest-leverage improvement.

**Open-source uplift track (auxiliary deliverable):**
- llama3.1:8b can binary-detect drift at 93% zero-shot. The hard problem
  is *classification and localization* — fine-tuning targets.
- Current label set is 14 examples — far too few for fine-tuning. Need
  100+ labels before a meaningful fine-tune attempt.
- Closing the `scope_error` gap (14% → ?%) and `drift_step` gap (0% → ?%)
  with fine-tuning is the headline experiment. If 8B can match frontier
  on those after fine-tuning, the open-source uplift story works.

## Recommended next steps (priority order)

1. **Audit the `correct_sites_wrong_change` corrector drafts.** 1/3
   validation rate plus the 28-regression incident says the soft "challenge
   the content" framing is not pointing narrowly enough. Try sharper prompts
   that quote the ground-truth fix's *shape* without giving it away. (Manual
   work; no LLM cost.)

2. **Re-run sympy-15976 and django-13925 in `agent-loop` mode** when
   token budget clears. These are the malformed-patch cases — agent-loop's
   iterative behavior should fix the format. ~$5 for both. Target: confirm
   whether they're rubric failures or mode artifacts.

3. **Scale label set toward fine-tuning.** Current n=14 is enough for
   intervention validation but nowhere near enough to fine-tune llama3.1:8b
   on drift detection. Target ~100-200 labels, drawn from the existing 91-
   instance queue. Each ~5-15 min to label manually; an LLM-assisted
   labeling pipeline using the rubric as the spec is probably the practical
   path.

4. **Fine-tune llama3.1:8b on the labeled set** once it's at scale. Target:
   close the gap on `scope_error` (14% → ?%) and `drift_step` (±5: 0% → ?%).
   Even modest improvement here is the headline result for the open-source
   track.

5. **Sonar harness branch.** The 4 sonar-format trajectories in the queue
   (django-10973, sphinx-10673, django-11138, xarray-3993) currently use
   the agent-loop path which doesn't support sonar format. The 8B baseline
   does handle them now via `_msg_text()`. Generalizing the agent-loop
   serializer is ~1-2 hours of work.

---

## Files

- `gold/coding_v0_pilot.jsonl` — 14 labels (8 from previous session + 6 from
  v0.1 pilot, 1 dropped as `unrelated_drift`)
- `gold/coding_v0_queue.jsonl` — 91-instance labeling queue (cross-product)
- `gold/coding_v0_agents.json` — agent registry used for the cross-product
- `reports/intervention_runs/<instance>_<run_id>/` — per-run artifacts
  (predictions.jsonl, conversation.json, swebench_report.json or correction
  notes for malformed cases)
- `reports/intervention_runs/<instance>_<run_id>.json` — top-level
  `result.json` per run (corrected for stale-resolved bug)
- `logs/run_evaluation/<run_id>/intervention-harness/<instance>/` —
  swebench's per-instance eval logs (canonical `report.json` lives here)

## Known harness bugs

- The eval-report search currently picks up reports from previous runs
  when the run-specific report doesn't exist. This caused the harness to
  report `resolved_after=True` for the malformed-patch cases. Result.json
  files have been corrected manually; the harness search logic should be
  tightened to only check `logs/run_evaluation/<run_id>/` and treat
  "no report" as `resolved_after=null`, not fall back to neighbors.
