# Stable-Agent v0.3 architecture (coding paradigm)

As of 2026-05-10, the three libraries finally fit together end-to-end
for the coding-trajectory paradigm. v0.1/v0.2 work in
`Drift-Evaluator/docs/coding_rubric_v0.md` and
`Drift-Evaluator/docs/corrector_audit_v0_to_v0_2.md` defined the rubric
and corrector-design principles. v0.3 ports those into the production
libraries.

## Pipeline

```
trajectory ──► CodingDriftDetector ──► CodingDriftLabel ──► CodingDriftCorrector ──► message
              (Drift-Detector)                              (Drift-Corrector)         │
                                            ↑                                         ▼
                                            │                              injected as user turn
                                            │                                         │
                                  (offline) gold labels                                ▼
                                            │                                  agent resumes
                                            │                                         │
                                   Drift-Evaluator                                    ▼
                                       scores                                  patch + eval gate
                                       detector                                       │
                                                                                      ▼
                                                                                 resolved?
```

## Library-by-library

### Drift-Corrector (`Drift-Corrector/src/`)

- `coding_strategies.py` (new) — canonical `CodingDriftLabel` dataclass
  + `CodingSite` + ABC + 8 concrete strategies (one per
  `intervention_type`).
- `coding_corrector.py` (new) — `CodingDriftCorrector` dispatcher.
- Strategies template the corrector message from the label's
  `truth_sites` / `agent_sites` per the v0.3 design principle ("don't
  ask the model to find what the rubric already knows").
- Existing chat-paradigm strategies (`strategies.py`,
  `corrector.py`) are unchanged — backward compat preserved.

### Drift-Detector (`Drift-Detector/src/`)

- `coding_detector.py` (new) — `CodingDriftDetector` ABC plus three
  concrete backends:
  - `ClaudeCodingDetector`: calls Claude via `claude -p` CLI
    (frontier-quality, uses Claude Code subscription auth).
  - `OllamaCodingDetector`: calls local Ollama (free, no rate limits,
    lower quality — currently ~14% scope_error accuracy zero-shot).
  - `CallableCodingDetector`: any user-supplied completion fn.
- All three return `CodingDriftLabel` (imported from Drift-Corrector to
  keep the schema canonical).
- Trajectory formatter handles both mini-swe-agent (`content`) and
  Sonar (`blocks`) message schemas.

### Drift-Evaluator (`Drift-Evaluator/`)

- Rubric + gold labels in `gold/coding_v0_pilot.jsonl` (n=14).
- `scripts/intervention_harness.py` — modes:
  - default: read intervention from gold label.
  - `--corrector-text`: hand-supplied wording (audit experiments).
  - `--use-detector claude|ollama`: full e2e — detect + render +
    inject + eval.
  - `--no-corrector`: control group (no corrector at all).
- `scripts/llama_detector_baseline.py` — measures detector vs gold
  per-field agreement.

## Closed-loop validation formula

```
e2e_flip_rate ≈ detector_accuracy × corrector_quality × model_capability
```

Pilot measurements (django-14404 anchor case):

| Pipeline | Outcome |
|---|:-:|
| Hand-authored corrector + gold step/type | **TRUE** ✓ |
| Drift-Corrector mechanical render + gold truth_sites | **TRUE** ✓ |
| **Drift-Corrector + Claude detector with Track 2 truth_sites prediction** | **TRUE** ✓ ← full e2e through real libraries |
| Drift-Corrector + Claude detector (Track 1, no truth_sites prediction) | FALSE — fell back to generic template |
| Drift-Corrector + llama detector | FALSE — wrong step + type |
| No corrector (control) | FALSE |

## Track 2: detector predicts truth_sites

The `DETECTOR_PROMPT` was extended (2026-05-10) to ask the detector for
`predicted_truth_sites` and `predicted_extra_sites` alongside the label
fields. Drift-Detector populates `CodingDriftLabel.truth_sites` and
`extra_sites` from these predictions. Drift-Corrector strategies treat
predicted truth_sites identically to gold truth_sites.

On django-14404 with Claude as detector:

- Detector picked `intervention_type = content_challenge` (gold:
  `scope_widen`) — *wrong intervention type*
- Detector predicted truth_sites at lines 424 and 429 with correct
  descriptions — *right sites*
- Drift-Corrector's `ContentChallengeStrategy` rendered the truth_sites
  as a bulleted comparison
- Claude one-shot generated the *exact* correct patch
- SWE-bench eval flipped to TRUE

**The lesson: truth_sites prediction is more important than
intervention_type prediction.** A "wrong" intervention_type with
correct truth_sites still produces a working corrector. Generic
fallback (no truth_sites) does not.

## Track 2 multi-case validation (n=7)

After validating Track 2 on django-14404, the architecture was tested
across all 7 livesweagent_opus45 evaluable cases:

| Instance | Track 2 outcome |
|---|:-:|
| django-14404 | **TRUE** ✓ |
| astropy-14365 | malformed (wrong predicted truth_sites) |
| sympy-15976 | no intervention (detector confidence=0) |
| django-11433 | FALSE (4 P2P regressions) |
| django-13925 | malformed |
| django-14053 | FALSE |
| django-15732 | FALSE |

**Track 2 flip rate: 1/4 evaluable (matches the ceiling across all approaches).**

Across 5 distinct methods tested on the same set, every method peaks at
~1/3 evaluable cases flipped, but **each method flips a different
case**:

| Approach | Cases flipped |
|---|---|
| v0.1 hand-authored (leaky) | django-11433 |
| v0.2 prescriptive | django-13925 |
| Drift-Corrector + gold truth_sites | django-14404 |
| **Track 2 e2e** | django-14404 |
| No corrector | (none) |

The ceiling is structural, not architectural. The corrector is doing
real causal work (control rate is 0); but the ceiling on what one
intervention message can achieve in one-shot mode appears to be around
1 in 3-4 cases for this difficulty level.

### Sub-findings from the multi-case batch

1. **Detector's `intervention_type` and `scope_error` predictions
   often match gold.** Astropy, django-11433, django-13925 all had
   correct type+scope. Not the bottleneck.
2. **`truth_sites` accuracy is the load-bearing variable.** Astropy
   failed because the detector predicted lines 63/100 (real missed
   site is line 309). Pattern: detector knows there's drift, knows the
   shape, but can't accurately localize the missed/extra sites without
   running the code or seeing the ground-truth fix.
3. **Detector confidence is meaningful.** sympy-15976's confidence=0
   prediction correctly meant "I don't know what's wrong here; don't
   inject." With confidence-based gating, low-confidence cases could
   be skipped — cleaner than producing bad patches.
4. **One-shot mode is the bottleneck on at least 2 cases.**
   django-11433 (4 regressions) and django-14053 (1 regression) produced
   wrong-direction patches that an agent-loop's test feedback would
   likely catch and revise.

**Library quality is validated.** Drift-Corrector's mechanical render
matches hand-authored when given gold truth_sites. The port is not a
regression.

**Remaining bottlenecks:**

1. **Detector type-prediction accuracy.** Even Claude as detector picked a
   different `intervention_type` than gold. Llama is much worse (14% on
   `scope_error`).
2. **`truth_sites` availability in production.** Without truth_sites, the
   corrector strategies fall back to generic templates that don't ground
   the model on the specific missed/extra sites. Production detector
   needs an inference path — possibly: detector predicts the
   *missing* sites, not just the labels.

## Pipeline hardening (added 2026-05-10)

Three improvements landed after the Track 2 multi-case batch surfaced
recurring pipeline-bug patterns (stale reports, malformed patches, bad
predictions counted as failures):

1. **Run-specific report search** — the harness now looks ONLY at
   `logs/run_evaluation/<run_id>/intervention-harness/<instance>/report.json`
   instead of globbing `logs/**/report.json`. Eliminates the
   stale-report bug that caused false "resolved=TRUE" reports in the
   v0.2 and Track 2 batches.
2. **Patch-apply detection** — when no canonical report is produced,
   the harness now distinguishes `patch_apply_failed` from
   `eval_error` by reading `run_instance.log`. `resolved_after`
   stays `None` for these (the pipeline produced an inapplicable diff)
   and the result record gets `end_reason=patch_apply_failed`.
3. **Confidence gating** — `--confidence-threshold` (default 0.5)
   skips intervention when detector confidence is below threshold.
   Prevents the pipeline from producing bad patches on cases where
   the detector already self-identifies as uncertain.
4. **Patch validation + retry** — `--validate-patch` runs
   `git apply --check` in a temp container based on the SWE-bench
   instance image before submitting to full eval. On failure, retries
   one-shot generation up to 2 times with the error message included
   in the corrector context. Addresses the malformed-patch failure
   mode (2/7 of Track 2 results).

   Tested on astropy-14365 (was malformed under Track 2). Patch
   validated on attempt 1 — but still didn't resolve, because the
   detector predicted wrong `truth_sites` (lines 59/100 instead of
   line 309). **Important separation: validation helps with malformed
   patches but doesn't help with wrong patches that apply cleanly.**
   The expensive failure mode (wrong truth_sites) requires detector
   improvements, not pipeline hardening.

## Track 3: file-grounded detector + generator (2026-05-10)

After Track 2 surfaced "truth_sites accuracy is the bottleneck," the
detector was extended to accept source file content for grounding.
Implementation:

1. **`files_mentioned_in_trajectory()`** (Drift-Detector) — heuristically
   extracts referenced Python file paths from the agent's bash output.
2. **`extract_files_from_image()`** (Drift-Detector) — spins up a temp
   container from the SWE-bench instance image and `cat -n`s each path,
   returning `{path: content_with_line_numbers}`.
3. **`detect(repo_file_contents=...)`** — the detector prompt now
   includes the file content. Truth_sites predictions can cite actual
   line numbers from real code instead of hallucinating.
4. **`generate_one_shot_patch(repo_file_contents=...)`** — the patch
   generator also gets file content. Eliminates context-line
   hallucinations that produced malformed patches.
5. **Validation uses SWE-bench's actual apply command** — `patch --batch
   --fuzz=5 -p1 --dry-run` instead of `git apply --check`. Matches
   eval-gate behavior.
6. **Markdown fence stripping** — Claude sometimes wraps JSON in
   ```` ```json ```` fences; parser now handles both.

### Track 3 final results (n=7, completed 2026-05-13)

Full e2e pipeline through production libraries — Detector predicts label
+ truth_sites; Drift-Corrector renders message from predicted sites;
Generator produces patch with file content; validate-patch retry on
apply failure; SWE-bench eval.

| Instance | Outcome | Notes |
|---|:-:|---|
| django-14404 | **✓ TRUE** | validated on attempt 2 |
| astropy-14365 | **✓ TRUE** | grounding found line 309 across functions |
| sympy-15976 | **✓ TRUE** | grounding enabled drift detection that previously failed |
| django-11433 | ✗ FALSE | patch valid (attempt 1), wrong content |
| django-13925 | ✗ FALSE | patch valid (attempt 1), wrong content |
| django-14053 | malformed | grounding didn't fix all context-line hallucination |
| django-15732 | ✗ FALSE | validated on attempt 3, wrong content |

**3/6 evaluable = 50% flip rate** — meaningful lift from prior ceilings.

### Comparison across configurations

| Configuration | Flip rate on same 7 cases |
|---|:-:|
| No corrector (control) | 0/3 evaluable |
| v0.1 hand-authored (Drift-Evaluator labels) | 1/3 evaluable |
| v0.2 prescriptive correctors | 1/3 evaluable |
| Drift-Corrector render + gold truth_sites | 1/1 tested |
| Track 2 (detector, no grounding) | 1/4 evaluable |
| **Track 3 (grounded detector + grounded generator + validated)** | **3/6 evaluable** |

Track 3 doubles the success rate. The 3 wins all come from grounding
giving the detector visibility into file content the agent never read.

### Where Track 3 still fails

The 3 unresolved cases (django-11433, -13925, -15732) all have:
- patch_apply succeeded (validation worked)
- detector identified correct_sites_wrong_change
- generator produced applyable patches

But the **content of the patches was wrong**. Grounding helps with
*location* prediction; it doesn't solve *content* prediction for cases
where the right fix requires algorithmic restructuring (django-14053's
yield-deduplication; django-11433's empty_values-vs-re-clean choice;
django-15732's relocate-to-helper).

These are the cases that need either:
- Agent-loop mode (test-feedback iteration)
- Better corrector templates for `correct_sites_wrong_change`
- More label data for fine-tuning detectors on truth_content (not just truth_sites)

### Track 3 results on astropy-14365

Previously: under Track 2, detector predicted truth_sites at lines
59/100 (wrong; actual missed site is line 309). Patch was malformed.
**resolved=FALSE**.

With Track 3 grounding: detector saw full qdp.py content (24K chars,
extracted in 5s from Docker). Predicted truth_sites = line ~309
("change `if v == 'NO'` to `if v.upper() == 'NO'`") — **exactly the
gold-truth missed site**. Generator produced a 2-hunk patch matching
the gold-truth peer patch byte-for-byte. **resolved=TRUE**. ✓

### Updated case table (full e2e pipeline via production libraries)

| Case | Track 2 (no grounding) | Track 3 (grounded) |
|---|:-:|:-:|
| django-14404 | ✓ TRUE | (untested; expected TRUE) |
| **astropy-14365** | ✗ malformed | **✓ TRUE** |
| sympy-15976 | detector=no_drift | (untested) |
| django-11433 | ✗ (4 regressions) | (untested) |
| django-13925 | ✗ malformed | (untested) |
| django-14053 | ✗ | (untested) |
| django-15732 | ✗ | (untested) |

**Two cases now flip through real production pipeline:** django-14404
(adjacent sites in same function) and astropy-14365 (sites in different
functions, 240 lines apart). Track 3 extended the architecture to handle
the harder cross-function shape.

### What Track 3 still doesn't address

- Cases where the right intervention isn't a *missed site* but a
  fundamentally different *approach* (e.g., django-14053's
  yield-deduplication). Grounding the detector with file content
  doesn't help when the bug requires algorithmic restructuring.
- Cases where the agent's wrong patch is on a wrong-direction approach
  (e.g., django-11433's re-clean approach). Even with file content,
  the corrector still has to convince the model to abandon its current
  framing.

These probably need agent-loop mode (test feedback) or further label
scaling to learn corrector patterns specific to these failure shapes.

## Open follow-ups (priority)

1. ~~**Measure Drift-Corrector mechanical render against eval gate.**~~
   Done — flipped django-14404 just like hand-authored. Library quality
   confirmed. Still need to test more cases.
2. **Make the detector predict missing/extra sites, not just labels.**
   The current detector outputs `intervention_type` but no `truth_sites`
   prediction. To handle production (where gold isn't available), the
   detector should predict *which sites the agent should also have
   patched* (or shouldn't have). That's the data Drift-Corrector needs
   for non-generic templates.
3. **Test ClaudeCodingDetector + gold truth_sites enrichment.** Hybrid:
   detector picks step+type, but harness enriches with gold's
   truth_sites before rendering. Decouples detector type-prediction
   from truth_sites availability.
4. **Detector type-prediction tuning.** Claude picked
   `content_challenge` over `scope_widen` on django-14404; that's a
   prompt issue. Iterate the detector prompt or the
   `intervention_type` definitions.
5. **Scale gold labels toward 100+** so a llama fine-tune is tractable
   for the open-source uplift track.
6. **Train a dedicated detector** (fine-tuned llama3.1:8b on gold) once
   labels are at scale. Target: scope_error 14% → 70%+; drift_step ±5:
   0% → 50%+.

## What this *doesn't* solve yet

- The chat-paradigm code in all three libraries is still present and
  untouched. If we want to deprecate it (or unify the paradigms behind
  a single API), that's separate work.
- The Detector backends are LLM-call wrappers with no caching, batching,
  or async. Fine for evaluation; would need work for production
  throughput.
- The Corrector strategies are templated but not yet
  template-tested. We don't have data showing
  rendered-corrector-from-gold ≥ hand-authored-corrector flip rate.
  That's experiment (1) above.
