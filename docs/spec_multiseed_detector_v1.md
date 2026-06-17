# Spec: same-agent multi-seed drift corpus + detector (v1)

**Status:** design, 2026-06-16. Not frozen — this is the build spec; a
per-experiment prereg (decision rules, n) comes *after* the pilot in §7.

## Why this exists

Every detector attempt to date ran on **observational leaderboard
trajectories**, where pass/fail is entangled with problem difficulty and
agent identity. The matched-corpus + strength-decomposition work
(`strength_decomp.py`, `project_matched_auc_bias`) proved the apparent
"drift signal" there is agent identity: an under-iteration model flips
0.93 → 0.46 by swapping which agent failed. That regime is closed.

The one untried regime: **one agent, one model, many problems × many seeds.**
Within a single (problem, agent) cluster, difficulty is held by the problem
and agent strength is held by fixing the agent — both confounds removed *by
construction*. Any feature separating passing from failing seeds inside a
cluster is behavioral by definition. This is the only data that can resurrect
or cleanly bury the detector thesis, and it is impossible to obtain from a
leaderboard (for a fixed problem, resolved/failed runs there must come from
different agents — that IS the bias).

Prior is low (two prior falsifications). Expected value is mostly "settle it
honestly"; the runs double as the realistic failed-trajectory generator the
corrector work needs, so they are not wasted under either outcome.

## 1. Agent

`mini-swe-agent` (minimal, scriptable, single-config; easy to instrument and
to stop at step boundaries for checkpoint eval). SWE-agent proper is the
fallback if mini's scaffold is too weak to reach the capability band in §3.

Rationale: we need a scaffold we can (a) drive headless in a loop over
seeds, (b) pause at each step to snapshot the cumulative diff, (c) read raw
model responses from. mini-swe-agent gives all three; heavier agents bury the
trajectory in framework noise.

## 2. Model

A **mid-tier hosted model at temperature ≈ 0.7** (seed variation requires
sampling temperature > 0). Local 14b is out — `project_llama_experiment_v0`
showed it fails as a patch generator, and the agent must actually solve a
meaningful fraction or there is no pass/fail split to study.

Fork on signal type 3 (introspection, §6.3):
- **Default (signals 1, 2, stall-label):** any hosted mid-tier model. Needs
  only trajectories + checkpoint F2P. Ship this first.
- **Stretch (signal 3):** an open model served locally via vLLM/llama.cpp
  with `logprobs` enabled, *strong enough* to stay in the capability band.
  Only pursue if the model can both solve ~half the problems and expose
  token logprobs. Do not block v1 on it.

Fix and record: model id, temperature, top_p, scaffold version, max steps.

## 3. Problem set — the capability frontier

The corpus is only useful where seeds **split** (some pass, some fail). A
problem the agent always solves or never solves carries zero within-cluster
signal.

1. Candidate pool: SWE-bench Verified, restricted to repos that build and
   test in our existing execution harness (django, sympy, xarray, pylint,
   sklearn, flask, requests — *not* the C-extension repos that broke OOD
   sourcing).
2. **Pilot pass-rate estimation:** run the agent at K=4 seeds over ~120
   candidate problems. Keep problems whose pilot pass rate is in **[0.25,
   0.75]** — the frontier band. Target ≈ 50 retained "frontier problems."
3. This pilot is also the §7 power check: if fewer than ~30 problems land in
   the band, widen the pool or change model tier before the full run.

## 4. Seeds and scale

- **8 seeds per frontier problem** (seeds 0–7, distinct RNG seeds; record the
  exact seed and any framework nondeterminism caveats).
- 50 problems × 8 = **400 runs**. A run is in the **analysis set** iff its
  problem-cluster has ≥1 pass and ≥1 fail seed (clusters that ended up
  all-pass/all-fail in the full run are dropped from within-cluster analysis
  but kept for the corrector-data side).
- Budget note: caps have repeatedly bitten this project. Drive the run with
  the same resumable, round-based pattern as `ood_filearm_v1.py`
  (per-(problem,seed) done-set, idempotent re-runs).

## 5. What to log per run (load-bearing)

One JSONL row per `(problem, seed)` plus a per-step sidecar. Capture enough to
compute *every* signal type from one corpus — re-running to add a field later
is the expensive mistake.

Per run:
- `problem_id, seed, model, temperature, scaffold_version, max_steps`
- `resolved: bool` (terminal SWE-bench eval)
- full `trajectory`: ordered list of steps; each step =
  `{action, raw_model_text, tool, tool_result_excerpt}`

Per step (the checkpoint sidecar — this is the in-flight execution layer):
- `step_idx`
- `cumulative_diff` (agent's patch so far)
- `diff_stats`: files touched, +/- lines (reuse `ood_filearm` diff parsing)
- **`f2p_pass, f2p_total, p2p_pass, p2p_total`** — apply `cumulative_diff` at
  the base commit, run the problem's FAIL_TO_PASS and PASS_TO_PASS tests,
  count passes. This is the execution-progress signal; it is the spine of the
  stall-label and of signal 2.
- structural features at this prefix (reuse `repetition_signals.py`:
  `max_diff_similarity`, `max_file_read_repeats`, `max_hunk_header_repeats`)
- `logprob_summary` (signal 3, if available): mean/min token logprob of the
  step's action, self-consistency over R resampled continuations.

Checkpoint cadence: every step is ideal but expensive (one test run per
step). Default to **every step for ≤20-step runs, every 2 steps beyond** to
cap Docker cost; record the cadence so curves are comparable.

Reuse, don't reinvent: the checkpoint test-runner is `ood_filearm_v1.py`'s
`checkout_base → apply_diff → run_tests` loop, generalized to count F2P/P2P
instead of a single pass/fail, and pointed at intermediate diffs.

## 6. The analyses this corpus unlocks (none possible on leaderboard data)

### 6.1 Divergence from the agent's own success manifold
For each problem, build the distribution over passing seeds of (feature vector
at step *t*). Score a held-out run by distance to that same-problem success
distribution at matched *t*. Cross-problem comparison never happens →
difficulty/agent confounds structurally absent. Leave-one-seed-out within
problem; report within-cluster AUC for pass vs fail.

### 6.2 Per-step counterfactual value
Using the seed distribution as the counterfactual, estimate whether a given
action lowered P(resolve). Concretely: does the F2P trajectory after step *s*
fall below the same-problem seed median from step *s* onward? This is credit
assignment, not trajectory classification — a more actionable target and a
distinct contribution.

### 6.3 Introspective / logprob signals (stretch)
Token-level uncertainty and answer self-consistency across resampled
continuations, as a within-cluster pass/fail separator. Different signal
family from structural counts; requires §2 stretch serving.

### 6.4 Stall-label (reframed target)
In-flight-v1 found failure here is **under-fix, not regression**. So predict
*stall*, not terminal failure: a run is "stalled at step s" if F2P pass-count
does not increase for **k consecutive checkpoints while edits continue**
(diff still changing). The multi-seed data makes this well-defined — a failing
seed's F2P plateaus while sibling seeds' F2P keeps climbing on the same
problem. Detector target becomes change-point detection on the F2P curve.
Tune k on the pilot; report detection lead time vs terminal failure.

## 7. Before the full run — pilot gate + prereg

1. Run §3 pilot (K=4 × 120 problems). Confirm ≥30 frontier problems and that
   F2P checkpoint eval is stable (gold patch drives F2P to full; empty diff
   leaves it at 0) — the same sanity gate as `ood_filearm` evaluability.
2. *Then* write `prereg_multiseed_detector_v1.md`: pick the primary analysis
   (recommend 6.1 within-cluster AUC), set n, set the decision rule and the
   chance baseline (within-cluster, the null is each cluster's own pass rate),
   and the leak rule (no feature may use post-hoc/terminal info; checkpoint
   features at step *t* use only the prefix ≤ *t*).
3. Only after the prereg is frozen, run the full 400 and analyze.

## 8. Validity rules (carried from the corrector prereg)

- Checkpoint features at step *t* use **only** the trajectory prefix ≤ *t* —
  no terminal/outcome leakage into in-flight features.
- Execution (F2P/P2P) is ground truth; no LLM-judge in the detector loop.
- Within-cluster baselines, not pooled: the chance line is the per-problem
  pass rate, never the global one (pooled AUC is the exact trap that produced
  the falsified 0.76).
- Every run accounted for; all-pass/all-fail clusters reported, not silently
  dropped.

## Artifacts to produce
- `scripts_tmp/gen_multiseed.py` — resumable agent runner (per problem×seed)
- `scripts_tmp/checkpoint_eval.py` — intermediate-diff F2P/P2P runner
  (generalizes `ood_filearm_v1.run_tests`)
- `datasets/multiseed_v1/` — run rows + checkpoint sidecars
- `prereg_multiseed_detector_v1.md` — frozen after the pilot
