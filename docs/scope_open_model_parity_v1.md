# Revised scope: open models at/above frontier on drift detection + correction

**Status:** proposed re-scope, 2026-06-16. Supersedes the "frontier drift
detection is primary, open-source uplift is auxiliary" framing
([[project-open-source-uplift]]). This document is the strategic spine; the
build specs it references are `spec_multiseed_detector_v1.md` and
`prereg_corrector_file_loc_v1.md`.

## 0. The pivot in one line

From **measuring** drift (a benchmarking project) to **closing a capability
gap** (a training + scaffolding + head-to-head-eval project). The deliverable
becomes an open model (+ scaffold) that matches or beats a frontier model on a
*narrowly specified* detection and correction task — with frontier reframed as
the **teacher and the baseline to beat**, not the product.

## 1. The bar (state it precisely; do not overclaim)

Achievable and defensible:
> On execution-scored bug correction inside a test-driven loop, AND on
> white-box drift detection, an open 14–32B model + scaffold + rejection-
> sample fine-tuning matches or beats a frontier model used via API,
> measured head-to-head with confidence intervals.

NOT claimed: open model beats frontier at open-ended one-shot coding. We pick
regimes where open-weight access is the edge and frontier's raw-reasoning
advantage is least decisive.

Evidence this is the right premise: in this session, swapping a raw bash agent
loop for a structured SEARCH/REPLACE edit tool moved qwen2.5-coder:14b from
0 → 2/4 non-empty patches on one instance. A large fraction of the open↔frontier
gap is **scaffold, not weights** — which is what makes the pivot tractable.

## 2. Pick the regimes where open models can actually win

### 2a. Detection → go white-box (the strongest "better than frontier" angle)
Frontier APIs do not expose logprobs, token entropy, cheap resampling, or
activations. Open weights do. Build detection on signals frontier *cannot*
offer:
- self-consistency / disagreement across resampled continuations,
- token-level logprob / entropy spikes at the drift point,
- (stretch) activation probes for "off-track" states.
This abandons the falsified observational-trajectory classification entirely
and reframes detection as white-box uncertainty estimation, where open weights
are a structural advantage rather than a disadvantage.

### 2b. Correction → execution is the equalizer
Open models are weaker one-shot but execution feedback is ground truth for
everyone. A test-driven loop (generate → run tests → read failure → revise)
substitutes for the reasoning a frontier model does silently, and a weak model
has *more* headroom to gain from it. All pieces already exist: container eval,
F2P/P2P checkpoints, the SEARCH/REPLACE apply loop.

## 3. Three workstreams

### WS1 — Rejection-sampling data engine (the unlock; half-built already)
The multi-seed harness (`gen_multiseed.py`) that *measures* "does the open
model land in the frontier band" IS the engine that *improves* it:

> generate (open model, many seeds) → execution-filter to verified-correct
> patches → SFT / rejection-sample → re-generate → measure uplift.

Every execution-verified resolved seed is a training target (problem →
correct, test-passing patch). This is STaR/rejection-sampling and it needs no
human labels — tests are the filter. Detection analog: use frontier models (or
gold) as the *teacher* to label drift/truth-sites, distill into the open model
(the validated recipe localize → hand over code → fresh solve becomes the
teacher signal).

### WS2 — White-box detector
Serve the open model with logprob/activation access (vLLM, not ollama chat).
Extract the §2a signals on the multi-seed corpus (within-cluster, where drift
is behavioral by construction). Head-to-head vs a frontier LLM-judge detector
on the same trajectories — open should win on signal classes frontier can't
compute.

### WS3 — Test-driven corrector
Close the loop from §2b: trigger on failing tests → fresh re-solve at located
sites → apply → re-run tests → revise up to k rounds. Compare open-in-loop vs
frontier-in-loop vs frontier-one-shot, execution-scored.

## 4. Head-to-head evaluation (the project's success metric)

- **Matched scaffold:** open and frontier get the *same* tools/loop. Comparing
  open+scaffold to frontier+bare-API is cheating; comparing to frontier+same-
  scaffold is the honest test.
- **Execution-scored only:** F2P/P2P pass, never an LLM judge in the loop.
- **CIs, not point estimates:** "as good as" = open within/above frontier's CI.
- **Leak rules carried over** (`prereg_corrector_file_loc_v1.md`): no oracle
  localization, within-cluster baselines for detection, no post-fix inputs.
- **Compute is a first-class requirement:** rejection-sampling + SFT/RL needs
  GPUs; a 14B on a 32GB Mac under emulation is a prototype only. The Linux-host
  direction (`RUN_ON_LINUX.md`) is the on-ramp; budget GPU for training.

## 5. What to retire

- Observational-leaderboard drift **detection** (case-control bias; dead for
  any model — see [[project-matched-auc-bias]], [[project-matched-refit-v0]]).
- The LLM-judge proxy as ground truth (use execution).
- "Frontier drift detection is the primary product" framing — frontier becomes
  teacher + baseline, open-model uplift becomes the deliverable.

## 6. Sequence

1. Finish the qwen mini-pilot (in flight) → measures the *starting* open↔frontier
   gap and confirms the band exists. This is WS1's first data point.
2. If a band exists: stand up the rejection-sampling loop (WS1) on a GPU host;
   measure uplift round over round.
3. In parallel, WS2 white-box detector on the same corpus (cheap; reuses runs).
4. WS3 test-driven corrector head-to-head once WS1 has a fine-tuned checkpoint.
5. Freeze a head-to-head prereg before claiming parity.

## 7. Honest risks

- Parity may hold only in the narrow regime (§1); say so, don't generalize.
- If the mini-pilot shows qwen-14b resolves ~nothing even with scaffolding, the
  base model is too weak and WS1 needs a stronger open base (32B+) and real
  GPU — re-evaluate before committing.
- White-box detection (WS2) is the most novel and least de-risked; treat as
  research, not a guaranteed deliverable.
