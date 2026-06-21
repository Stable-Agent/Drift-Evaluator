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

## 0.5 Prior art to build on (verified 2026-06-20) — do NOT reinvent

The correction half of "open ≈ frontier" is **already substantially
demonstrated** by open projects. Build on them; our novelty is detection +
the closed loop, not the data engine.

- **SWE-smith** (github.com/SWE-bench/SWE-smith, NeurIPS 2025 spotlight): 52K
  task instances, 26K trajectories, 250+ Docker envs, and **SWE-agent-LM-32B —
  a fine-tuned Qwen2.5-Coder hitting 40.2% pass@1 on SWE-bench Verified**, open
  checkpoint on HF. This is the parity thesis for *correction*, already done.
- **SWE-Gym** (github.com/SWE-Gym/SWE-Gym): executable training env, 2.4K tasks
  (234 Lite), prebuilt images, GPT-4o/Sonnet trajectories; 32B → 32% Verified
  with self-improvement; **+14% absolute from <500 trajectories**. This IS the
  rejection-sampling data engine of WS1, maintained.
- **Aider** / **mini-swe-agent**: robust open agent scaffolds (SEARCH/REPLACE
  edit formats, ollama support) — replace our hand-rolled loop.

Implication: WS1 adopts SWE-Gym/SWE-smith rather than rebuilding; the **agent
becomes SWE-agent-LM-32B** (≈40% Verified) instead of raw qwen2.5-coder:14b
(mini-pilot: 25% band, 0 full resolves — too weak). Our distinctive
contribution narrows to **white-box detection** and the **execution-in-the-loop
corrector** — the parts SWE-Gym/SWE-smith do not target.

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

### WS1 — Rejection-sampling data engine (ADOPT SWE-Gym/SWE-smith, don't rebuild)
The loop is: generate (open model, many seeds) → execution-filter to
verified-correct patches → SFT/rejection-sample → re-generate → measure uplift.
**SWE-Gym and SWE-smith already provide this** (envs, trajectories, the +14%-
from-<500-trajectories result, and a 40.2% checkpoint). So WS1 = use their
data/env + checkpoint as the starting point; our `gen_multiseed.py` becomes the
*local within-cluster measurement + white-box signal extractor* on top, not the
training engine itself. Detection analog still ours: frontier (or gold) as
teacher to label drift/truth-sites, distilled into the open model.

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

## 6. Sequence (revised: build on prior art)

1. ✅ Mini-pilot done — qwen2.5-coder:14b: 25% band, 0 full resolves (too weak
   as the agent). Harness validated end-to-end.
2. **Swap the agent to SWE-agent-LM-32B** (≈40% Verified) and re-pilot the
   band — far more in-band expected. Run arm64-native locally (Miniforge fix)
   or on GPU; 32B inference is the point a GPU host earns its keep.
3. WS2 white-box detector on the multi-seed corpus (cheap; our novelty; needs
   the model served with logprobs via vLLM).
4. WS1 uplift: start from SWE-Gym/SWE-smith data+checkpoint; measure round-over-
   round gains with our execution-verified within-cluster filter.
5. WS3 execution-in-the-loop corrector, head-to-head vs frontier.
6. Freeze a head-to-head prereg before claiming parity.

## 7. Honest risks

- Parity may hold only in the narrow regime (§1); say so, don't generalize.
- If the mini-pilot shows qwen-14b resolves ~nothing even with scaffolding, the
  base model is too weak and WS1 needs a stronger open base (32B+) and real
  GPU — re-evaluate before committing.
- White-box detection (WS2) is the most novel and least de-risked; treat as
  research, not a guaranteed deliverable.
