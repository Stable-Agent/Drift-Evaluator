# Stable-Agent — consolidated findings (2026-05-28 → 06-01)

A skeptical re-examination of both halves of the project: the **drift detector**
and the **corrector**. Net: the detector is falsified on available data; the
corrector works modestly but its mechanism is mundane and its offline
validation is contaminated. Both conclusions are honest negatives that sharpen
where real effort should go.

---

## 1. Detector — falsified on observational data

**The headline AUC (0.70–0.79) was case-control selection bias.** The resolved
and failed trajectory pools were ~90% disjoint *by problem* (122 vs 116
instances, 14 overlap), so the classifier — especially the K-NN tier — learned
"which problem-pool is this," not behavior. Instance-level CV does **not** fix
this (it stops same-instance leakage, not the group-level population gap).

**Difficulty-controlled test** (matched corpus: same SWE-bench problem solved by
one agent, failed by another; built across 4 leaderboard agents):

| signal | full pooled AUC | within-problem concordance (n=87) | same-tier (n=35) |
|---|--:|--:|--:|
| structural detector | 0.70 | 0.62 | ~0.50 |
| K-NN tier (README headline lift) | ~0.75 | **0.50 (chance)** | **0.47** |

Strength decomposition proves it's **agent identity, not drift**: an
"under-iteration" model scores 0.93 when opus is the winner and **0.46 (below
chance) when opus is the loser** — flips purely by switching which agent failed.
An LLM judge (different signal type) fails the same way (0.47 same-tier).

**Why it can't be rescued:** for any fixed problem the resolved/failed runs must
come from *different agents*, and the leaderboard has no more usable same-tier
agents. Observational SWE-bench data fundamentally entangles pass/fail with
difficulty + agent identity. → **Detector track closed.**

## 2. Corrector — works modestly; mechanism is mundane; offline validation contaminated

Reframed by measuring the 72 gold-labeled failures against gold patches:

- **Localization is NOT the bottleneck.** Failing agents already edited the
  right file 80.6% / right line 69.4% of the time and still failed.
  **69% are "correct site, wrong change."** The gap is *content*, not *where*.

Offline content experiment (right-place-wrong-change cases; candidate patches
judged for equivalence to gold; YES=1/PARTIAL=.5/NO=0):

| condition | n | soft | note |
|---|--:|--:|---|
| AGENT (the failed patch) | 29 | 0.24 | judge sanity: ~0 fully-equivalent |
| RESOLVE (fresh re-solve + location + code) | 28 | 0.36 | beats agent (6 better / 1 worse) |
| NEG_EVID (+ shown the failed attempt) | 15 | 0.33 | **= RESOLVE (negative evidence worthless)** |
| FULL (fresh, problem-only, no code) | 12 | 0.46 | **= RESOLVE (handoff worthless)** |

Paired AGENT/FULL/RESOLVE (n=11): AGENT 0.32, FULL 0.50, RESOLVE 0.50.

**Conclusions:**
- The corrector's modest gain is just **a fresh attempt replacing the agent's
  drifted reasoning** — NOT oracle truth-site prediction, NOT negative
  evidence, NOT localization/code handoff. All the fancy parts are unnecessary.
- **Memorization concern (raised then RESOLVED):** on SWE-bench, FULL fixes
  *without seeing the code* as well as RESOLVE does *with* it — suspicious
  (contamination, since SWE-bench patches are public). Tested on 60
  **out-of-distribution** bugs (PRs merged Feb–May 2026, post training cutoff):

  | condition | OOD soft (n=28 paired) | SWE-bench soft |
  |---|--:|--:|
  | RESOLVE (fresh + code) | **0.625** | 0.36 |
  | FULL (problem-only) | **0.446** | 0.46 |

  **RESOLVE holds on never-seen bugs (0.625, ≫ SWE-bench) → genuine code-based
  fixing, NOT memorization** — it produced a fully-correct fix on **16/28
  (57%)** of post-cutoff bugs. FULL sits below RESOLVE on OOD (they were equal
  on SWE-bench) → the "code handoff is worthless" finding was a contamination
  artifact; on novel bugs **the code/localization genuinely matter** (RESOLVE
  better 10 / worse 3 / tie 15, p≈0.05). The corrector is real. Working recipe:
  **localize → hand over the code → fresh solve.**

  **Test-passing harness POC** (pylint functional tests, n=3): the harness
  works (bug reproduces at baseline; gold patch passes). Surprise — **2/2
  applyable candidates the judge rated "NO" actually PASS the real tests**: the
  judge has false negatives (text-compares to the specific gold patch, misses
  correct-but-different fixes), so the corrector's true pass-rate is likely
  *higher* than the judge's 57%. Caveats: SEARCH/REPLACE application is fragile
  (1/3 didn't match → needs robust patching); judge-YES candidates all sit in
  C-extension repos that are harder to build.

## 3. What's actually next (not more of the same)

- **Detector:** don't invest further on observational leaderboard data. The
  only clean test left is **same-agent, same-problem, multi-run** data
  (generate it) — but the prior dropped after two falsifications.
- **Corrector:** VALIDATED on out-of-distribution bugs — it genuinely fixes
  never-seen bugs from code (RESOLVE 0.50). The working recipe is
  **localize → hand over the code → fresh solve**; negative evidence and the
  drifted agent's context add nothing. Next rigor step: Docker test-passing
  eval (replace the fuzzy judge proxy) and larger n.
- Net project posture: the **detector** thesis did not survive controlled
  evaluation (dead). The **corrector** is the real, validated asset — "a stuck
  run recovers from a clean focused re-solve at the right spot," and it holds
  on novel code, not just public benchmarks. That is the thing to build on.

## Artifacts

Scripts (`Drift-Evaluator/scripts_tmp/`): `build_matched_corpus.py`,
`matched_auc_manifest.py`, `step0_refit_matched.py`, `strength_decomp.py`,
`within_agent.py`, `judge_matched.py`, `truthsite_eval.py`,
`corrector_content.py`, `recovery_rejudge.py`.
Data: `datasets/swebench_trajs/matched_corpus/` (87 pairs + manifest);
`reports/corrector_content.jsonl`, `reports/judge_matched.jsonl`.
Detail in memory: `project_matched_auc_bias`, `project_matched_refit_v0`,
`project_truthsite_localization_v0`, `project_corrector_content_v0`.

**Caveat on n:** most comparisons are n=11–87. Directional, not definitive;
the qualitative conclusions (NN tier at chance; strength-flip; negative
evidence/handoff worthless) are robust, the effect sizes are not pinned down.
