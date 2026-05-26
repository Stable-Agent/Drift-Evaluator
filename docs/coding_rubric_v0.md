# Coding-Agent Drift Rubric (`coding_v0.1`)

> **v0.1 changes (2026-05-08, after 7-trajectory pilot on LiveSWEAgent + Opus 4.5):**
> - Demoted `pr_author_test_unsimulated` and `custom_tool_tax` from drift modes to **trajectory-wide observations** (both fired 7/7 in pilot — zero diagnostic value as binary modes).
> - Tightened `self_confirming_verification` to require *demonstrable* miss-coverage of the actual `FAIL_TO_PASS` scenario, not just "agent wrote a custom test."
> - Added a precise definition of "site" in section 3 — needed to disambiguate same-file-different-function cases (django-15732 surfaced this).
> - Added `content_challenge` intervention type. Pilot found `correct_sites_wrong_change` is the modal scope error (4/7), not under_fix or over_fix — the rubric's center of gravity needs to shift toward "right place, wrong content."
> - Section 11 updated with pilot-confirmed weaknesses.

You are labeling **coding-agent trajectories** from SWE-bench Verified. Each
trajectory ends in either a `resolved=true` (the agent's patch passes the
hidden tests) or a `resolved=false` (it doesn't). Your job is to diagnose,
on the failed runs, *why* the agent failed in a way the corrector can act on.

This rubric departs from the chat rubric (`gold_set_rubric.md`) in three
important ways:

1. **You are not classifying "drift vs not".** The outcome is already known
   from the test gate. Your job is to explain the *shape* of the failure.
2. **Your labels are validatable, not just adjudicatable.** A labeled drift
   step is *correct* if a corrector intervention at that step plausibly
   flips the outcome. This replaces inter-rater κ as the primary quality
   signal. (κ is still tracked, but is not the ceiling.)
3. **The unit of analysis is the trajectory, not the turn.** Coding
   trajectories are 50–200 messages. Drift is something that happens at one
   or two specific steps inside the trajectory; the rest is mostly fine.

Part 1 (sections 1–7) is the operating manual. Part 2 (sections 8–11) is
methodology — read once.

---

## 1. The labeling task in 60 seconds

For each failed trajectory, the labeling tool shows you:

- The **issue text** (the GitHub PR description the agent was given)
- The **agent trajectory** (full message log: thoughts + bash + outputs)
- The **agent's submitted patch** (`patch.diff`)
- The **ground-truth patch** from the SWE-bench task (the *correct* fix)
- The **`FAIL_TO_PASS` test names** (the hidden tests that determine resolution)
- Optional: the **test_output.txt** showing how the agent's patch failed

You produce one labeled record per trajectory, with four blocks of fields:

| Block | Purpose |
|---|---|
| **Scope analysis** | Compare the agent's patched sites to the ground-truth patched sites. Classify the shape of the scope error. |
| **Drift modes** | Mark which of the three diagnostic drift modes are present. For each, identify the step the mode crystallized at. |
| **Trajectory observations** | Record the two non-diagnostic observations (`pr_author_test_unsimulated`, `custom_tool_tax`). Tracked but not counted as drift modes — see section 4. |
| **Intervention** | Identify the earliest step at which a corrector could have changed the outcome, and what kind of intervention would have worked. |

Severity does *not* exist in this rubric. The outcome is binary; what matters
is whether intervention can rescue it.

**Decision procedure.**

1. Run scope analysis first. It anchors every later judgment.
2. Then walk the trajectory once and mark drift modes as you encounter them.
3. Then choose the intervention point — usually the earliest step where any
   of the drift modes locked in.

Do not form an overall "this agent was confused" vibe and back-fill — that's
the failure mode this rubric exists to prevent.

---

## 2. What you must not see

Before each trajectory, the tool hides:

- Which agent produced the trajectory
- Other labelers' annotations on the same trajectory
- Whether this trajectory has already been used in an intervention experiment

If you accidentally see any of these, set `contaminated = true` on that
trajectory and continue. We discard contaminated samples; we do not try to
correct them after the fact.

You *do* see the ground-truth patch and the failing-test names — those are
oracle inputs that the agent did not have, and your job depends on them.

---

## 3. Scope analysis (do this first)

The empirical finding behind this rubric: **the dominant failure mode of
strong coding agents is misjudging the scope of a change**. Either too narrow
(missed a second site that also needed patching) or too broad (extended a fix
to a structurally similar but semantically different site).

### 3a. Identify the patched sites on each side

A **site** is `(file, function-or-method, behavioral effect)`. Two changes
are at the same site iff they share all three. The behavioral-effect rule
exists because some functions are large and a fix at line 423 can have
nothing to do with a fix at line 568 of the same function — different
behavioral effect, different site.

Practical disambiguation:

- Same file, same function, behavioral effects depend on each other (one
  edit's correctness is contingent on the other) → **one site**.
- Same file, different functions, but the change conceptually flows through
  both (e.g., parameter at one site, consumed at the other) → **one site,
  two locations**. Record both locations under one site.
- Same file, different functions, addressing different parts of the bug
  surface → **separate sites** (this was django-15732 — agent patched the
  call-site, ground truth patched the resolution helper; same file, but
  semantically distinct sites).
- Different files, related concerns → almost always separate sites.

When in doubt, ask: "could the corrector intervene at one site without
implying anything about the other?" If yes → separate sites.

For each side (agent and ground truth), record per site:

```
{ "file": "path/to/file.py",
  "approx_line": 423,
  "what_changed": "short description",
  "why": "what the change accomplishes" }
```

Use the line numbers as written in `patch.diff`. Approximate is fine —
exactness is not the point.

### 3b. Classify the scope error

Compute:

- `missed_sites` = sites in the ground truth not patched by the agent
- `extra_sites` = sites the agent patched that are not in the ground truth

Then assign one of:

| Label | Definition |
|---|---|
| `under_fix` | `missed_sites` is non-empty, `extra_sites` empty. Agent patched a strict subset. |
| `over_fix` | `extra_sites` non-empty and at least one extra is on the *causal path* of the failure. Agent did the right thing *and* something extra that broke it. |
| `mixed` | Both `missed_sites` and `extra_sites` non-empty, with both contributing to failure. |
| `wrong_site` | Agent patched a site that is not in the ground truth and missed all real sites. Different bug entirely. |
| `correct_sites_wrong_change` | Sites are right but the *content* of the change is wrong (e.g., agent patched the right line but with the wrong replacement). |
| `unrelated_drift` | Trajectory failed for reasons orthogonal to the patch — environment failure, ran out of turns, didn't submit, etc. Use sparingly; if this is >20% of failures the labeling tool is letting bad runs through. |

The vast majority of cases will be `under_fix`, `over_fix`, or
`correct_sites_wrong_change`. If you reach for `mixed` or `wrong_site` more
than ~10% of the time, recheck — usually one of the simpler categories fits.

---

## 4. Drift modes

Three diagnostic modes, each independently present-or-absent. They are
**not** mutually exclusive — a single failed trajectory often exhibits all
three.

For each mode you mark `present=true`, record the **step number** (the
0-indexed message index in the trajectory) at which the mode crystallized,
plus a short evidence quote from that step.

Two additional observations (`pr_author_test_unsimulated`, `custom_tool_tax`)
were originally proposed as drift modes but the v0 pilot showed both fire
near 100% on this agent population, so they have zero per-case diagnostic
value as binary axes. They are still recorded — see section 4.6 — but
**do not count as drift modes** for the purpose of selecting trajectories,
training detectors, or scoring rubric coverage.

### 4a. Premature hypothesis lock-in

> The agent committed to a fix shape early, before evidence justified it,
> and never re-examined the framing as new information arrived.

`true` when **all three** hold:

1. By some early step S₀ (typically the first or second `assistant` message
   after reading the issue), the agent has stated a specific fix hypothesis
   ("change X to Y"), not a search direction ("look at file Z").
2. At least one later step contained evidence that should have prompted
   widening or revision — e.g., a code read showing the changed symbol used
   elsewhere, a related function with overlapping concerns, an existing
   case-handling utility that suggests the bug surface is broader.
3. The agent did not respond to that evidence by re-examining the original
   framing. It either ignored the evidence or noted it without updating.

Record `lock_in_step = S₀` and `cue_step` = the earliest later step where
revision was warranted but didn't happen.

`false` when:

- The agent's first hypothesis happened to be correct and was confirmed.
  This is success, not drift. (Don't penalize being right on the first try.)
- The agent had a search direction, not a fix, in early steps.
- The agent did revisit the framing — even if it landed in the same place,
  the revisitation matters.

### 4b. Self-confirming verification

> The agent designed its own tests around the hypothesis it was testing,
> rather than around the bug. The tests verified the hypothesis, not the fix.

`true` requires **all three** of:

1. The agent wrote a custom test script during the trajectory.
2. The test would have *demonstrably failed to surface* the actual
   `FAIL_TO_PASS` scenario. You verify this by reading the agent's test
   alongside the ground-truth `FAIL_TO_PASS` test name and asking: "would
   the agent's test, on the broken code, have flagged the same failure that
   the ground-truth test does?" If no → self-confirming.
3. The agent treated the test passing as confirmation that the fix works.

Record `test_step` = the step at which the misleading test was written or
last run.

**Why this is tightened from v0.** v0's criterion was just "test scoped to
the agent's hypothesis." Pilot showed this fires 7/7 — every agent writes
hypothesis-aligned tests. The rubric becomes diagnostic only when you
require the test to *miss* the actual bug surface. Expected v0.1 fire rate:
~60–70%.

`false` when:

- The agent didn't write a custom test (no opportunity to commit this mode).
- The agent wrote a test that *would* have caught the actual `FAIL_TO_PASS`
  scenario, even if it ultimately wasn't run against the right code.
- The agent wrote a test, ran it, and updated its hypothesis based on the
  result.

If you can't determine miss-coverage from reading alone, mark `present=null`
(uncertain) and note why. Don't default to `true`.

### 4c. PR-author test unsimulated *(observation, not drift mode)*

> The agent ran existing tests but never asked "what new test would the
> author of this PR add?"

This was a drift mode in v0; v0 pilot showed it fires 7/7. Demoted to an
observation: still recorded under `trajectory_observations.pr_author_test_unsimulated`
in the output schema, but does **not** count as a drift mode for selection
or scoring.

Use the same criteria as v0:

- `true` when the agent treated existing-suite-passes as sufficient
  verification, without reasoning about what new test the PR would add.
- `false` when the agent explicitly considered the PR-author's new tests,
  even briefly.

Record `verification_step` for analysis, but do not factor this into the
intervention decision. (The corresponding intervention type
`pr_author_test` is *still* available — see 5b — because for individual
trajectories this can still be the right corrector message even though it
isn't a useful per-case classifier.)

### 4d. Recovery-without-revisit

> When the agent's first attempt failed, it debugged within the same
> hypothesis frame instead of reconsidering whether the framing was right.

`true` when:

1. The agent had a verification failure at some step (test failure, syntax
   error, type error, broken file).
2. The agent recovered by re-fitting *within* the same hypothesis — e.g.,
   "the regex flag syntax is wrong, let me try a different syntax" rather
   than "wait, is the regex even the right place to fix this?"
3. The recovery was successful at the verification level (the new attempt
   passed the agent's checks).

This is distinct from 4a (premature lock-in): 4a is about the *first*
hypothesis. 4d is about the *recovery* hypothesis. A trajectory can have
4d without 4a if the first attempt was reasonable but the recovery was
narrow.

Record `failure_step` and `recovery_step`.

`false` when there was no recovery cycle, or the agent did re-examine the
framing on recovery.

### 4e. Custom-tool tax *(observation, not drift mode)*

> The agent built ad-hoc tools (custom edit scripts, fix scripts) during
> the trajectory and burned messages constructing, breaking, and recovering
> from them.

This was a drift mode in v0; v0 pilot showed it fires 7/7 on LiveSWEAgent
(it's a feature of the mini-swe-agent style — the agent reflexively builds
tools mid-trajectory). Demoted to observation. Still recorded as
`trajectory_observations.custom_tool_tax.wasted_steps` for cross-agent
comparisons (different agent populations may show very different rates),
but does **not** count as a drift mode.

Mark `true` when the agent created an in-trajectory tool that consumed ≥3
messages of overhead. The corrector has no good intervention here ("build
fewer tools" is a coaching note, not an actionable nudge), so this never
drives the `intervention.type` choice.

---

## 5. Intervention

Identify the **earliest step** at which a corrector message could plausibly
have changed the outcome. Then classify the intervention type.

### 5a. Earliest useful intervention step

This is usually the lock-in step from 4a, but not always:

- If 4a is false but 4b is true, the intervention point is *before* the
  custom test was written — at the moment the agent decided what to verify.
- If 4d is true, there's a *second* intervention opportunity at the recovery
  point, even if the first was missed.

Pick the earliest step where a one-message corrector intervention has a
plausible (>30%) chance of flipping the outcome. The threshold is loose
because you can't actually test it — it's a thought experiment.

### 5b. Intervention type

Pick one:

| Type | Prompt shape | Targets which mode/error |
|---|---|---|
| `scope_widen` | "You've identified one site for this change. List all uses of `<symbol>` in this function and check whether each requires the same change." | scope_error=under_fix |
| `scope_narrow` | "You're applying this change in two places. Are these semantically the same case, or is one structurally similar but does something different?" | scope_error=over_fix |
| `content_challenge` | "You picked the right location. Before locking in your specific change, write the simplest possible 1–2 line modification that would address this issue. Why isn't that enough? If you can't justify the complexity, use the simpler form." | scope_error=correct_sites_wrong_change *(modal error in v0 pilot — 4/7)* |
| `relocate_fix` | "Your fix is at site X (the call site / specific helper). The bug is more general than that one site — the right fix lives at site Y (the resolution helper / shared dispatcher / parent function). Move the fix there. Name the target helper explicitly." | scope_error=correct_sites_wrong_change when the agent's site choice is too narrow and the actual fix needs to live in a shared upstream helper. Added v0.2 (2026-05-09) after the django-15732 audit revealed `scope_widen` reinforced the wrong site instead of redirecting. |
| `hypothesis_broaden` | "Before patching, list 2–3 alternative interpretations of the bug. Which best explains the reported symptom?" | 4a premature_hypothesis_lock_in |
| `verification_test_design` | "Write a test that would *fail* before your fix and *pass* after, exercising the exact scenario in the issue. Run it both ways." | 4b self_confirming_verification |
| `pr_author_test` | "If this issue were closed by a PR, what new test would the PR add? Write it. Does your fix make it pass?" | 4c (now observation) — but still a valid intervention choice |
| `framing_revisit` | "That approach failed. Before trying a fix to the failure, restate what the bug actually is in one sentence." | 4d recovery_without_revisit |

Note that intervention types are not in 1-to-1 correspondence with drift
modes. A trajectory with `scope_error=correct_sites_wrong_change` and
`premature_hypothesis_lock_in=true` could be served by `content_challenge`
*or* `hypothesis_broaden`; pick whichever the labeled `cue_step` better
supports.

### 5c. Corrector message draft (optional)

Free text, ~1–3 sentences. The actual message you'd inject. Useful for the
intervention experiment downstream — labels with concrete drafts are easier
to test.

---

## 6. Worked examples

Three labeled cases from the LiveSWEAgent + Opus 4.5 / SWE-bench Verified
slice. Read all three before labeling your first trajectory.

### Example A — `django__django-14404` (under-fix)

**Bug.** `AdminSite.catch_all_view` builds a redirect URL using
`request.path_info`, which strips `FORCE_SCRIPT_NAME`. The function uses
`path_info` *twice*: once for `resolve()` (correct — routing wants the
script-name-stripped path) and once for `HttpResponsePermanentRedirect()`
(wrong — the redirect needs the full path).

**Agent's patch.** Changed line 423 from
`path = '%s/' % request.path_info` to `path = '%s/' % request.path`. This
swaps *both* downstream uses, breaking `resolve()`.

**Ground-truth patch.** Inlines the two uses: keeps `path_info` for the
`resolve()` call, uses `path` only for the redirect.

**Label:**

```yaml
scope:
  agent_sites: [{file: django/contrib/admin/sites.py, line: 423, what: "binding 'path' to '%s/' % request.path"}]
  truth_sites: [{file: ..., line: 425, what: "use path_info inside resolve()"},
                {file: ..., line: 429, what: "use path inside HttpResponsePermanentRedirect()"}]
  scope_error: under_fix
  missed_sites: ["use path inside redirect, leave resolve untouched"]

drift_modes:
  premature_hypothesis_lock_in:
    present: true
    lock_in_step: 2
    cue_step: 8        # agent reads the function and sees `path` used twice
    quote: "On line 424, the code uses request.path_info instead of request.path."
  self_confirming_verification:
    present: true
    test_step: 24
    quote: "request.path: /my_prefix/admin/something — Path that would be generated from request.path: /my_prefix/admin/something/"
    # Test only checked the redirect URL string — never invoked resolve()
  pr_author_test_unsimulated: { present: true, verification_step: 38 }
  recovery_without_revisit: { present: false }
  custom_tool_tax: { present: true, wasted_steps: 6 }   # custom edit_file.py + recovery via git checkout

intervention:
  step: 8
  type: scope_widen
  draft: "You're about to change `path = '%s/' % request.path_info`. The variable `path` is used in two places below — `resolve(path, ...)` and `HttpResponsePermanentRedirect(path)`. Do both uses want the same value?"
```

### Example B — `astropy__astropy-14365` (under-fix, repeats with recovery)

**Bug.** `ascii.qdp` parses QDP files case-sensitively. Lowercase `read serr`
should also be accepted. Two sites: a regex matching commands, and a literal
comparison `if v == "NO"` in the row parser.

**Agent's patch.** Added `re.IGNORECASE` to the regex compile. Did not touch
the `v == "NO"` literal.

**Ground-truth patch.** Both: `re.IGNORECASE` *and* `v.upper() == "NO"`.

**Label:**

```yaml
scope:
  scope_error: under_fix
  missed_sites: ["v.upper() == 'NO' in _get_tables_from_qdp_file"]

drift_modes:
  premature_hypothesis_lock_in:
    present: true
    lock_in_step: 6
    cue_step: 14   # agent literally reads `command[1].lower()` (existing case handling) but doesn't generalize
    quote: "the regex on line 71 is compiled without re.IGNORECASE"
  self_confirming_verification:
    present: true
    test_step: 50  # custom edge-case tests for "lowercase no values" — pass because IGNORECASE handles regex matching, not the v=='NO' literal
  pr_author_test_unsimulated: { present: true, verification_step: 44 }
  recovery_without_revisit:
    present: true
    failure_step: 28   # 2 test failures from inline (?i) flag
    recovery_step: 42  # switched to compile-time re.IGNORECASE; never widened scope
  custom_tool_tax: { present: true, wasted_steps: 10 }   # edit_file.py + fix_qdp.py + fix_qdp2.py

intervention:
  step: 14
  type: scope_widen
  draft: "You noticed `command[1].lower()` already exists — that means *some* code in this module already case-handles. Before fixing the regex, list every place where the QDP parser does case-sensitive comparison. Which need IGNORECASE; which need .upper()?"
```

### Example C — `sympy__sympy-15976` (over-fix)

**Bug.** Symbols ending in numbers (e.g. `x2`) render invisibly in MathML
*Presentation* output because `<msub>` is wrapped in an outer `<mi>` element
that browsers don't render. Bug is in `MathMLPresentationPrinter._print_Symbol`
only.

**Agent's patch.** Modified *both* `_print_Symbol` methods — Presentation
(correct site) *and* Content (wrong: `<ci>` is the load-bearing wrapper of
Content MathML; removing it breaks Content output).

**Ground-truth patch.** Modified Presentation only.

**Label:**

```yaml
scope:
  scope_error: over_fix
  extra_sites: ["MathMLContentPrinter._print_Symbol changes"]

drift_modes:
  premature_hypothesis_lock_in:
    present: true
    lock_in_step: 20
    cue_step: 10   # agent earlier wrote: "the issue is with presentation MathML... I should look at the one at line 745"
    quote: "I can see that the content MathML printer has the same issue. Both printers need to be fixed."
    # Note: the lock-in here is unusual — agent had it RIGHT at step 10, then drifted to over-fix at step 20.
  self_confirming_verification: { present: true, test_step: 24 }   # custom test only checked Presentation output
  pr_author_test_unsimulated: { present: true, verification_step: 44 }
  recovery_without_revisit: { present: false }
  custom_tool_tax: { present: true, wasted_steps: 6 }

intervention:
  step: 20
  type: scope_narrow
  draft: "You said earlier the issue is with Presentation MathML specifically. Now you're proposing changes to Content MathML too. What does the `<ci>` wrapper mean in Content MathML? Is the same bug actually present there?"
```

Notice the rare pattern in C: the agent *correctly* scoped at step 10 and
*then* drifted at step 20. This is still `premature_hypothesis_lock_in` —
the lock-in just happened on a *broadening* hypothesis instead of the first
hypothesis. The mode definition (4a) covers this: the rule is "committed to
a fix shape and didn't re-examine," regardless of whether that shape was
narrow or broad.

---

## 7. Edge case catalog

1. **Trajectory ran out of turns / hit token limit.** If the agent never
   submitted a patch, the failure mode is `unrelated_drift`. Do not try to
   diagnose drift in a trajectory that didn't get to commit.
2. **Patch didn't apply cleanly.** Read `report.json:patch_successfully_applied`.
   If false, the agent's edit was malformed; this is `unrelated_drift`
   unless you can clearly trace it to a drift mode.
3. **All `FAIL_TO_PASS` tests still fail but `PASS_TO_PASS` also breaks.**
   That's two failures: the fix is wrong (track via scope analysis) and the
   fix introduced regressions (note in `notes`). Don't blur them.
4. **`FAIL_TO_PASS` partially passes** (some new tests pass, some fail).
   Common in cases where the agent partially solved a multi-faceted bug.
   Code as `under_fix` and note which sub-cases were missed.
5. **Trajectory is identical to a successful one in scope and approach,
   but fails on a flaky test.** Mark `unrelated_drift` and flag for
   environment debugging — these contaminate the dataset.
6. **You disagree with the ground-truth patch.** Sometimes the SWE-bench
   "correct" patch is itself debatable. Label the agent against the
   ground-truth patch as the rubric requires, but add a `notes` entry. The
   maintainer will catch systematic issues this way.
7. **Drift mode "almost true" — close call.** Default to `false` and add a
   `notes` quote. False positives on drift modes hurt the corrector training
   more than false negatives.
8. **You can't pick an intervention point.** That's a label of "no clear
   intervention" — record `intervention.step = null` and explain why. This
   is itself useful signal: trajectories where no early intervention helps
   are a different population from those where an early nudge would have
   worked.

---

## 8. Output format (reference)

One JSON record per labeled trajectory, written by the labeling tool:

```json
{
  "annotator_id": "alice",
  "trajectory_id": "django__django-14404",
  "trajectory_source": "livesweagent_opus45",
  "scope": {
    "agent_sites": [{"file": "...", "approx_line": 423, "what": "...", "why": "..."}],
    "truth_sites": [{"file": "...", "approx_line": 425, "what": "...", "why": "..."}],
    "missed_sites": ["..."],
    "extra_sites": [],
    "scope_error": "under_fix"
  },
  "drift_modes": {
    "premature_hypothesis_lock_in": {"present": true, "lock_in_step": 2, "cue_step": 8, "quote": "..."},
    "self_confirming_verification": {"present": true, "test_step": 24, "quote": "..."},
    "recovery_without_revisit":    {"present": false}
  },
  "trajectory_observations": {
    "pr_author_test_unsimulated":  {"present": true, "verification_step": 38},
    "custom_tool_tax":             {"present": true, "wasted_steps": 6}
  },
  "intervention": {
    "step": 8,
    "type": "scope_widen",
    "draft": "..."
  },
  "contaminated": false,
  "notes": "",
  "labeled_at": "2026-05-08T22:14:00Z",
  "elapsed_seconds": 612
}
```

`elapsed_seconds` is expected to be 5–15 minutes per trajectory at first;
it falls to 3–8 once the labeler has internal patterns for the modes.

---

## 9. Calibration and the intervention test

This is the rubric's main methodological departure from the chat rubric.

**Inter-rater κ is reported but is not the ceiling.** Two raters labeling
the same trajectory will disagree on subtle judgments — was msg 14 really a
"cue" the agent should have responded to? — and that disagreement is
tolerable as long as the labels predict where intervention helps.

**The intervention test is the primary quality signal.** For a labeled
intervention point, the corrector authors a message of the named type,
inject it at that step, and resume the trajectory in the original sandbox
(SWE-bench Docker images, deterministic). If the test gate flips
(`resolved=false → resolved=true`), the label was *correct*. If a different
labeler put the intervention 5 steps later and that *also* worked, both
labels are correct (intervention windows aren't unique).

The intervention test is expensive (one full agent re-run per labeled
trajectory) — typical budget is 1 run per labeled point per intervention
type. Budget for ~$5–20 per labeled trajectory at current API costs.

**Calibration items.** Same idea as the chat rubric: ~5 trajectories with
maintainer-set ground-truth labels embedded at random positions. If a
labeler's per-mode agreement with calibration drops below 70% (binary κ
proxy), pause and re-read the rubric.

**Warmup.** First three trajectories are walked through together by the
two primary annotators. Disagreements are discussed, the rubric is updated
if a real ambiguity surfaces, and the warmup labels are discarded.

---

## 10. What `coding_v0` is — and isn't — used for

**Used for:**

- Selecting trajectories for the intervention experiment (failed-but-solvable,
  with at least one labeled drift mode and a non-null intervention.step).
- Authoring corrector message templates per `intervention.type`.
- Training a step-level drift detector if the labeled set grows past ~200
  trajectories. Detectors target `intervention.step` — predict the step at
  which a corrector should fire — not the binary outcome.

**Not used for:**

- Ranking agents on SWE-bench. The leaderboard outcomes already do that.
- Defining what "drift" *is* in some absolute sense. This rubric defines
  what drift looks like *for the corrector to act on*. A trajectory with no
  drift modes labeled may still be "wrong" in some abstract sense — the
  rubric just says the corrector probably can't help it.

---

## 11. Known weaknesses (read once)

Some confirmed by the v0 pilot (n=7 LiveSWEAgent trajectories), some still
hypothetical.

- **Confirmed: `correct_sites_wrong_change` is the modal scope error.** v0
  was designed around `under_fix`/`over_fix` because that's what the first
  three cases showed. Pilot found 4/7 are right-place-wrong-change. The
  rubric's center of gravity has shifted in v0.1 (new `content_challenge`
  intervention type, scope-error definitions sharpened). Future revisions
  may want to subdivide `correct_sites_wrong_change` further: over-engineered
  vs subtle Boolean bug vs scope-too-narrow-on-bug-class.
- **Confirmed: two v0 modes had no per-case diagnostic value.** Both
  `pr_author_test_unsimulated` (7/7) and `custom_tool_tax` (7/7) fire on
  every diagnosable trajectory. Demoted to observations in v0.1.
- **Confirmed: `self_confirming_verification` was undertight in v0** (also
  7/7 by the loose criterion). v0.1 requires demonstrable miss-coverage of
  the actual `FAIL_TO_PASS` test. Expected to drop to ~60–70% under the
  tighter rule; if it's still >85% after a 30-trajectory pilot, tighten
  again or demote.
- **Confirmed: lock-in steps span a wide range** (msgs 2–26 in pilot).
  "First THOUGHT commits to a fix" is too simple a signature — some agents
  explore for 20+ messages before locking in. Detectors built on this need
  to look at *cumulative* commitment, not just the first message.
- **Confirmed: same-file-different-function disambiguation matters.**
  django-15732 needed the v0.1 site definition (file + function +
  behavioral effect) because the agent and ground truth patched different
  helpers in the same file.
- **Hypothetical: lock-in cue detection is rater-judgment-heavy.** "When
  should the agent have revisited?" requires reading the trajectory closely.
  Two raters may pick different `cue_step` values. The intervention test
  validates the *step*, not the rater's reasoning, so this is tolerable —
  but expect lock-in cue_step κ in the 0.4–0.6 band, not higher.
- **Confirmed (sample size 1): drift pool is small.** With LiveSWEAgent as
  target and 3 peer agents, the failed-but-solvable pool was 28 tasks. To
  reach ~200 labeled drift cases, you need either more peer agents (5–10)
  to grow each agent's failed-but-solvable count, or expand from Verified
  (500) to full SWE-bench (~2300) and SWE-bench Live for fresh tasks.
- **Hypothetical: outcomes carry incomplete signal.** SWE-bench gives a
  binary `resolved`, but the *kind* of failure (FAIL_TO_PASS only vs also
  PASS_TO_PASS regression) carries information the rubric doesn't yet use.
  django-11433 broke a pre-existing test in addition to failing the new
  one — that suggests a different drift mode than a clean "missed the new
  case" failure. v0.2 may want a `breaks_pass_to_pass` field as a separate
  observation.
- **Hypothetical: trajectory-format heterogeneity hurts cross-agent
  rubric application.** Step indices in LiveSWEAgent (mini-swe-agent) and
  Sonar (block-based) point to different things. Labels are not directly
  comparable across formats. The labeling tool should normalize step
  indices to "agent action" granularity (one numbered step per agent
  intent: thought + tool call, regardless of how it's serialized).
- **Hypothetical: subtle bugs evade rubric.** django-13925's failure shape
  (subtle Boolean structure that gives an extra warning on inheritance)
  isn't well captured by any drift mode; it was filed as
  `correct_sites_wrong_change` but the *why* is unclear from trajectory
  inspection alone. The intervention test should validate that the labeled
  step is the right intervention point — if it isn't, that's signal the
  rubric is missing a category.

A rubric is a contract between annotators *and* a contract with the
intervention experiment. v0.1 reflects what 7 trajectories taught us. The
next revision should be triggered by either (a) crossing 30 labeled
trajectories, or (b) the first batch of intervention-test results landing.
Whichever comes first.
