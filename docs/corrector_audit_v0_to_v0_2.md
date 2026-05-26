# Corrector audit: v0.1 → v0.2 (2026-05-09)

`correct_sites_wrong_change` is the modal scope error in the v0.1 pilot
(8 / 14 labels), and it's where the corrector messages performed worst:
1/3 outcome flips on evaluable patches, plus the 28-regression incident
on django-14053. This audit reads each v0.1 corrector against its
trajectory, diagnoses *why* it succeeded or failed, and proposes v0.2
redrafts.

The audit itself doesn't run any LLM calls. It's pure analytical work
based on what we already have on disk. v0.2 should be retested when
budget clears.

---

## Per-case diagnosis (livesweagent_opus45 only — those have outcome data)

### django-11433 — worked, but for the wrong reason

**v0.1 corrector** (`hypothesis_broaden` at step 22):
> "Before locking in the 'compare cleaned_data to detect modification'
> approach, list 2-3 alternative interpretations of the bug. One
> alternative: the skip condition is over-broad because it skips even
> when cleaned_data has a non-empty value. **What's the simplest
> condition under empty_values semantics?**"

**Ground-truth fix:** `and cleaned_data[f.name] in form_field.empty_values`.

**Diagnosis:** *the corrector leaked the keyword.* Saying "empty_values
semantics" is naming the exact attribute the right fix uses. Of course
the model produced the right patch — we told it the answer. Doesn't
generalize.

**v0.2 redraft:**
> "Your fix runs `clean()` on the raw value to detect modification —
> that's a lot of code. Before keeping it, ask: is there a single
> attribute already on the form field that captures the meaning of
> 'this value should be treated as not provided'? If yes, the entire
> fix is one expression added to the existing skip condition. If no,
> justify each line of the re-clean approach."

Points at "single attribute exists" without naming it.

---

### django-13925 — malformed patch

**v0.1 corrector** (`pr_author_test` at step 36):
> "Before applying the fix, sketch the test the PR author would have
> added. There are at least two cases to cover: an inherited model whose
> parent has an EXPLICIT pk, and an inherited model whose parent has an
> AUTO-created pk. Does your fix produce the right warning count in BOTH?"

**Outcome:** patch was malformed — wouldn't apply.

**Diagnosis:** `pr_author_test` is a *reasoning* intervention. It asks
the model to think about test coverage. It doesn't help the model
produce a structurally correct diff. For one-shot mode (no iteration,
no edit/verify cycle), reasoning correctness ≠ syntactic correctness.

The deeper problem: the agent's restructured Boolean logic in
`_check_default_pk` is hard to express as a clean diff. The model needed
to articulate the *behavior change* clearly, not just identify test
cases.

**v0.2 redraft (switch type to `scope_narrow` + concrete walkthrough):**
> "Trace your fix step-by-step on this case: `class Child(Parent)` where
> `Parent` has auto-created pk. After your walk through `parent_link`
> chains, what's the value of `pk.auto_created` for Child? Does your
> fix return a warning here? Now ask: should the warning fire on
> *Child* specifically, or only on the ancestor that actually has the
> auto-created pk? Re-read the issue — the user complained about
> warnings on the *child* class, not the parent. Adjust your return
> path accordingly."

Specific concrete case + names the actual bug (warning attribution to
wrong class).

---

### django-14053 — applied but broke 28 PASS_TO_PASS tests

**v0.1 corrector** (`hypothesis_broaden` at step 18):
> "You're about to fix this as 'yield only if hash changed across
> passes.' Consider the alternative framing: 'yield each adjustable
> file exactly once, with its final state.' Which matches the issue
> text more literally? Sketch what each does on a file whose hash flips
> A→B→A across passes."

**Ground-truth fix:** hold adjustable files through all passes; yield
each ONCE at the end with its final hashed name.

**Outcome:** model went the right direction but produced a fix that
broke 28 existing tests.

**Diagnosis:** the corrector pointed correctly to the right framing. But
"yield each exactly once with final state" is hard to land in one shot —
it requires changing the *control flow* of `post_process`, not just the
yield condition. Without iteration / test feedback, the model produced a
draft that was structurally close but had subtle ordering bugs that
broke `PASS_TO_PASS` cases.

This is partly a *one-shot mode artifact* — agent-loop would have run
tests and noticed regressions. But it's also a corrector design issue:
"consider the alternative framing" is too abstract for a one-shot fix.

**v0.2 redraft (more prescriptive):**
> "Reread the issue: 'returns the same original filename multiple times.'
> The simplest implementation: separate adjustable files from
> non-adjustable. Yield non-adjustable immediately during the first
> pass. For adjustable files, *do not yield during any intermediate
> pass* — accumulate them in a dict keyed by filename with the latest
> hashed name. After all passes complete, iterate the dict once and
> yield each adjustable file with its final hash. Translate this
> algorithm directly to code; do not modify any non-adjustable yield
> behavior."

Spells out the algorithm step-by-step. Less open-ended; more landable
in a single shot.

---

### django-15732 — applied but didn't resolve

**v0.1 corrector** (`scope_widen` at step 26):
> "You've identified primary_key as one duplicate-constraint case.
> Before patching, list all duplicate-constraint scenarios where
> `_delete_composed_index` could match >1 constraint: (a) PK overlap
> with unique_together, (b) field-level unique=True with
> unique_together, (c) multi-column unique with index_together. Which
> does your fix cover?"

**Ground-truth fix:** *post-filter by name suffix in the helper*
(`_uniq` for unique_together, `_idx` for index_together) — moving the
fix from the call site to the resolution layer.

**Outcome:** model produced a patch at the call site that filters
`primary_key=False`. Failed FAIL_TO_PASS.

**Diagnosis:** the corrector accepted the agent's *site choice* (the
call site) and tried to widen the fix WITHIN that site. But the right
move was to relocate the fix to the helper level. The corrector
reinforced a wrong-site framing.

This is the most instructive failure: the rubric labeled this as
`correct_sites_wrong_change` but it's arguably more `wrong_site` —
agent and ground-truth fixed in the same file but at different
helpers. The label needs revisiting.

**v0.2 redraft (redirect site, name target):**
> "Your fix filters constraints at the call site of
> `_delete_composed_index`. That handles only one specific overlap
> (primary_key). The duplicate-constraint problem is broader — the
> *resolution* helper itself is what returns ambiguous matches. Look
> at where constraint names are *resolved* (search for the
> `_constraint_names` helper). The general fix lives there: when
> multiple constraints match, post-filter by the constraint name's
> suffix. Constraints from `unique_together` end in `_uniq`; from
> `index_together` they end in `_idx`. Move the fix to the helper
> level."

Names the target helper; tells the model to *relocate*, not just
modify.

---

## Patterns identified

Four design problems show up across these cases:

### 1. Answer leakage

The v0.1 prompts often name the literal solution keyword
("empty_values semantics", "_uniq suffix"). When that happens, success
attributes credit to the corrector that should attribute to the
prompt. **Fix:** point at the *shape* of the answer ("a single
attribute on the field", "the constraint's name suffix") without
naming it.

### 2. Direction without execution

`hypothesis_broaden` correctly redirects but doesn't help the model
execute. In one-shot mode, the model can't iterate. **Fix:** for
one-shot, follow up "consider this framing" with a concrete algorithmic
prescription. For agent-loop, hold the corrector at the directional
level and let iteration fill in details.

### 3. Wrong-site reinforcement

`scope_widen` and `content_challenge` both implicitly *accept* the
agent's site choice and iterate within it. When the right move is to
relocate to a different site, neither type fits. **Fix:** introduce a
new intervention type — `relocate_fix` — that explicitly tells the
agent "your site is wrong, look at this other helper."

### 4. Reasoning interventions can't fix patch shape

`pr_author_test` is a reasoning prompt. It improves the model's
*understanding* but not the *structure* of the proposed diff. In
one-shot mode, that's a real problem. **Fix:** pair `pr_author_test`
with an explicit "now write the simplest patch that passes the test
you sketched" instruction.

---

## Two-tier corrector strategy (open question)

The audit reveals a tension between two deployment contexts:

- **Frontier + agent-loop (production):** v0.1-style open-ended
  questions are probably fine. Frontier models with iteration can take a
  directional hint and refine.
- **One-shot validation (the cheap sanity check):** the model needs
  prescriptive correctors. v0.2 drafts above lean prescriptive.

**Hypothesis:** Stable-Agent should ship two corrector libraries — one
for production frontier deployments (open-ended), one for cheap
intervention-test validation (prescriptive). Both serve the same labels
but with different verbosity profiles.

**Test:** run the v0.2 prescriptive correctors on the same 4 cases in
one-shot mode. Compare flip rates against v0.1. If v0.2 ≥ 75% flip rate
on the 4 cases, two-tier strategy is validated.

---

## v0.2 corrector summary table

| Case | v0.1 type | v0.2 type | v0.2 character |
|---|---|---|---|
| django-11433 | hypothesis_broaden | hypothesis_broaden | non-leaky version of v0.1 |
| django-13925 | pr_author_test | scope_narrow + concrete walkthrough | name the misattribution |
| django-14053 | hypothesis_broaden | hypothesis_broaden + algorithm | spell out the algorithm |
| django-15732 | scope_widen | **relocate_fix** (new type) | name target helper |

The introduction of `relocate_fix` as a fifth intervention type would
need to be added to `coding_rubric_v0.md` §5b.

---

## v0.2 test results (partial — n=1 of 4 due to rate-limit)

Ran the audited v0.2 correctors on the same 4 cases. 3 of 4 hit the
Claude Code rate-limit window after 3 retries; only django-11433
completed.

### django-11433 v0.2 result

- v0.1 (leaky): `resolved=TRUE`. Model wrote the 17-line re-clean
  approach, which happens to pass.
- v0.2 (non-leaky): `resolved=FALSE`. Model wrote a *simpler* 1-line
  guard `if f.name in form.fields and ...` — different (and wrong)
  attribute.

**This is the most important single result of the audit.** The v0.2
prompt asked the model to find "a single attribute that distinguishes
no-input from real-input." It picked `f.name in form.fields` —
structurally what was asked for (one attribute, simple expression) but
semantically the wrong one. The right answer was `empty_values`, which
v0.1's leakage handed it on a plate.

**Updated hypothesis (v0.3):** for *one-shot mode*, correctors must
explicitly name the answer keyword. The model can follow shape
prescriptions ("write a simple expression") but cannot reliably pick
the *correct* attribute from among similar candidates without
explicit naming.

For *agent-loop mode*, this is probably less of an issue — the model
can try `f.name in form.fields` first, run tests, see it fail, and
iterate to `empty_values`. We don't have agent-loop budget to test
this right now, but it's the natural next experiment.

### Full v0.2 results (n=4 of 4)

| Case | v0.1 result | v0.2 result | Net change |
|---|---|---|---|
| django-11433 | **TRUE** (leaky) | FALSE | regression — leak was load-bearing |
| django-13925 | malformed | **TRUE** | improvement — concrete trace + attribution |
| django-14053 | FALSE (28 P2P regressions) | malformed | swapped failure modes |
| django-15732 | FALSE | FALSE | relocated to helper, wrong discriminator |

**Headline: v0.1 and v0.2 both flip 1/3 evaluable cases. Different cases.**

### What v0.2 actually showed

The two corrector libraries don't dominate each other. They have different
strengths:

- **django-11433** (one-line conditional fix, attribute named `empty_values`):
  v0.1's keyword leak is necessary. v0.2's "find the right attribute"
  framing led the model to a *different* simple attribute (`f.name in
  form.fields`) — structurally what was asked for but semantically wrong.
  *Generalization:* one-shot can't reliably pick the right specific keyword
  from similar candidates without leakage.
- **django-13925** (warning attribution, restructured Boolean): v0.2's
  concrete trace ("walk this case in detail; what's `pk.auto_created` for
  Child?") helped where v0.1's `pr_author_test` reasoning didn't. The
  model needed *symbolic execution guidance*, not test-design framing.
  *Generalization:* for cases where the bug is in *attribution*
  (which class, which value gets the warning), prescriptive symbolic
  walkthroughs work better than test-design prompts.
- **django-14053** (yield deduplication): v0.1 went the wrong direction
  (yield-on-change), produced a fix that broke 28 tests. v0.2 went the
  right direction (yield-once-final-state) but the patch was malformed —
  the algorithm is hard to land in a single diff without iteration.
  *Generalization:* this case may not be one-shot solvable at all.
  Agent-loop is probably the right answer.
- **django-15732** (constraint resolution): v0.2's `relocate_fix`
  successfully redirected the model to the helper (a structural win)
  but the model picked a different discriminator than the ground truth
  expects. *Generalization:* `relocate_fix` works at the *redirect* layer
  but doesn't guarantee the right detail-level fix.

### What this means for corrector design

Both libraries hit a **1/3 ceiling** on the same 4 cases. That isn't a
v0.1-vs-v0.2 problem — it's a deeper one:

1. The 1/3 isn't random. v0.1 succeeds where v0.2 fails (django-11433),
   and vice versa (django-13925). Per-case wording matters more than
   library style.
2. **The corrector design problem isn't single-axis (verbose ↔ terse).**
   It's per-bug. Different bugs need different verbal interventions —
   keyword leakage for some, symbolic walkthroughs for others, structural
   relocation for others.
3. There's a hard ceiling for *one-shot* mode. Cases like django-14053
   require iteration; no wording will land them in a single diff. Don't
   try to make every case one-shot solvable.

### Updated v0.3 hypothesis

Stop searching for a universal corrector style. Instead:

- **Match corrector type to bug shape.** The rubric's `intervention.type`
  field already does this in principle — the audit reveals it's the right
  unit, just under-specified. Each `(scope_error, intervention_type)` pair
  needs a small set of *template patterns* (1-3 wordings) that the
  labeler picks from based on the specific case.
- **Acknowledge a one-shot ceiling.** Some cases are agent-loop-only.
  Label them as such — don't waste evaluation runs trying to one-shot them.
- **Test v0.1 in agent-loop on django-11433 + django-15732 + django-14053
  when budget clears.** That's the experiment that disambiguates
  "one-shot mode artifact" from "rubric problem."

---

## Updated recommendations

The original v0.2 hypothesis ("prescriptive but non-leaky correctors
help one-shot mode") is **disconfirmed by django-11433**. New picture:

1. **For one-shot validation (cheap budget-friendly mode):** keep the
   v0.1 leaky-prescriptive correctors. They're not artistically clean
   but they work.
2. **For agent-loop / production:** the v0.1 correctors are probably
   fine. Test when budget clears.
3. **Don't try to remove leakage from the corrector library.** It
   appears load-bearing for one-shot validation. The leakage is the
   work the model can't do on its own.
4. **The right question isn't "v0.1 vs v0.2" but "what's the
   minimum keyword that lifts one-shot to ≥75%?"** That's a v0.3
   experiment when budget clears.

---

# Audit 2: `under_fix` correctors (n=2)

Smaller pool than `correct_sites_wrong_change` — only two `under_fix`
labels in the pilot. But the comparison is unusually clean: same scope
type, same intervention type (`scope_widen`), one worked, one didn't.

| Case | Outcome | Sites | Distance |
|---|:-:|---|---|
| django-14404 | **TRUE ✓** | both in `catch_all_view` (one function) | adjacent lines |
| astropy-14365 | FALSE | regex in `_line_type` line 71; literal `==` in `_get_tables_from_qdp_file` line 309 | different functions, ~240 lines apart |

## v0.1 corrector text comparison

**django-14404 (worked):**
> "You're about to change `path = '%s/' % request.path_info`. The
> variable `path` is used in two places below — `resolve(path, ...)`
> and `HttpResponsePermanentRedirect(path)`. Do both uses want the
> same value?"

**astropy-14365 (failed):**
> "You noticed `command[1].lower()` already exists in the downstream
> code — that means *some* code in this module already case-handles.
> Before fixing the regex, list every place where the QDP parser does
> case-sensitive comparison. Which need IGNORECASE; which need
> `.upper()`?"

## What's different

Three observable differences:

### 1. The django corrector *names both sites by name*

`resolve(path, ...)` and `HttpResponsePermanentRedirect(path)` are quoted
explicitly. The agent doesn't have to find anything — both targets are
in the prompt.

The astropy corrector says "list every place where the QDP parser does
case-sensitive comparison." The model has to *enumerate* a list. That's
a harder cognitive task than "modify these two specific calls."

### 2. The sites' physical distance

In django-14404, both uses appear within ~5 lines of each other in the
same function. The agent has them visually adjacent in any file view.

In astropy-14365, the missed site (line 309) is in a different function,
~240 lines from the regex (line 71). The agent had viewed the regex
function but never read the row-parser. The corrector didn't name the
function it was hiding in.

### 3. The astropy corrector's "noticed" hint is misleading

The corrector says "You noticed `command[1].lower()` already exists" as
if to provide a *cue*. But `command[1].lower()` is the *correctly-handled*
case (the agent already saw it and concluded existing case-handling was
sufficient). Anchoring the model on something it already saw and
dismissed sends it back to the wrong well.

## Generalization: when does `scope_widen` work?

Across these two cases, `scope_widen` succeeds when:

1. The corrector **names the missed site explicitly** (function name,
   approximate line, or quoted code at that site).
2. The missed site is **physically close** to where the agent is
   currently working — same function, or named function the agent has
   already opened.
3. The corrector does **not anchor on existing observations** the
   agent already dismissed.

When any of these breaks, `scope_widen` collapses into "go find
something" — and one-shot models don't reliably search.

## Proposed v0.2 corrector for astropy-14365

> "Your `re.IGNORECASE` flag fixes the line-type detection regex. But
> once a line is detected as data, the row parser at
> `_get_tables_from_qdp_file` (around line 309) has `if v == 'NO':` —
> a literal Python comparison that's still case-sensitive. That's a
> second site requiring a parallel fix. Make the literal comparison
> case-insensitive too."

This:
- Names the function (`_get_tables_from_qdp_file`).
- Quotes the exact missed code (`if v == 'NO':`).
- States the relationship (parallel fix; same kind of mistake).
- Doesn't reference `command[1].lower()` (the misleading hint).

## Cross-audit synthesis: a corrector design principle emerges

Combining the `correct_sites_wrong_change` and `under_fix` audits, a
single principle covers most of the failed correctors:

> **Don't ask the model to find what the rubric already knows.**

The rubric's `truth_sites` field has the missed sites and their
locations. The corrector should *use that field directly* — quoting
the missed site, naming its enclosing function, optionally even
showing 2-3 lines of context.

This contradicts the v0.1 design impulse, which favored "questions
that prompt the model to discover the answer." That's elegant for
agent-loop where the model has time to search. It's a poor fit for
one-shot, where every cycle of search-and-find is a wasted attempt.

A v0.3 corrector library that *templates from the label's `truth_sites`
field* would be both more reliable on one-shot and easier to author
(less creative writing per case, more mechanical fill-in).

---

## Recommended next moves (ordered)

1. **Add `relocate_fix` to the rubric** (rubric edit, no LLM cost).
2. **Re-label django-15732 with the new type** (manual, no LLM cost).
3. **Re-run the 4 cases above with v0.2 correctors in one-shot mode**
   when budget clears. ~4 calls + 4 evals = ~$2-3.
4. **If v0.2 lifts flip rate to ≥3/4, declare v0.2 the validated
   corrector library** for one-shot validation.
5. **Don't change the v0.1 wording for production yet** — production
   uses agent-loop where v0.1 may be fine. Test v0.1 in agent-loop on
   the same 4 cases when budget clears, separately.
