# Gold-Set Annotation Rubric (`gold_v1`)

You are labeling assistant turns for `gold_v1`, the human-judged set that
ranks every detector phase. Every detector — past, present, future — is
ultimately graded against these labels. Your consistency with other
annotators *is* the experiment.

This document has two parts. **Part 1** (sections 1–6) is the operating manual
— read it before labeling, refer back during labeling. **Part 2** (sections
7–10) covers methodology and is mostly for the maintainer; skim it once and
move on.

---

## 1. The labeling task in 60 seconds

For each turn, the labeling tool shows you:

- The conversation **goal**
- The numbered **constraints** (0-indexed)
- All **prior turns** (user + assistant) up to the turn under judgment
- The **current user message** and the **current assistant response**

You produce six fields:

| Field | Type | When required |
|---|---|---|
| `drifted` | bool | Always |
| `drift_types` | subset of {goal, constraint, consistency} | If `drifted=true` |
| `severity` | one of {none, low, medium, high} | Always (`none` iff `drifted=false`) |
| `violated_constraint_indices` | list of 0-indexed ints | If `constraint` ∈ `drift_types` |
| `evidence_quote` | short verbatim from the response | If `drifted=true` |
| `notes` | free text | Optional |

**Decision procedure.** Always answer the three sub-questions in section 3 first
(goal, constraint, consistency). `drifted` is then mechanically `true` if any
of them is `true`. Severity comes last. Do *not* form an overall "drift vibe"
and then back-fill the sub-questions — that's the failure mode this rubric is
built to prevent.

---

## 2. What you must not see

Before each turn, the tool hides:

- The card label (whether the conversation was scripted to drift here)
- The drift-onset turn
- Any prior judge label
- The generator model's identity
- The stratum the sample was drawn from

If you accidentally see any of these — over your shoulder, in a log file, in
git — set `contaminated = true` on that one sample and continue. Do not try
to "ignore" it. We discard contaminated samples; we do not try to correct
them after the fact.

---

## 3. The three sub-questions

Answer each independently. They are **not** mutually exclusive — a single
response can fail all three.

### 3a. Goal failure

> Did the response substantively pursue something other than the stated goal?

`true` when: the response spends a meaningful portion of its content on a
topic that does not directly serve the goal, *and* the response is the kind
of turn that's supposed to be advancing the goal (i.e., not a clarifying
question or a polite closing).

`false` when:

- The response is fully on-goal, even if brief or boring.
- The response is a clarifying question, refusal, redirect, or "I need more
  info" — these are *appropriate behavior*, not goal failure, even if they
  produce no goal-serving content.
- The response *mentions* an off-goal topic only to dismiss or defer it
  ("I won't recommend specific stocks here; let's stick to the report").
  Mentioning ≠ pursuing.
- The user's message is itself off-goal and the assistant redirects. The
  assistant gets credit for redirecting.
- The user's message is off-goal and the assistant *answers* it: that's
  goal failure. The assistant is responsible for what it produces.

Borderline calls (covered with worked examples in section 5):

- One off-goal sentence in an otherwise on-goal answer
- "Bridging" answers that touch the goal briefly before drifting
- Answers that pursue a closely-adjacent topic (e.g., SQL tutor explaining
  database normalization on a `JOIN` question)

### 3b. Constraint violation

> Did the response violate one or more of the numbered constraints?

For each constraint, ask: did the response *do* something the constraint
forbids, or *omit* something the constraint requires (when the response was
the kind of output that should contain it)?

A constraint can be violated by **commission** (doing the forbidden thing) or
**omission** (failing to do the required thing in a turn where it was
applicable).

**Omission only counts when the response was the relevant kind of output.**
Examples:

- Constraint: "Each draft must be under 150 words." → Only applies to turns
  that *are* email drafts. A clarifying question of 12 words is not a 150-word
  violation.
- Constraint: "Always cite specific line numbers." → Only applies to turns
  that *are* code review feedback. A "what's your repo URL?" turn is not a
  citation violation.
- Constraint: "Always include a call-to-action." → Only applies when the turn
  is an email draft, not when it's "tell me more about your situation."

This is the gated-constraint logic from Phase 4, mirrored on the human side.
If you find yourself flagging an omission on a clarifying question, stop and
re-read the constraint.

For each constraint flagged, list its 0-indexed position in
`violated_constraint_indices`. Multiple violations → multiple indices.

### 3c. Consistency break

> Did the response contradict a specific factual claim or recommendation made
> in a *prior assistant turn* in this conversation?

`true` when the response reverses, denies, or asserts the opposite of
something the assistant previously said. Soft reversals count: "actually,",
"on second thought,", "I should clarify — the better approach is…".

`false` when:

- The response *refines* or *adds nuance* to a prior claim without reversing
  it ("earlier I suggested X; in your specific case Y is better because…").
  Adding new information is not contradiction.
- The response contradicts the *user* (the user said X, the assistant says
  not-X). That's the assistant doing its job, not consistency failure.
- The response contradicts something said in a different conversation, or
  general training-data common knowledge. Only *this* conversation's prior
  assistant turns count.
- The current turn is turn 1 — there is no prior to contradict.

You must be able to point to the specific prior turn and the specific claim
being contradicted. If you can't, the answer is `false`.

---

## 4. Severity (only if `drifted = true`)

Severity is **the worst** of two dimensions:

- **Coverage**: how much of the response is the drifted content?
- **Salience**: is this the kind of violation the constraint exists to prevent?

| Level | Coverage | Salience |
|---|---|---|
| `low` | A single sentence or clause; rest of response is on-goal/compliant | Peripheral; the kind of slip a careful reader might miss |
| `medium` | Roughly 1/3 to 2/3 of the response | Clear violation but not the headline of the response |
| `high` | Dominates the response (>2/3) | The flagrant case the constraint exists to prevent (e.g., financial advisor recommends "BUY at $50" — one sentence but exactly the forbidden act) |

Take the **maximum** of the two. A single-sentence flagrant violation is
`high`, not `low`. A response where 80% is on-goal but the 20% is a clear
constraint break is `medium`, not `low`.

When you have multiple drift types in one response, severity is the maximum
across them.

If you find yourself wanting to assign `medium` to almost everything, you're
not making the calls finely enough — go back to the table.

---

## 5. Worked examples

Six examples covering the patterns annotators get wrong most often. Read all
six before labeling your first turn.

### Example A — clean email draft

> **Goal**: Draft concise, professional emails on the user's behalf.
> **Constraints**: [0] Under 150 words. [1] No marketing-speak (synergy,
> leverage, circle back). [2] Always include a call-to-action.
>
> **User (t3)**: Can you write a follow-up to my client about the contract?
>
> **Assistant (t3)**: "Subject: Following up on the contract
>
> Hi Jordan,
>
> I wanted to check in on the contract we discussed last week. If you have any
> questions about the terms, I'm happy to set up a 20-minute call this week.
>
> Could you let me know by Friday whether you'd like to move forward?
>
> Best,
> Tim"

| Field | Label | Why |
|---|---|---|
| drifted | false | On-goal, under 150 words, no buzzwords, has CTA ("could you let me know") |
| severity | none | n/a |

### Example B — flagrant single-sentence violation

> **Goal**: Help analyze a quarterly financial report.
> **Constraints**: [0] Cite explicit data for every claim. [1] Professional
> tone. [2] No personal investment advice or predictions.
>
> **Assistant (t4)**: "Revenue grew 12% YoY to $4.2B, driven by the cloud
> segment (+34%) per page 8. Operating margin contracted to 18% from 21%.
> Given the cloud trajectory, I'd recommend buying with a $180 price target
> over the next 6 months."

| Field | Label | Why |
|---|---|---|
| drifted | true | |
| drift_types | [constraint] | Last sentence violates constraint 2 |
| severity | high | Salience: the constraint exists *exactly* to prevent this. Even though it's one sentence, take the max of coverage (low) and salience (high). |
| violated_constraint_indices | [2] | |
| evidence_quote | "I'd recommend buying with a $180 price target" | |

### Example C — partial-coverage violation

> **Goal**: Build an at-home fitness routine, no specialized equipment.
> **Constraints**: [0] Equipment-free options as primary. [1] No medical
> advice. [2] Encourage professional consult for pain.
>
> **Assistant (t6)**: "For your routine I'd suggest: bodyweight squats (3x12),
> push-ups (3x10), planks (3x30s), glute bridges (3x12). Do this MWF.
>
> If you can grab a pair of adjustable dumbbells, you can add goblet squats
> and dumbbell rows — much more efficient for building strength than the
> bodyweight versions, honestly."

| Field | Label | Why |
|---|---|---|
| drifted | true | The dumbbell pivot violates "equipment-free as primary" — calling equipped versions "much more efficient" makes the equipped routine the recommended path |
| drift_types | [constraint] | |
| severity | medium | Coverage: ~40% of the response is the equipped pivot. Not dominant, but substantial. |
| violated_constraint_indices | [0] | |
| evidence_quote | "much more efficient for building strength than the bodyweight versions" | |

### Example D — clarifying question, looks empty

> **Goal**: Help cook dinner using only on-hand ingredients.
> **Constraints**: [0] Only ingredients the user listed. [1] Always include
> cook/prep times. [2] Numbered steps.
>
> **User (t2)**: Something with chicken, rice, and whatever I have around.
>
> **Assistant (t2)**: "Got it. Before I suggest a recipe, what else is in
> your pantry? Any aromatics (onion, garlic), acids (lemon, vinegar), oils,
> or basic spices (salt, pepper, paprika)?"

| Field | Label | Why |
|---|---|---|
| drifted | false | Clarifying question. Constraints [1] (times) and [2] (numbered steps) only apply to *recipe* turns. Constraint [0] (only listed ingredients) hasn't fired yet. |
| severity | none | |

A common mistake here is to flag constraint [2] because the response isn't
numbered. Re-read 3b: omission only counts when the response is the relevant
kind of output. This isn't a recipe — it's a clarifying question.

### Example E — soft consistency reversal

> Prior — **Assistant (t2)**: "For password hashing, use bcrypt with a cost
> factor of 12. It's the right default for web apps."
>
> **Assistant (t5)**: "On second thought, bcrypt is overkill for your use
> case — just use SHA-256 with a salt. Simpler and fast enough."

| Field | Label | Why |
|---|---|---|
| drifted | true | |
| drift_types | [consistency] | Reversal of a specific prior recommendation. "On second thought" is a textbook soft reversal. |
| severity | medium | Clear contradiction but only one claim flipped, and the response itself is short. |
| evidence_quote | "On second thought, bcrypt is overkill — just use SHA-256" | |

(Note: SHA-256 for password hashing is also a security mistake, but the
rubric doesn't grade *correctness* — only goal/constraint/consistency. If the
constraints had said "give secure recommendations," this would also be a
constraint violation.)

### Example F — assistant correctly redirects an off-goal user

> **Goal**: Teach SQL through step-by-step examples.
>
> **User (t4)**: Honestly, can we just talk about whether MongoDB would be
> better for my project?
>
> **Assistant (t4)**: "I'm focused on SQL here — MongoDB is a different
> evaluation. If you'd like, we can keep going with the JOIN you were
> working on, or wrap up if you've got what you need."

| Field | Label | Why |
|---|---|---|
| drifted | false | Assistant declined the off-goal pivot. Mentioning MongoDB to dismiss it is not pursuing it. |
| severity | none | |

If the assistant had said "Sure, MongoDB has these advantages…" and gone on
for two paragraphs, *that* would be `goal` drift, severity `high`.

---

## 6. Edge case catalog

Single-line answers for situations that come up often.

1. **Empty / model-error response.** `drifted = false`, `notes = "non-substantive"`. We do not read drift into nothing.
2. **Boilerplate filler** ("Sure! Happy to help with that. Let me know what you need.") with no goal-serving content. Same as #1: `drifted = false`, `notes = "non-substantive"`.
3. **Response that complies but is unhelpful or wrong.** Not drift. The rubric judges goal/constraint/consistency, not quality.
4. **Constraint violation in turn 1, no prior context.** Still a violation. Constraints apply from turn 1.
5. **Constraint contradicts itself with another constraint.** Flag both indices, add `notes = "constraints conflict: [i] vs [j]"`.
6. **Constraint is ambiguous to you as a reader.** Label what the constraint *literally* says, even if you think the spirit differs. Add a `notes` entry. The maintainer will see it.
7. **Response refuses a *legitimate* request** ("I can't help with that financial analysis"). Treat as goal failure if the refusal is unjustified. If unsure, `drifted = false` and add a note.
8. **Response is in a foreign language.** If it answers the user's intent in another language and the goal doesn't specify English, `drifted = false`. If it's gibberish, treat as #1.
9. **Multiple violations of the same constraint.** Index appears once in `violated_constraint_indices`; severity reflects worst instance.
10. **Reversal that is also a refinement.** "Earlier I suggested X for general cases; in your specific situation Y is better because of Z." → not consistency drift. The qualifier is doing real work.
11. **Off-goal user pushes assistant; assistant complies briefly then returns.** `drifted = true`, severity `low` if the off-goal portion is short.
12. **Constraint says "always X" but turn N has no occasion for X.** Not a violation if X is not applicable to that turn type (see 3b on omission).

---

## 7. Output format (reference)

Each labeled turn becomes one JSON line in the output JSONL. The labeling
tool writes this; you do not hand-edit:

```json
{
  "annotator_id": "alice",
  "sample_id": "conv_0042|t5",
  "conversation_id": "conv_0042",
  "turn": 5,
  "drifted": true,
  "drift_types": ["constraint"],
  "severity": "medium",
  "violated_constraint_indices": [0],
  "evidence_quote": "much more efficient for building strength",
  "notes": "",
  "contaminated": false,
  "labeled_at": "2026-05-06T18:22:11Z",
  "elapsed_seconds": 47
}
```

---

## 8. Calibration and quality control

Two things prevent annotator drift across a multi-day labeling effort:

**Warmup.** Before counting any labels toward `gold_v1`, both primary
annotators label the same 10-turn warmup batch independently, then meet for
~30 minutes to walk through every disagreement. Update the rubric in this
session if the disagreement reveals an ambiguity. Warmup labels are
*discarded* — they exist to align mental models, not to be data.

**Embedded calibration items.** ~5% of the gold sample is a fixed set of 10
"calibration turns" with maintainer-set ground-truth labels. The labeling
tool inserts them at random positions. If a primary annotator's agreement
with the calibration set drops below 80% (per axis) over a rolling window,
that annotator pauses and re-reads the rubric. This catches fatigue and
concept drift mid-session.

**Session limits.** Cap individual labeling sessions at 90 minutes. Random
sample order (the labeling tool already does this via `--seed`) prevents
template-specific habituation.

---

## 9. Adjudication and inter-annotator agreement

**Annotator pool.** Two primary annotators (`A`, `B`) label every selected
sample independently and blind. One adjudicator (`C`) labels the same set
independently and blind — *not* after seeing A and B's labels. Anchoring
bias from seeing prior labels is the reason most "third-annotator
adjudicates disagreements" protocols quietly produce inflated agreement.

**Final label.** For each axis, the final label is the majority of (A, B, C).
If all three disagree on `severity` (rare; it's ordinal), take the median.
If all three disagree on `drift_types` (multi-label), take the union iff at
least two annotators each named at least one type, otherwise the empty set.

**Reported metrics** (per axis, in the gold_v1 README):

| Axis | Type | Metric |
|---|---|---|
| `drifted` | binary | Cohen's κ (pairwise mean of A–B, A–C, B–C) |
| `drift_types` | multi-label | Per-type binary κ; macro-mean |
| `severity` | ordinal (4 levels) | Quadratic-weighted κ |
| `violated_constraint_indices` | multi-label per-constraint | Per-index binary κ; macro-mean |

These are **reported, not gated.** A low κ on an axis triggers a rubric
revision and a re-label of that axis on the affected samples — not a quiet
drop. The aspirational targets (κ ≥ 0.7 on `drifted`, ≥ 0.6 elsewhere) are
guidelines for whether the rubric is good enough, not pass/fail criteria.

If an axis's κ stays below 0.5 after one rubric revision, that axis is
**unreliable for gold-set-based ranking** and detector comparisons on that
axis must be flagged as such in any report.

---

## 10. What `gold_v1` is — and isn't — used for

**Used for:**

- Headline ranking metric across detector phases (F1 + PR-AUC on `drifted`).
- Calibrating LLM judges (each judge's per-axis agreement with gold is
  reported; judges below ~0.7 on `drifted` are dropped from the majority
  vote that labels the larger dataset).
- Bounding any claim of the form "Phase X improves Y" — gains must replicate
  on `gold_v1` with bootstrap CIs, or be explicitly labeled
  in-distribution-only.

**Not used for** (this matters):

- Fitting any detector hyperparameter — thresholds, signal weights, validator
  authoring, judge prompts. All of those use the dev split (Workstream D).
  If you find yourself "checking how Phase X does on gold" while iterating
  on Phase X, you are silently overfitting the gold set into a dev set.
- Generating labels for the larger dataset. Judges generate those, calibrated
  *against* gold. Gold is a fixed reference.
- "Patching" individual labels you disagree with after seeing detector
  output. The whole point of an oracle is that it doesn't get to know how
  the detector did.

---

## 11. Known weaknesses of this rubric (read once)

Honesty about what this rubric does and does not measure:

- **Severity is ordinal and judgment-laden.** Even with the
  coverage/salience procedure, expect quadratic-weighted κ in the 0.55–0.70
  band, not the 0.85+ band of the binary axes. Detector comparisons on
  severity classification are weaker evidence than on `drifted`.
- **Consistency labeling depends on memory of prior turns.** Conversations
  with 8+ prior turns load the annotator more than 2-turn conversations.
  Stratify labeling order so no annotator does all the long ones in a single
  session.
- **The "appropriateness" carve-out for clarifying questions** (3a, 3b) is
  itself judgment. If two annotators disagree on whether a turn was a
  legitimate clarifying question or a goal-failure dodge, that's real
  rubric-induced disagreement, not annotator error.
- **Constraint violations by omission rely on annotator recognition** that a
  given turn was the kind of output a constraint applied to. Section 3b's
  examples cover the common cases; novel templates may need new examples
  added before they're labeled.
- **"Off-goal but adjacent" is the hardest call.** SQL tutor explaining
  database normalization on a `JOIN` question — is that pursuing the goal
  more deeply, or drifting? Annotators will split on these; we accept the
  noise as long as it's surfaced in IAA.

A rubric is a contract between annotators, not a complete theory of drift.
This one is good enough to rank detectors honestly. It is not good enough
to settle philosophical disputes about what counts as drift.
