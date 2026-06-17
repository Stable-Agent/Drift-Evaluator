# Pre-registration: corrector with deployable localization (FILE arm) — v1

**Frozen:** 2026-06-09, before any generation run against `ood_cases_v2.jsonl`.
Any deviation after this date must be logged in the **Deviations** section at
the bottom — not silently absorbed.

---

## 1. Background and motivation

The OOD corrector result (RESOLVE soft 0.625, full-fix 16/28 on post-cutoff
bugs — `project_corrector_content_v0`) has two known leaks:

1. **Oracle window.** The "original code" handed to RESOLVE was reconstructed
   from the gold diff's own hunks (`ood_source.py:gold_and_code`), i.e. the
   model received exactly the lines the real fix touched. Live systems cannot
   produce that localization (hybrid loop v0–v2 demonstrated this).
2. **Post-fix problem text.** Problem statements were PR title + body —
   written *after* the fix, frequently describing it. SWE-bench uses pre-fix
   issue text for exactly this reason.

Additionally, the LLM judge was verified against real tests only in the
NO direction (false negatives found, raising the estimate); the YES
direction is unverified.

This experiment removes all three at once and decides whether the corrector
is deployable.

## 2. Hypothesis

**H1 (deployability):** Given only pre-fix issue text plus the full pre-fix
content of the correct file (file-level localization — the fidelity failed
agents achieve live, 80.6% right-file per `project_truthsite_localization_v0`),
a fresh single-shot solve fixes a substantial fraction of never-seen bugs,
as measured by *executing the PR's own tests*.

**H2 (leak size):** The gold-hunk WINDOW arm outperforms FILE; the gap
quantifies how much of the prior 0.625 was the oracle window.

## 3. Conditions (paired, same bugs, same generator model)

| arm | problem text | code provided | localization fidelity |
|---|---|---|---|
| FULL | issue title+body (pre-fix) | none | none (floor) |
| **FILE** | issue title+body (pre-fix) | entire file at pre-fix commit | file-level = deployable |
| WINDOW | issue title+body (pre-fix) | gold-hunk window (current method) | oracle (ceiling) |

Generator: same model/prompt family as `corrector_content.py` (single-shot
SEARCH/REPLACE; one retry permitted iff the patch fails to *apply* —
mirrors the validated `--validate-patch` lesson; the retry rule applies
identically to all arms).

## 4. Case sourcing (`ood_source_v2.py`)

- Bug-fix PRs merged after 2026-02-15 (post training cutoff) in buildable
  repos: sympy, django, xarray, pylint, pytest, scrapy.
- **Must close a linked GitHub issue created before the PR was opened.**
  Problem statement = issue title + body only. PR body is never used.
- Issue body containing a diff/patch block → case excluded (manual screen
  during sourcing; logged).
- PR must change at least one test file (real, executable bug fix) and
  exactly one primary source file with gold diff ≤ 6,000 chars and pre-fix
  file ≤ 60 KB.
- Target: source ≥ 45 cases; expected evaluable after build attrition ≥ 30.

## 5. Outcome measurement

**Primary metric — execution:** candidate applies cleanly AND the test(s)
added/changed by the PR pass at the pre-fix commit with the candidate
applied. Binary per (case, arm). Apply-failure after the one retry counts
as **fail**, not attrition (application robustness is part of the system).

**Attrition (excluded from the paired analysis, fully logged):** repo/commit
fails to build or the gold patch itself fails its own tests in our harness —
environment problems, not corrector problems. A case is evaluable only if
the gold patch passes its tests at the pre-fix commit.

**Secondary — judge calibration:** every candidate is also rated by the
existing Haiku equivalence judge. Report the full judge × execution
confusion matrix, including the previously unmeasured YES-but-fails rate.

## 6. Decision rules (set now)

Let `p_arm` = execution pass rate over evaluable paired cases (n ≥ 30 target,
n ≥ 20 minimum to decide; below 20, outcome is "underpowered — extend
sourcing," not a verdict).

- **Corrector deployable:** `p_FILE ≥ 0.40` AND paired FILE−FULL difference
  > 0 with one-sided sign-test p < 0.05. → Build the product on
  file-level handoff.
- **Oracle artifact:** `p_FILE < 0.25` OR FILE not detectably better than
  FULL. → The 0.625 was the window leak; corrector thesis falls back to
  "fresh re-solve only" and the project pivots.
- **Intermediate** (0.25 ≤ p_FILE < 0.40 or sign-test inconclusive):
  directional only; extend n by +15 cases once, then take the rule's verdict.
  No second extension.
- **Leak size (H2):** report paired WINDOW−FILE with bootstrap 95% CI.
  No gate; this number is for honest reinterpretation of the prior result.

## 7. Leak audit (checked before the run)

Rule: *no experimental input may be derived from the fix, the outcome, or
anything authored after the fix.*

- [ ] Problem text comes from an issue **opened before the PR** — pre-fix. ✔ by construction
- [ ] FILE content fetched at the PR's **base commit** — pre-fix. ✔ by construction
- [ ] WINDOW arm is gold-derived **by design** (labeled oracle ceiling, never headlined as live performance).
- [ ] File *name* in FILE/WINDOW arms is gold-derived — accepted and disclosed: it proxies the 80.6% live right-file rate. (Sensitivity note: live file-level localization is right ~4/5 times; deployable estimate ≈ 0.8 × p_FILE.)
- [ ] Judge never decides the primary metric.
- [ ] All sourced cases run; none dropped after seeing results except per §5 attrition rules.

## 8. Reporting

One report (`reports/corrector_file_loc_v1.md`): per-arm pass rates with
95% Wilson CIs, paired comparisons with sign tests, attrition table (every
sourced case accounted for), judge confusion matrix, and the decision-rule
verdict stated verbatim. Negative or intermediate outcomes get the same
prominence as positive.

## Deviations

- **2026-06-09 (sourcing amendment, before any generation run):** django
  tracks bugs in Trac, not GitHub issues, so the "linked GitHub issue" rule
  excluded django entirely (first sourcing pass: 21 cases, 0 django).
  Amended: for django, the problem statement is the linked Trac ticket
  (parsed from the `Fixed #NNNNN --` PR title; ticket definitionally
  predates the fix PR; the Trac CSV `description` field contains only the
  reporter's original pre-fix report, not comments). Same diff/patch-block
  screen applies. Also widened search breadth (per-repo search limit
  40→100, per-repo keep cap 10→12) to reach the pre-stated n≥45 sourcing
  target; selection *criteria* unchanged. (A `\bfix\b` title-regex bug that
  silently excluded "Fixed …" titles was also corrected.)
- **2026-06-09 — CASE LIST FROZEN before any generation:** 34 cases
  (django 12, pylint 10, sympy 7, xarray 2, scrapy 2, pytest 1);
  `ood_cases_v2.jsonl` sha256 prefix `82669d48a8cac3a1`. Below the 45
  sourcing target — if evaluable n lands in [20, 30) the §6 minimum applies;
  the single +15 extension remains available per §6.
