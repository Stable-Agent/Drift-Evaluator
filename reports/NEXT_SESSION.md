# Next session — resume Track 3 (grounded + validated)

**Rate limit clears at: 2026-05-12 01:00 PDT (1 AM)**

When the window resets, paste this single command:

```bash
.venv/bin/python Drift-Evaluator/scripts/run_pilot_batch.py \
  --instances django__django-14404,astropy__astropy-14365,sympy__sympy-15976,django__django-11433,django__django-13925,django__django-14053,django__django-15732 \
  --use-detector claude --ground-detector --validate-patch
```

It auto-skips definitive cases (django-14404 already ✓ under this config) and
stops at the first transient failure if the window is exhausted again.

## State as of 2026-05-11 21:58 PDT

Track 3 grounded + validated config status:

| Instance | Status |
|---|:-:|
| django-14404 | **✓ TRUE** (validated patch on attempt 2) |
| astropy-14365 | transient (rate-limit during retry; was ✓ under grounded-only earlier) |
| sympy-15976 | transient (detector previously said no-drift even with grounding) |
| django-11433 | transient |
| django-13925 | transient |
| django-14053 | transient |
| django-15732 | transient |

## How to check status

```bash
# Summarize all runs
.venv/bin/python Drift-Evaluator/scripts/pilot_status.py summary

# Outstanding for the target config
.venv/bin/python Drift-Evaluator/scripts/pilot_status.py outstanding \
  --instances django__django-14404,astropy__astropy-14365,sympy__sympy-15976,django__django-11433,django__django-13925,django__django-14053,django__django-15732 \
  --use-detector claude --ground-detector --validate-patch
```

## What was learned this session

- Track 3 (grounded detector + grounded generator) works end-to-end
- django-14404 ✓ via full production pipeline (Detector → Corrector → eval)
  using only the libraries, no manual labels
- astropy-14365 ✓ under grounded-only (no validation) — confirmed earlier
- Patch validation + retry was load-bearing for django-14404
  (first patch attempt failed apply-check; retry recovered)
- Two production bugs fixed in this session:
  1. eval_path resolution failed `relative_to` → status entry not written
  2. Run config not persisted in result.json → backfill couldn't recover
- Both fixed; future runs auto-write to pilot_status.jsonl correctly

## Open future-work (after Track 3 batch completes)

1. Once all 7 evaluated under grounded+validated, compute the headline
   flip rate. Hypothesis: 3-4 of 7 flip (django-14404 ✓ confirmed; astropy
   ✓ likely; sympy-15976 likely no-drift; others unknown).
2. Test ClaudeCodingDetector outcome-conditioning (the `outcome_failed`
   flag) impact when removed — could the detector predict drift on
   trajectories it doesn't know failed? This is the production "in-flight"
   mode test.
3. Open-source uplift: write the (trajectory, truth_sites) training-data
   prep script. Target: fine-tune llama3.1:8b on the 14+ gold labels to
   predict truth_sites. Currently: 0% drift_step ±5 accuracy zero-shot.
