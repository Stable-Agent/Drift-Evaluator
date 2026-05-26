# Resume Sonnet Labeling — Handoff

Companion to `RESUME.md`. Same protocol, same scripts, same 508 deterministic
samples — but the labeler is Sonnet 4.6 instead of Opus, producing a second
independent rater for the inter-rater agreement study.

Output file: `gold/gold_v1_sonnet.jsonl`. Sample selection (seed 20260506) is
identical to `gold_v1_opus.jsonl` so the two files line up 1:1 by `sample_id`.

## Why a fresh session per batch

Each blind-view read consumes ~2-5 KB of Claude's context. After ~130 labels
the session is too thick to keep going. The labels file is durable, so a new
session can pick up exactly where this one stopped.

## How to start each batch session

1. Open Claude Code in this project root (`Stable-Agent/Drift-Evaluator/`).
2. **Switch the session model to Sonnet 4.6**: `/model claude-sonnet-4-6`.
   This is the entire point — the labeling Claude must *be* Sonnet, not Opus.
   If you skip this you produce Opus-vs-Opus labels and the inter-rater
   ceiling number is meaningless.
3. Paste the prompt below verbatim as your first message.

## Resume prompt

```
Resume Sonnet labeling of gold/gold_v1_sonnet.jsonl from where it left off.
You ARE Sonnet 4.6 acting as a second independent rater for the inter-rater
study; do not reference Opus's labels, do not look at gold_v1_opus.jsonl.

Context:
- Task: apply docs/gold_set_rubric.md to assistant turns from datasets/v1_synthetic.
- Sample selection is deterministic (seed 20260506); 508 turns total, identical
  to the Opus selection so the two label files line up 1:1 by sample_id.

For each sample:
1. Get the next batch of N pending sample_ids:
   `python3 scripts/_resume_status.py 10 --annotator sonnet --out gold/gold_v1_sonnet.jsonl`
2. For each (conv_id, turn), load a *blind* view (no card label, no metadata):
   `python3 scripts/_blind_view.py datasets/v1_synthetic <conv_id> <turn>`
3. Apply the rubric mechanically:
   - Answer goal_failure / constraint_violations / consistency_break sub-questions
   - Derive `drifted` as the disjunction
   - Assign severity per §4 of the rubric (max of coverage and salience)
   - Quote evidence verbatim from current_assistant
4. Append records to gold/gold_v1_sonnet.jsonl via:
   `cat records.json | python3 scripts/_append_label.py gold/gold_v1_sonnet.jsonl`

Output schema for each record (matches gold_v1_opus.jsonl with annotator + meta swapped):
{
  "annotator_id": "sonnet",
  "sample_id": "<conv_id>|t<turn>",
  "conversation_id": "<conv_id>",
  "turn": <int>,
  "drifted": <bool>,
  "drift_types": [],                           // ["goal","constraint","consistency"] subset
  "severity": "none"|"low"|"medium"|"high",
  "violated_constraint_indices": [],
  "evidence_quote": "",                        // verbatim substring of current_assistant
  "notes": "",
  "contaminated": false,
  "_sonnet_meta": {
    "model": "claude-sonnet-4-6",
    "prompt_version": "rubric_v1",
    "parse_error": false,
    "subquestions": {
      "goal_failure": <bool>,
      "constraint_violations": [],
      "consistency_break": <bool>
    },
    "reasoning": "one short sentence"
  }
}

Hard rules from §1 of the rubric:
- drifted is the disjunction of the three sub-questions; do not set it independently.
- drift_types is the union, in canonical order ["goal","constraint","consistency"].
- If drifted=false: drift_types=[], severity="none", evidence_quote="".
- If drifted=true: severity is one of low/medium/high; evidence_quote is a non-empty
  verbatim substring of the current_assistant response.

Work in batches of ~10. After ~130 turns the session context will be too thick;
end the session and start fresh — the labels file is durable.
```

## When all 508 are done

From any session (Opus or Sonnet), run:

```
python3 scripts/interrater_report.py
```

This joins `gold/gold_v1_opus.jsonl` and `gold/gold_v1_sonnet.jsonl` on
`sample_id` and writes `reports/interrater_opus_vs_sonnet.md` with:

- Cohen's κ on binary drift (the "near human level" ceiling)
- Severity κ (quadratic-weighted, ordinal) plus a tier-collapsed κ on the
  Drift-Corrector's `gentle / structured / emergency` strategy buckets
- Per-type κ on goal / constraint / consistency
- Jaccard on `violated_constraint_indices` (drives the Structured-Anchor
  callout text in the corrector)
- Per-template binary-drift agreement
