# Resume Opus Labeling — Handoff

130 of 508 turns labeled into `gold/gold_v1_opus.jsonl`. Remaining: 378.

## Why a fresh session

Each blind-view read consumes ~2-5 KB of Claude's context. After ~130 labels
the session is too thick to keep going. The labels file is durable, so a new
session can pick up exactly where this one stopped.

## How to resume in a new Claude Code session

1. Open Claude Code in this project root (`Stable-Agent/Drift-Evaluator/`).
2. Paste the prompt below verbatim as your first message.

### Resume prompt

```
Resume Opus labeling of gold/gold_v1_opus.jsonl from where it left off.

Context:
- Task: apply docs/gold_set_rubric.md to assistant turns from datasets/v1_synthetic.
- Sample selection is deterministic; opus_label.py picks 508 turns.
- 130 are already done in gold/gold_v1_opus.jsonl. Resume with the next pending samples.

For each sample:
1. Get the next batch of N pending sample_ids:
   `python3 scripts/_resume_status.py 10`
2. For each (conv_id, turn), load a *blind* view (no card label, no metadata):
   `python3 scripts/_blind_view.py datasets/v1_synthetic <conv_id> <turn>`
3. Apply the rubric mechanically:
   - Answer goal_failure / constraint_violations / consistency_break sub-questions
   - Derive `drifted` as the disjunction
   - Assign severity per §4 of the rubric (max of coverage and salience)
   - Quote evidence verbatim from current_assistant
4. Append records to gold/gold_v1_opus.jsonl via:
   `python3 scripts/_append_label.py gold/gold_v1_opus.jsonl < records.json`

Output schema for each record (matches existing entries in the JSONL):
{
  "annotator_id": "opus",
  "sample_id": "conv_NNNN|tN",
  "conversation_id": "conv_NNNN",
  "turn": N,
  "elapsed_seconds": 0,
  "drifted": <bool>,
  "drift_types": [<subset of "goal","constraint","consistency">],
  "severity": <"none"|"low"|"medium"|"high">,
  "violated_constraint_indices": [<0-indexed ints>],
  "evidence_quote": <verbatim string from current_assistant; "" if not drifted>,
  "notes": "",
  "contaminated": false,
  "_opus_meta": {
    "model": "claude-opus-4-7-via-claude-code",
    "prompt_version": "via_chat_v1",
    "parse_error": false,
    "subquestions": {"goal_failure": <bool>, "constraint_violations": [<ints>], "consistency_break": <bool>},
    "reasoning": "<one sentence>",
    "usage": {}
  }
}

Process in batches of 10. After each batch, append to JSONL and report the new total. Continue until either all 378 remaining are done or context gets thick — whichever comes first. If context gets thick, stop after the current batch and tell me to start another fresh session with the same prompt.

Start by running `python3 scripts/_resume_status.py 10` to see what's next.
```

## Verifying state at any time

- Progress + next pending samples: `python3 scripts/_resume_status.py 20`
- Total labeled so far: `wc -l gold/gold_v1_opus.jsonl`
- Inspect a record: `tail -1 gold/gold_v1_opus.jsonl | python3 -m json.tool`

## Notes

- The script `opus_label.py` would do this automatically via the Anthropic API
  if you ever set `ANTHROPIC_API_KEY`. It uses the same JSONL format and the
  same resume semantics, so any in-Claude-Code work composes with it.
- `annotator_id` is `opus` for both this session and the API path. If you
  want to keep them distinguishable, change one before merging — but for
  current purposes they're the same model so it doesn't matter.
- If a future session disagrees with an earlier label, don't edit in place;
  add a new record with `annotator_id: "opus_v2"` so the divergence is
  auditable.
