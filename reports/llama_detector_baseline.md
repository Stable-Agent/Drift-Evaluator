# llama3.1:8b zero-shot drift-detection baseline

Evaluated **14** of 14 labels (parse errors: 0).
Reference labels: `gold/coding_v0_pilot.jsonl`
Model: `llama3.1:8b`

## Per-field agreement with rubric labels

| Field | Match rate |
|---|---:|
| `drifted` (binary) | 93% (13/14) |
| `scope_error` (exact) | 14% (2/14) |
| `intervention_type` (exact) | 14% (2/14) |
| `drift_step` (exact match) | 0% (0/14) |
| `drift_step` (within ±5) | 0% (0/14) |
| `drift_modes` (mean Jaccard) | 0.20 |

## Reading this

This is the **starting gap** an open-source uplift story would close. 
The 14 labels were produced by Claude Opus 4.7 against the v0.1 rubric. 
Llama3.1:8b without fine-tuning, given the same trajectories, agrees at the 
rates above. Closer to 100% on each field = less gap to close.

Frontier-model production deployments use Claude/GPT/etc. directly. 
This baseline tells us how much fine-tuning + larger label sets would 
have to lift llama3.1:8b to match.
