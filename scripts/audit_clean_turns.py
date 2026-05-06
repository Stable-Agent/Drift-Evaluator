#!/usr/bin/env python
# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Sample audit of turns currently labeled drifted=False.

Goal: detect false negatives — turns the deterministic card calls "clean"
but the model actually drifted. If false-negative rate is low (<5%), the
asymmetric relabel approach (only re-checking drift=True turns) is fine.
If high, we need a full symmetric label rebuild.

Samples:
  - N1 turns from clean conversations (any drift type possible)
  - N2 pre-onset turns from non-clean conversations (only the conversation's
    drift_type is plausible, since the model wasn't instructed to drift in
    other ways)

For each sampled turn, runs the LLM judge with the strict v2 prompts and
records the verdict.

Output: report-only, no dataset changes. Prints disagreement rate per category.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=PROJECT_ROOT / "datasets" / "v1_synthetic")
    p.add_argument("--cache", type=Path,
                   default=PROJECT_ROOT / ".cache" / "audit_clean_judge_llama3.1_8b.jsonl")
    p.add_argument("--model", type=str, default="llama3.1:8b")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-clean", type=int, default=25,
                   help="number of turns to sample from clean conversations")
    p.add_argument("--n-pre-onset", type=int, default=25,
                   help="number of pre-onset turns to sample from non-clean convs")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "reports" / "audit_clean_turns.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Reuse helpers from relabel_dataset
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import relabel_dataset as rd  # type: ignore

    from drift_evaluator.schema import load_dataset
    from drift_evaluator.ollama_client import OllamaClient, OllamaConfig, ChatMessage

    client = OllamaClient(OllamaConfig(model=args.model))
    if not client.health():
        print("Error: Ollama not reachable", file=sys.stderr)
        return 1

    cache = rd._load_cache(args.cache)
    convs = load_dataset(args.dataset)

    # Build candidate pools
    clean_pool = []   # (conv, turn) — any clean-labeled turn from clean convs
    preon_pool = []   # (conv, turn, drift_type) — pre-onset turn of non-clean conv
    for conv in convs:
        dtype = conv.metadata.get("drift_type", "clean")
        onset = conv.metadata.get("drift_onset_turn")
        if dtype == "clean":
            for t in conv.turns:
                if not t.label.drifted:
                    clean_pool.append((conv, t))
        else:
            for t in conv.turns:
                if t.turn < (onset or 1) and not t.label.drifted:
                    preon_pool.append((conv, t, dtype))

    rng = random.Random(args.seed)
    clean_samples = rng.sample(clean_pool, min(args.n_clean, len(clean_pool)))
    preon_samples = rng.sample(preon_pool, min(args.n_pre_onset, len(preon_pool)))
    print(f"[audit-clean] sampled {len(clean_samples)} clean-conv turns, "
          f"{len(preon_samples)} pre-onset turns", file=sys.stderr)

    # Build tasks
    DRIFT_TYPES = ["goal_drift", "constraint_violation", "consistency_break"]
    KIND_BY_TYPE = {
        "goal_drift": "goal",
        "constraint_violation": "constraint",
        "consistency_break": "consistency",
    }
    DRIFT_VERDICT = {
        "goal_drift": "OFF_GOAL",
        "constraint_violation": "VIOLATES",
        "consistency_break": "CONTRADICTS",
    }

    tasks = []  # (sample_id, drift_type, identifier, payload)
    # For clean convs: check all 3 drift types
    # For constraint check: try each constraint, mark drift if ANY violate
    for conv, turn in clean_samples:
        sid = f"clean|{conv.id}|t{turn.turn}"
        # goal
        ident = f"{conv.id}|t{turn.turn}|goal"
        tasks.append((sid, "goal_drift", ident, ("goal", conv.goal, turn.assistant)))
        # consistency (only if has prior turns)
        priors = [t.assistant for t in conv.turns if t.turn < turn.turn][-3:]
        if priors:
            ident = f"{conv.id}|t{turn.turn}|cons"
            tasks.append((sid, "consistency_break", ident,
                          ("consistency", priors, turn.assistant)))
        # constraints: check the worst-case (any constraint violated)
        # We'll check just the first constraint for cost; if model violates, it
        # might violate any. (Could expand to all 3 if needed.)
        if conv.constraints:
            for ci, c in enumerate(conv.constraints[:3]):
                ident = f"{conv.id}|t{turn.turn}|c{ci}"
                tasks.append((sid, "constraint_violation", ident,
                              ("constraint", c, turn.assistant)))

    # Pre-onset turns of non-clean convs: only check the conv's drift_type
    for conv, turn, dtype in preon_samples:
        sid = f"preon|{conv.id}|t{turn.turn}|{dtype}"
        kind = KIND_BY_TYPE[dtype]
        if kind == "goal":
            ident = f"{conv.id}|t{turn.turn}|goal"
            tasks.append((sid, dtype, ident, ("goal", conv.goal, turn.assistant)))
        elif kind == "consistency":
            priors = [t.assistant for t in conv.turns if t.turn < turn.turn][-3:]
            if priors:
                ident = f"{conv.id}|t{turn.turn}|cons"
                tasks.append((sid, dtype, ident, ("consistency", priors, turn.assistant)))
        else:
            # constraint — use the target constraint from metadata
            target_idx = conv.metadata.get("target_constraint_index")
            if target_idx is not None:
                c = conv.constraints[target_idx]
                ident = f"{conv.id}|t{turn.turn}|c{target_idx}"
                tasks.append((sid, dtype, ident,
                              ("constraint", c, turn.assistant)))

    print(f"[audit-clean] {len(tasks)} judge calls to make", file=sys.stderr)

    def _judge(task):
        sid, dtype, identifier, payload = task
        kind = payload[0]
        # Use the v2 prompt-version key matching strict relabel
        cache_key = rd._key(args.model, f"{kind}_v2", identifier)
        if cache_key in cache:
            return sid, dtype, *cache[cache_key]
        if kind == "constraint":
            constraint, response = payload[1], payload[2]
            messages = [
                ChatMessage("system", rd.SYSTEM_PROMPT_CONSTRAINT),
                ChatMessage("user", rd._constraint_prompt(constraint, response)),
            ]
        elif kind == "goal":
            goal, response = payload[1], payload[2]
            messages = [
                ChatMessage("system", rd.SYSTEM_PROMPT_GOAL),
                ChatMessage("user", rd._goal_prompt(goal, response)),
            ]
        else:
            priors, response = payload[1], payload[2]
            messages = [
                ChatMessage("system", rd.SYSTEM_PROMPT_CONSISTENCY),
                ChatMessage("user", rd._consistency_prompt(priors, response)),
            ]
        try:
            raw = client.chat(messages, temperature=0.0, seed=42)
        except Exception as exc:
            return sid, dtype, None, f"<error:{type(exc).__name__}>"
        verdict, reason = rd._parse_verdict(raw)
        rd._append_cache(args.cache, cache_key, verdict, reason)
        cache[cache_key] = (verdict, reason)
        return sid, dtype, verdict, reason

    started = time.time()
    completed = 0
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_judge, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
            completed += 1
            if completed % 25 == 0:
                el = time.time() - started
                rate = completed / max(el, 1e-9)
                eta = (len(tasks) - completed) / max(rate, 1e-9)
                print(f"[audit-clean] {completed}/{len(tasks)}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min", file=sys.stderr)

    # Aggregate by sample
    sample_verdicts: dict[str, list[tuple[str, str, str]]] = {}
    for sid, dtype, verdict, reason in results:
        sample_verdicts.setdefault(sid, []).append((dtype, verdict, reason))

    # A sample is "actually drifted" if ANY of its judge calls returned the
    # drift verdict for that drift_type.
    DRIFT_VERDICT = {
        "goal_drift": "OFF_GOAL",
        "constraint_violation": "VIOLATES",
        "consistency_break": "CONTRADICTS",
    }
    actual_drift_count = 0
    actual_drift_examples = []
    for sid, calls in sample_verdicts.items():
        is_drift = any(v == DRIFT_VERDICT.get(dt) for dt, v, _ in calls)
        if is_drift:
            actual_drift_count += 1
            drifted_calls = [(dt, v, r) for dt, v, r in calls
                             if v == DRIFT_VERDICT.get(dt)]
            actual_drift_examples.append({
                "sample_id": sid,
                "drifted_judgments": [
                    {"drift_type": dt, "verdict": v, "reason": r}
                    for dt, v, r in drifted_calls
                ],
            })

    n_clean_samples = len(clean_samples)
    n_preon_samples = len(preon_samples)
    n_total = n_clean_samples + n_preon_samples
    fn_rate = actual_drift_count / max(n_total, 1)

    print(f"\n[audit-clean] Results:", file=sys.stderr)
    print(f"  Total clean-labeled turns audited: {n_total}", file=sys.stderr)
    print(f"    - from clean convs: {n_clean_samples}", file=sys.stderr)
    print(f"    - pre-onset of non-clean convs: {n_preon_samples}", file=sys.stderr)
    print(f"  Judge says these are actually drifted: {actual_drift_count}",
          file=sys.stderr)
    print(f"  False-negative rate: {fn_rate:.1%}", file=sys.stderr)
    threshold_pct = 0.05
    if fn_rate > threshold_pct:
        print(f"  ⚠️  False-negative rate exceeds {threshold_pct:.0%} threshold; "
              f"consider full label rebuild.", file=sys.stderr)
    else:
        print(f"  ✓ False-negative rate is below {threshold_pct:.0%}; "
              f"asymmetric relabel is acceptable.", file=sys.stderr)

    # Save report
    report = {
        "n_clean_samples": n_clean_samples,
        "n_preon_samples": n_preon_samples,
        "n_total": n_total,
        "actual_drift_count": actual_drift_count,
        "false_negative_rate": fn_rate,
        "examples": actual_drift_examples[:20],  # cap output size
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[audit-clean] report saved to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
