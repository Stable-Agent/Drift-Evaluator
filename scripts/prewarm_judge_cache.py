#!/usr/bin/env python
# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Parallel prewarm of the LLM-judge cache.

Iterates every (response, constraints) pair in a dataset and calls
LLMJudgeConstraintSignal.measure() concurrently via a thread pool, populating
the on-disk cache. After this finishes, run_phase3.py becomes near-instant
because every measure() call hits the cache.

Requires:
  - Ollama serving locally (default http://localhost:11434)
  - Set OLLAMA_NUM_PARALLEL=4 (or higher) BEFORE starting `ollama serve`
    so the server actually handles concurrent requests in parallel.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=PROJECT_ROOT / "datasets" / "v1_synthetic")
    p.add_argument("--cache", type=Path,
                   default=PROJECT_ROOT / ".cache" / "llm_judge_llama3.1_8b.jsonl")
    p.add_argument("--model", type=str, default="llama3.1:8b")
    p.add_argument("--workers", type=int, default=4,
                   help="must be <= OLLAMA_NUM_PARALLEL on the server")
    p.add_argument("--report-every", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    from drift_evaluator.schema import load_dataset
    from drift_evaluator.ollama_client import OllamaClient, OllamaConfig
    from drift_detector.llm_judge_signal import LLMJudgeConstraintSignal

    client = OllamaClient(OllamaConfig(model=args.model))
    if not client.health():
        print(f"Error: Ollama not reachable or model {args.model!r} not installed",
              file=sys.stderr)
        return 1

    # Single shared signal instance so all threads share one cache object.
    signal = LLMJudgeConstraintSignal(
        client=client,
        model=args.model,
        cache_path=args.cache,
    )

    print(f"[prewarm] loading dataset from {args.dataset}", file=sys.stderr)
    convs = load_dataset(args.dataset)
    tasks = [
        (conv.id, turn.turn, turn.assistant, conv.constraints)
        for conv in convs
        for turn in conv.turns
    ]
    print(f"[prewarm] {len(tasks)} turns; "
          f"workers={args.workers}; cache={args.cache}", file=sys.stderr)

    started = time.time()
    completed = 0
    errors = 0

    def _do(task):
        conv_id, turn_n, response, constraints = task
        # measure() is cache-aware: returns instantly if all constraints cached.
        signal.measure(response, constraints)
        return conv_id, turn_n

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_do, t) for t in tasks]
        for fut in as_completed(futures):
            completed += 1
            try:
                fut.result()
            except Exception as exc:
                errors += 1
                if errors < 10:
                    print(f"[prewarm] error: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
            if completed % args.report_every == 0 or completed == len(tasks):
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (len(tasks) - completed) / rate if rate > 0 else 0.0
                print(f"[prewarm] {completed}/{len(tasks)}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min  errors={errors}",
                      file=sys.stderr)

    elapsed = time.time() - started
    print(f"[prewarm] done in {elapsed/60:.1f}min "
          f"({completed} completed, {errors} errors)", file=sys.stderr)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
