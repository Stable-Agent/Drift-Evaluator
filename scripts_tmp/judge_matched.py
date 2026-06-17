#!/usr/bin/env python
"""Experiment A: LLM judge on the 87 matched pairs, with strength decomposition.

Tests a DIFFERENT signal type than the (dead) structural features, on the
difficulty-controlled matched corpus. The decisive question: does the LLM's
within-problem ranking survive the strength decomposition, or does it ALSO
flip with agent strength (= reading agent identity, not drift)?

Reuses the prompt / claude-CLI call / parser from llm_judge_pilot.py.
Judges each trajectory once (cached, resumable), then pairs them up.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys, time, re, argparse
import numpy as np

MANIFEST = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs/matched_corpus/manifest.jsonl")
OUT = pathlib.Path("Drift-Evaluator/reports/judge_matched.jsonl")
CLAUDE_BIN = "/opt/homebrew/bin/claude"
STRONG = {"opus"}

PROMPT = """You are evaluating a coding agent's trajectory.

The agent was given a bug-fix task in a Python repo. The trajectory below contains its reasoning, tool calls, and tool results.

Your job: estimate the probability that this agent FAILED to produce a patch that passes the hidden test suite. Output a SINGLE number between 0.0 and 1.0 — nothing else. 0.0 = certain success; 1.0 = certain failure; 0.5 = equally likely.

Most coding agents on hard bugs make recoverable mistakes; default toward 0.5-0.7 unless you see strong evidence either way.

=== TRAJECTORY (last {n_chars} chars) ===

{trajectory_text}

=== END TRAJECTORY ===

Output ONLY a single number in [0.0, 1.0]. No prose."""

def load(p):
    obj = json.load(open(p))
    if isinstance(obj, dict):
        return obj.get("messages", []), (obj.get("info", {}) or {}).get("submission", "")
    return obj, ""

def msg_text(m):
    c = m.get("content", "")
    return c if isinstance(c, str) else (json.dumps(c) if isinstance(c, list) else "")

def fmt(messages, max_chars=30000):
    lines = [f"[{i}] {m.get('role','?')}: {msg_text(m)[:3000]}"
             for i, m in enumerate(messages) if msg_text(m)]
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full

_NUM = re.compile(r"\b(0?\.\d+|1\.0+|0\.0+|0|1)\b")
def parse(raw):
    if not raw: return None
    m = _NUM.search(raw)
    if not m: return None
    try:
        v = float(m.group(1)); return v if 0 <= v <= 1 else None
    except ValueError: return None

def call(prompt, model, retries=2):
    for a in range(retries + 1):
        try:
            r = subprocess.run([CLAUDE_BIN, "-p", "--model", model, prompt],
                               capture_output=True, text=True, timeout=120)
            out = r.stdout.strip()
            if r.returncode == 0 and out:    # empty stdout = rate-limit/overload; retry
                return out
            sys.stderr.write(f"  retry {a+1}: exit {r.returncode} empty={not out}\n")
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"  retry {a+1}: timeout\n")
        time.sleep(4 + a * 6)              # backoff between retries
    return None

def concord(diffs):
    diffs = np.array([d for d in diffs if d is not None])
    if len(diffs) == 0: return float("nan"), 0
    w = (diffs > 0).sum(); t = (diffs == 0).sum()
    return (w + 0.5 * t) / len(diffs), len(diffs)

def run_judge(model, workers=8):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for l in OUT.open():
            if l.strip():
                done.add(json.loads(l)["path"])
    todo = []
    for r in rows:
        for role in ("resolved", "failed"):
            p = r[role]["path"]
            if p not in done:
                todo.append((p, r["instance"], role))
    print(f"{len(rows)} pairs, {len(todo)} to judge (model={model}, workers={workers})", flush=True)
    lock = threading.Lock()
    fp = OUT.open("a")
    counter = {"n": 0}

    def work(item):
        p, inst, role = item
        msgs, _ = load(p)
        if not msgs:
            return
        raw = call(PROMPT.format(n_chars=0, trajectory_text=fmt(msgs)), model)
        sc = parse(raw)
        with lock:
            fp.write(json.dumps({"path": p, "instance": inst, "role": role,
                                 "score": sc, "raw": (raw or "")[:80]}) + "\n")
            fp.flush()
            counter["n"] += 1
            if counter["n"] % 10 == 0 or sc is None:
                print(f"  [{counter['n']}/{len(todo)}] {inst} {role} -> {sc}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fp.close()

def analyze():
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    sc = {json.loads(l)["path"]: json.loads(l)["score"]
          for l in OUT.open() if l.strip()}
    buckets = {"resolved_stronger": [], "same_tier": [], "resolved_weaker": []}
    all_d, st_d = [], []
    for r in rows:
        rs, fs = sc.get(r["resolved"]["path"]), sc.get(r["failed"]["path"])
        if rs is None or fs is None:
            continue
        d = fs - rs                      # >0 => judge ranks failed run higher (correct)
        all_d.append(d)
        rt, ft = r["resolved"]["tier"], r["failed"]["tier"]
        if rt == ft: st_d.append(d)
        rstrong, fstrong = rt in STRONG, ft in STRONG
        if rstrong and not fstrong: buckets["resolved_stronger"].append(d)
        elif fstrong and not rstrong: buckets["resolved_weaker"].append(d)
        else: buckets["same_tier"].append(d)

    ca, na = concord(all_d)
    cs, ns = concord(st_d)
    print(f"\n=== LLM judge within-problem concordance ===")
    print(f"  ALL pairs:        {ca:.3f}  (n={na})")
    print(f"  same-tier subset: {cs:.3f}  (n={ns})")
    print(f"\n=== strength decomposition (decisive) ===")
    print(f"{'bucket':>18}  {'n':>3}  {'concord':>8}")
    for k in ["resolved_stronger", "same_tier", "resolved_weaker"]:
        c, n = concord(buckets[k])
        print(f"{k:>18}  {n:>3}  {c:>8.3f}")
    print("\n  if resolved_weaker (failed=stronger agent) is <0.5, the LLM is")
    print("  reading agent strength too. if >=0.5, it sees behavioral drift.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.analyze_only:
        run_judge(args.model, workers=args.workers)
    analyze()

if __name__ == "__main__":
    main()
