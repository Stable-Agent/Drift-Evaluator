#!/usr/bin/env python
"""LLM-judge pilot: does Haiku 4.5 as a continuous failure-prob feature
add signal over the structural detector?

Pilot setup:
  - 50 failed + 50 resolved trajectories, stratified across the 4 sources
  - Truncate each trajectory to last ~30K chars (Haiku context budget)
  - Prompt asks for a calibrated probability of agent failure in [0,1]
  - Compute: univariate AUC of LLM score, joint precision/recall when
    AND-gated with the structural score
"""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import re
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score
from drift_detector import Detector

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
SOURCES = ["livesweagent_opus45", "livesweagent_gemini3",
           "sonar_opus45", "sonar_sonnet45"]

PROMPT_TEMPLATE = """You are evaluating a coding agent's in-flight trajectory.

The agent was given a bug-fix task in a Python repo. The trajectory below contains its reasoning, tool calls, and tool results so far.

Your job: estimate the probability that this agent will FAIL to produce a patch that passes the hidden test suite. Output a SINGLE number between 0.0 and 1.0 — nothing else. 0.0 means certain success; 1.0 means certain failure; 0.5 means equally likely either way.

Most coding agents on hard bugs make recoverable mistakes; default toward 0.5-0.7 unless you see strong evidence either way.

=== TRAJECTORY (most recent {n_msgs} messages, last {n_chars} chars) ===

{trajectory_text}

=== END TRAJECTORY ===

Output ONLY a single number in [0.0, 1.0]. No prose, no explanation."""


def msg_text(m):
    if "blocks" in m:
        parts = []
        for b in m.get("blocks") or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("block_type") or b.get("type")
            if bt == "text":
                parts.append(b.get("text") or "")
            elif bt in ("tool_use", "tool"):
                nm = b.get("name") or b.get("tool_name") or ""
                inp = b.get("input") or b.get("arguments") or ""
                parts.append(f"[{nm} {json.dumps(inp)[:500] if not isinstance(inp, str) else inp[:500]}]")
            elif bt in ("tool_result", "tool_output"):
                tr = b.get("content") or b.get("output") or ""
                if isinstance(tr, list):
                    tr = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in tr)
                parts.append(str(tr)[:2000])
        return "\n".join(parts)
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def load_traj(path):
    d = json.load(path.open())
    if isinstance(d, dict) and "messages" in d:
        return d["messages"], (d.get("info") or {}).get("submission") or ""
    if isinstance(d, list):
        return d, ""
    return [], ""


def format_trajectory(messages, max_chars=30000):
    lines = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        text = msg_text(m)
        if not text:
            continue
        lines.append(f"[{i}] {role}: {text[:3000]}")
    full = "\n".join(lines)
    if len(full) > max_chars:
        return full[-max_chars:]  # keep most recent
    return full


CLAUDE_BIN = "/opt/homebrew/bin/claude"


def call_haiku(prompt, retries=2, model="claude-haiku-4-5-20251001"):
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "-p", "--model", model, prompt],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                sys.stderr.write(f"  retry {attempt+1}: exit {result.returncode} stderr[:200]={result.stderr[:200]!r}\n")
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"  retry {attempt+1}: timeout\n")
        time.sleep(5 + attempt * 5)
    return None


_NUM_RE = re.compile(r"\b(0?\.\d+|1\.0+|0\.0+|0|1)\b")


def parse_score(raw):
    if not raw:
        return None
    m = _NUM_RE.search(raw)
    if not m:
        return None
    try:
        v = float(m.group(1))
        if 0.0 <= v <= 1.0:
            return v
    except ValueError:
        pass
    return None


def sample_dataset(rng, per_class_per_source=12):
    """Stratified sample: per_class_per_source of each (source, label) cell."""
    rows = []
    for src in SOURCES:
        for suffix, label in [("_held_out", 0), ("_held_out_failed", 1)]:
            d = TRAJ_ROOT / f"{src}{suffix}"
            if not d.exists():
                continue
            files = sorted(d.glob("*.json"))
            rng.shuffle(files)
            for f in files[:per_class_per_source]:
                rows.append({"path": f, "source": src, "y": label})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=13,
                    help="trajectories per (source, label) cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=30000)
    ap.add_argument("--out", default="Drift-Evaluator/reports/llm_judge_pilot.jsonl")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = sample_dataset(rng, per_class_per_source=args.per_cell)
    rng.shuffle(rows)
    print(f"sampled {len(rows)} trajectories "
          f"({sum(r['y'] for r in rows)} failed, {sum(1 for r in rows if r['y']==0)} resolved)")

    det = Detector()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: skip rows already in the output
    done = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                r = json.loads(line)
                done.add((r["source"], r["instance"]))
    fp_out = out_path.open("a")

    for i, r in enumerate(rows, 1):
        key = (r["source"], r["path"].stem)
        if key in done:
            continue
        msgs, final_patch = load_traj(r["path"])
        if not msgs:
            continue
        struct_score = det.score(msgs, final_patch=final_patch)
        traj_text = format_trajectory(msgs, max_chars=args.max_chars)
        prompt = PROMPT_TEMPLATE.format(
            n_msgs=len(msgs), n_chars=len(traj_text), trajectory_text=traj_text)
        t0 = time.time()
        raw = call_haiku(prompt, model=args.model)
        dt = time.time() - t0
        llm_score = parse_score(raw)
        rec = {
            "source": r["source"], "instance": r["path"].stem, "y": r["y"],
            "struct_score": struct_score, "llm_score": llm_score,
            "llm_raw": (raw or "")[:200], "elapsed_s": round(dt, 1),
        }
        fp_out.write(json.dumps(rec) + "\n"); fp_out.flush()
        marker = "OK" if llm_score is not None else "PARSE_FAIL"
        print(f"  [{i:>3d}/{len(rows)}] {r['source']:30s} {r['path'].stem:30s} "
              f"y={r['y']} struct={struct_score:.2f} llm={llm_score if llm_score is not None else 'n/a':>5} "
              f"({dt:.0f}s, {marker})", flush=True)
    fp_out.close()

    # Analysis
    recs = [json.loads(l) for l in out_path.open() if l.strip()]
    recs = [r for r in recs if r["llm_score"] is not None]
    if not recs:
        sys.exit("no valid LLM scores")
    print(f"\n=== Analysis on {len(recs)} valid scores ===")
    y = np.array([r["y"] for r in recs])
    llm = np.array([r["llm_score"] for r in recs])
    struct = np.array([r["struct_score"] for r in recs])

    print(f"struct AUC: {roc_auc_score(y, struct):.3f}")
    print(f"LLM AUC:    {roc_auc_score(y, llm):.3f}")
    # Combined via simple average
    combo = 0.5 * llm + 0.5 * struct
    print(f"avg(struct, llm) AUC: {roc_auc_score(y, combo):.3f}")

    print(f"\nLLM score distribution:")
    print(f"  failed:   mean={llm[y==1].mean():.2f} median={np.median(llm[y==1]):.2f}")
    print(f"  resolved: mean={llm[y==0].mean():.2f} median={np.median(llm[y==0]):.2f}")

    print(f"\n=== AND-gate: struct AND llm (both > threshold) ===")
    print(f"{'struct_thr':>10s} {'llm_thr':>8s} {'TP/n_fired':>11s} {'precision':>10s} {'recall':>7s}")
    n_failed = int(y.sum())
    for struct_thr in [0.4, 0.5, 0.6, 0.7]:
        for llm_thr in [0.5, 0.6, 0.7, 0.8]:
            joint = (struct > struct_thr) & (llm > llm_thr)
            tp = int((joint & (y == 1)).sum())
            fp = int((joint & (y == 0)).sum())
            n = tp + fp
            if n == 0: continue
            prec = tp / n
            rec = tp / max(1, n_failed)
            print(f"  {struct_thr:>8.2f}  {llm_thr:>6.2f}  {tp:>4d}/{n:<6d} {prec:>9.1%}  {rec:>6.1%}")


if __name__ == "__main__":
    main()
