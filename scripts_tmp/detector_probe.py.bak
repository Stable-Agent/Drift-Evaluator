#!/usr/bin/env python
"""Drift-detection probe: can a SMALL OPEN model match a FRONTIER model at
detecting that a coding-agent trajectory is failing? Go/no-go on existing data.

Scores each of the multiseed trajectories with four detectors, predicting
FAILURE (positive = run did NOT resolve the bug):
  - structural : non-LLM features (n_steps, stuck, empty_patch, diff, rc-errors)
  - open_judge : local model P(fail) from the trajectory text
  - open_selfconsistency : std of P(fail) over K resamples (a WHITE-BOX signal a
                 frontier API can't cheaply give — uncertainty of the open model)
  - frontier_judge : frontier model P(fail), same text (the baseline to match)

Why this is a fair test despite a difficulty-confounded corpus: comparing two
JUDGES on the SAME trajectories cancels the confound (both face it), so
"open ≈ frontier?" is meaningful pooled. Within-cluster (same instance, mixed
pass/fail seeds) is the clean-but-thin check.

Resumable: LLM scores cached to reports/detector_probe_cache.jsonl.
Labels from eval_native.jsonl (execution-grounded).

Usage:
  python detector_probe.py --structural-only      # instant, no LLM
  python detector_probe.py --open-model qwen2.5-coder:14b --frontier-model sonnet
  python detector_probe.py --analyze-only
"""
from __future__ import annotations
import argparse, json, math, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path("Drift-Evaluator/datasets/multiseed_v1")
RUNS = ROOT / "runs.jsonl"
STEPS = ROOT / "steps"
EVAL = ROOT / "reports" / "eval_native.jsonl"
REPORTS = pathlib.Path("Drift-Evaluator/reports")
FRONTIER_CACHE = REPORTS / "detector_probe_cache.jsonl"   # shared frontier scores
OLLAMA = "http://localhost:11434/v1/chat/completions"

def cache_path(open_model: str) -> pathlib.Path:
    """Per-open-model cache so different judges don't overwrite each other."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", open_model)
    return REPORTS / f"detector_probe_cache__{safe}.jsonl"

def out_path(open_model: str) -> pathlib.Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", open_model)
    return REPORTS / f"detector_probe__{safe}.json"


def load_corpus():
    """-> list of {iid, seed, failed(bool), text, struct{...}}"""
    labels = {(r["instance_id"], r["seed"]): r["resolved"]
              for r in (json.loads(l) for l in open(EVAL) if l.strip())
              if r.get("resolved") is not None}
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        problems = {r["instance_id"]: r["problem_statement"] for r in ds}
    except Exception:
        problems = {}
    rows = []
    for r in (json.loads(l) for l in open(RUNS) if l.strip()):
        key = (r["instance_id"], r["seed"])
        if key not in labels:
            continue
        sp = STEPS / f"{r['instance_id']}__s{r['seed']}.json"
        steps = json.loads(sp.read_text()) if sp.exists() else []
        n = len(steps)
        rc_err = sum(1 for s in steps if s.get("rc") not in (0, None))
        struct = {
            "n_steps": n,
            "stuck": int(any(s.get("stuck") for s in steps)),
            "empty_patch": int(r["empty_patch"]),
            "final_patch_lines": r["patch"].count("\n"),
            "max_diff_lines": max((s.get("cum_diff", "").count("\n") for s in steps), default=0),
            "rc_err_frac": rc_err / n if n else 1.0,
            "secs": r.get("secs", 0),
        }
        # compact text the judges see: problem + action trail + final patch
        trail = "\n".join(
            f"[{s.get('action','?')}] {s.get('cmd','')[:120]}"
            f"{' (edit '+('OK' if s.get('applied') else 'failed')+')' if s.get('action')=='edit' else ''}"
            f" -> rc={s.get('rc')}"
            for s in steps if s.get("action"))
        prob = (problems.get(r["instance_id"], "") or "")[:1500]
        text = (f"BUG (abridged):\n{prob}\n\n"
                f"AGENT ACTION TRAIL ({n} steps):\n{trail[:2500]}\n\n"
                f"FINAL PATCH ({struct['final_patch_lines']} lines):\n{r['patch'][:1500]}")
        rows.append({"iid": r["instance_id"], "seed": r["seed"],
                     "failed": not labels[key], "text": text, "struct": struct})
    return rows


JUDGE_PROMPT = """You are monitoring an autonomous coding agent trying to fix a bug.
Below is the bug, the agent's action trail, and its final patch.

{text}

Question: did the agent FAIL to correctly fix the bug? Consider an empty/trivial
patch, flailing/repeated commands, or a patch that doesn't address the root cause
as failure. Reply with ONLY a probability from 0.00 (definitely fixed it) to 1.00
(definitely failed). Just the number."""

_NUM = re.compile(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])")

def parse_prob(txt):
    if not txt: return None
    m = _NUM.search(txt.strip())
    if not m: return None
    try: return max(0.0, min(1.0, float(m.group(1))))
    except ValueError: return None


def claude_bin():
    return shutil.which("claude") or "/opt/homebrew/bin/claude"

def frontier_call(text, model, timeout=120):
    try:
        r = subprocess.run([claude_bin(), "-p", "--model", model],
                           input=JUDGE_PROMPT.format(text=text),
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return parse_prob(r.stdout)
    except Exception:
        pass
    return None

def open_call(text, model, temp=0.0, timeout=120):
    import requests
    try:
        r = requests.post(OLLAMA, json={
            "model": model, "temperature": temp, "max_tokens": 12,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(text=text)}]},
            timeout=timeout)
        if r.status_code == 200:
            return parse_prob(r.json()["choices"][0]["message"]["content"])
    except Exception:
        pass
    return None


# ---- metrics ----
def auc(scores, labels):
    """ROC-AUC; positive label = failed. Mann-Whitney."""
    pairs = [(s, y) for s, y in zip(scores, labels) if s is not None]
    pos = [s for s, y in pairs if y]; neg = [s for s, y in pairs if not y]
    if not pos or not neg: return None, len(pairs)
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)), len(pairs)

def within_cluster_concordance(rows, score_key):
    """For each instance with >=1 failed and >=1 passed seed, fraction of
    (failed, passed) pairs where score(failed) > score(passed). Clean of the
    difficulty confound. Returns (mean_concordance, n_pairs, n_instances)."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        s = r.get(score_key)
        if s is not None:
            by[r["iid"]].append((s, r["failed"]))
    tot = w = insts = 0
    for iid, vs in by.items():
        f = [s for s, y in vs if y]; p = [s for s, y in vs if not y]
        if not f or not p: continue
        insts += 1
        for a in f:
            for b in p:
                tot += 1; w += (a > b) + 0.5 * (a == b)
    return (w / tot if tot else None), tot, insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-model", default="qwen2.5-coder:14b")
    ap.add_argument("--frontier-model", default="sonnet")
    ap.add_argument("--k-selfconsistency", type=int, default=3)
    ap.add_argument("--structural-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_corpus()
    if args.limit: rows = rows[:args.limit]
    print(f"corpus: {len(rows)} trajectories, "
          f"{sum(r['failed'] for r in rows)} failed / {sum(not r['failed'] for r in rows)} resolved",
          flush=True)

    CACHE = cache_path(args.open_model)
    OUT = out_path(args.open_model)
    cache = {}
    if CACHE.exists():
        for l in open(CACHE):
            if l.strip():
                c = json.loads(l); cache[(c["iid"], c["seed"])] = c
    # reuse already-paid frontier scores from the shared cache
    if FRONTIER_CACHE.exists() and FRONTIER_CACHE != CACHE:
        for l in open(FRONTIER_CACHE):
            if l.strip():
                fc = json.loads(l); k = (fc["iid"], fc["seed"])
                cache.setdefault(k, {}).setdefault("frontier_judge", fc.get("frontier_judge"))
                cache[k]["iid"], cache[k]["seed"] = fc["iid"], fc["seed"]

    if not args.analyze_only and not args.structural_only:
        cf = CACHE.open("a")
        for i, r in enumerate(rows):
            key = (r["iid"], r["seed"]); c = cache.get(key, {})
            # retry on missing OR None (rate-limited/transient) so resumption fills gaps
            need = (c.get("open_judge") is None or c.get("frontier_judge") is None
                    or (args.k_selfconsistency > 0 and c.get("open_sc_std") is None))
            if not need:
                continue
            if c.get("open_judge") is None:
                c["open_judge"] = open_call(r["text"], args.open_model, temp=0.0)
            if c.get("open_sc_std") is None and args.k_selfconsistency > 0:
                samples = [open_call(r["text"], args.open_model, temp=0.8)
                           for _ in range(args.k_selfconsistency)]
                samples = [s for s in samples if s is not None]
                if len(samples) >= 2:
                    mu = sum(samples) / len(samples)
                    c["open_sc_std"] = (sum((x - mu) ** 2 for x in samples) / len(samples)) ** 0.5
                    c["open_sc_mean"] = mu
            if c.get("frontier_judge") is None:
                c["frontier_judge"] = frontier_call(r["text"], args.frontier_model)
            c["iid"], c["seed"] = r["iid"], r["seed"]
            cache[key] = c
            cf.write(json.dumps(c) + "\n"); cf.flush()
            if (i + 1) % 10 == 0:
                print(f"  scored {i+1}/{len(rows)} (open={c.get('open_judge')}, "
                      f"frontier={c.get('frontier_judge')})", flush=True)
        cf.close()

    # attach scores
    for r in rows:
        c = cache.get((r["iid"], r["seed"]), {})
        r["open_judge"] = c.get("open_judge")
        r["frontier_judge"] = c.get("frontier_judge")
        r["open_sc_std"] = c.get("open_sc_std")
        for k, v in r["struct"].items():
            r[f"st_{k}"] = v

    labels = [r["failed"] for r in rows]
    detectors = ["frontier_judge", "open_judge", "open_sc_std",
                 "st_n_steps", "st_stuck", "st_empty_patch", "st_rc_err_frac",
                 "st_max_diff_lines"]
    print("\n=== DETECTION: predicting FAILURE (positive=failed) ===")
    print(f"{'detector':>22}  {'pooled AUC':>10} {'n':>4}   {'within-cluster':>14} {'pairs':>5} {'inst':>4}")
    report = {}
    for d in detectors:
        scores = [r.get(d) for r in rows]
        a, n = auc(scores, labels)
        wc, pairs, insts = within_cluster_concordance(rows, d)
        report[d] = {"pooled_auc": a, "n": n, "within_cluster": wc,
                     "wc_pairs": pairs, "wc_instances": insts}
        print(f"{d:>22}  {a if a is None else round(a,3)!s:>10} {n:>4}   "
              f"{wc if wc is None else round(wc,3)!s:>14} {pairs:>5} {insts:>4}")

    # headline comparison
    fa = report["frontier_judge"]["pooled_auc"]; oa = report["open_judge"]["pooled_auc"]
    print("\n=== HEADLINE: open vs frontier (pooled, same trajectories) ===")
    if fa and oa:
        print(f"  frontier={fa:.3f}  open={oa:.3f}  gap(open-frontier)={oa-fa:+.3f}")
        print(f"  -> {'OPEN MATCHES/BEATS frontier' if oa >= fa - 0.03 else 'open trails frontier'}")
    else:
        print("  (run without --structural-only/--analyze-only to get judge scores)")
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
