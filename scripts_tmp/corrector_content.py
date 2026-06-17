#!/usr/bin/env python
"""Thread #2 reframed: is the corrector's CONTENT (not location) generatable live?

Population: the failed trajectories where the agent was already in the right
place (file+line hit vs gold patch) but failed = pure content problem.

For each, three candidate fixes are scored against the gold patch via an LLM
equivalence judge:
  1. AGENT     - the agent's own (failed) submission         [baseline]
  2. RESOLVE   - fresh LLM given problem + original code region (no hint)
  3. NEG_EVID  - same + the agent's failed attempt, told it failed

If NEG_EVID doesn't beat AGENT, the corrector's 27% lift is lock-in-break
luck. If it clearly beats AGENT and RESOLVE, that's the live content edge.

Offline proxy: equivalence judging is fuzzy in absolute terms, but the same
judge across conditions makes the RELATIVE comparison informative. The agent
baseline (known-failed) is a built-in judge sanity check (should score low).
"""
from __future__ import annotations
import json, re, pathlib, subprocess, sys, time, threading, argparse, shutil
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
from truthsite_eval import parse_patch, eval_sites

GOLD = pathlib.Path("Drift-Evaluator/gold/coding_v0_opus_labeled.jsonl")
TRAJ = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
OUT = pathlib.Path("Drift-Evaluator/reports/corrector_content.jsonl")
GEN_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

def claude_bin():
    """Resolve dynamically each call: the CLI can auto-update mid-run and
    briefly swap the symlink, so don't cache the path."""
    return shutil.which("claude") or "/opt/homebrew/bin/claude"

def call(prompt, model, retries=1, timeout=260):
    # Pass prompt via STDIN (not argv): large prompts as argv made the CLI
    # return empty under concurrent load. stdin is robust to size.
    # Fast-fail: few in-call retries; the multi-round loop retries failures.
    # timeout: large-file prompts (e.g. sympy FILE arm) generate in ~255s,
    # right at the old hardcoded 260s — callers bump this to avoid the race.
    for a in range(retries + 1):
        try:
            r = subprocess.run([claude_bin(), "-p", "--model", model],
                               input=prompt, capture_output=True, text=True, timeout=timeout)
            out = r.stdout.strip()
            if r.returncode == 0 and out:
                return out
        except Exception:        # timeout, missing binary mid-update, etc. -> retry
            pass
        time.sleep(3 + a * 4)
    return None

def traj_path(src, tid):
    hit = list(TRAJ.glob(f"{src}*/{tid}.json"))
    return hit[0] if hit else None

def agent_submission(p):
    obj = json.load(open(p))
    return (obj.get("info", {}) or {}).get("submission", "") if isinstance(obj, dict) else ""

def before_code(agent_diff, gold_files):
    """Reconstruct original (pre-edit) code at the agent's hunks for gold files.
    original line = context(' ') or removed('-'); drop added('+'). No gold leak."""
    out = {}
    cur = None
    for line in agent_diff.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m:
            cur = m.group(1)
            continue
        if cur is None:
            continue
        gf = {g.split("/", 1)[-1] if g.startswith(("a/", "b/")) else g for g in gold_files}
        base = cur[2:] if cur.startswith(("a/", "b/")) else cur
        if not (base in gold_files or base in gf
                or base.rsplit("/", 1)[-1] in {g.rsplit("/", 1)[-1] for g in gold_files}):
            continue
        if line and line[0] in (" ", "-") and not line.startswith("---"):
            out.setdefault(base, []).append(line[1:])
    return {f: "\n".join(ls) for f, ls in out.items() if ls}

GEN = """Fix a bug in {repo}.

ISSUE (abridged):
{problem}

Fix location: {file}
Original code:
```python
{code}
```
{extra}
Output ONLY the minimal edit, nothing else (no prose, no explanation), as one or more blocks:
<<<SEARCH
<exact original lines to replace>
===
<replacement lines>
>>>REPLACE
Keep each block to only the lines that actually change."""

NEG_EXTRA = """
A prior agent tried this change here and it FAILED the tests (right place, wrong change):
```diff
{agent_diff}
```
Produce the CORRECT edit instead.
"""

# FULL = confound control: no file, no code handed over (fresh full-task proxy).
GEN_FULL = """Fix a bug in {repo}.

ISSUE:
{problem}

Identify the right file and lines yourself from your knowledge of {repo}.
Output ONLY the minimal edit, nothing else (no prose), as one or more blocks:
<<<SEARCH
<exact original lines from the repo>
===
<replacement lines>
>>>REPLACE"""

JUDGE = """A bug in {repo} has a known-correct REFERENCE fix:
```diff
{gold}
```
Here is a CANDIDATE fix (search/replace edit or diff):
```
{cand}
```
Does the candidate address the same root cause and produce behavior equivalent to the reference (ignoring cosmetic/style differences)? Reply with exactly one word: YES, NO, or PARTIAL."""

def parse_verdict(raw):
    if not raw: return None
    u = raw.strip().upper()
    toks = u.split()
    if toks:
        first = toks[0].strip(".,:;*")     # judge usually leads with the verdict
        if first in ("YES", "NO", "PARTIAL"):
            return first
    m = re.search(r"\b(YES|NO|PARTIAL)\b", u)   # \b avoids NOT/CANNOT -> NO
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="cap number of cases (0=all)")
    ap.add_argument("--rounds", type=int, default=6, help="resumable retry rounds")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(GOLD) if l.strip()]
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    meta = {r["instance_id"]: r for r in ds}

    cases = []
    for r in recs:
        iid = r["trajectory_id"]
        if iid not in meta: continue
        gold = parse_patch(meta[iid]["patch"])
        if not gold: continue
        afh, alh = eval_sites(r.get("scope", {}).get("agent_sites", []), gold)
        if not (afh and alh):  # require agent was in the right place
            continue
        p = traj_path(r["trajectory_source"], iid)
        if not p: continue
        sub = agent_submission(p)
        if not sub or "diff --git" not in sub: continue
        bc = before_code(sub, list(gold.keys()))
        if not bc: continue
        cases.append({"iid": iid, "repo": meta[iid]["repo"],
                      "problem": meta[iid]["problem_statement"][:1500],
                      "gold": meta[iid]["patch"], "agent_diff": sub,
                      "file": list(bc.keys())[0], "code": list(bc.values())[0][:2500]})
    cases.sort(key=lambda c: c["iid"])           # stable ordering for --limit/resume
    if args.limit:
        cases = cases[:args.limit]
    print(f"{len(cases)} right-place-wrong-change cases with reconstructable code", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    def load_done():
        # only SUCCESSFUL (verdict-present) tasks count as done -> failures retry
        d = {}
        if OUT.exists():
            for l in open(OUT):
                if l.strip():
                    r = json.loads(l)
                    if r.get("verdict"): d[(r["iid"], r["cond"])] = r
        return d
    done = load_done()
    lock = threading.Lock(); fp = OUT.open("a"); cnt = {"n": 0}

    def gen_and_judge(c, cond):
      try:
        if (c["iid"], cond) in done: return
        if cond == "AGENT":
            cand = c["agent_diff"]
        elif cond == "FULL":
            cand = call(GEN_FULL.format(repo=c["repo"], problem=c["problem"]), GEN_MODEL)
        else:
            extra = NEG_EXTRA.format(agent_diff=c["agent_diff"][:1500]) if cond == "NEG_EVID" else ""
            cand = call(GEN.format(repo=c["repo"], problem=c["problem"], file=c["file"],
                                   code=c["code"], extra=extra), GEN_MODEL)
        verdict = None
        if cand:
            verdict = parse_verdict(call(JUDGE.format(repo=c["repo"], gold=c["gold"][:2500],
                                                      cand=cand[:2500]), JUDGE_MODEL))
        with lock:
            fp.write(json.dumps({"iid": c["iid"], "cond": cond,
                                 "verdict": verdict, "gen_ok": cand is not None,
                                 "cand": (cand or "")[:3000]}) + "\n")
            fp.flush(); cnt["n"] += 1
            if cnt["n"] % 15 == 0:
                print(f"  [{cnt['n']}] {c['iid']} {cond} -> {verdict}", flush=True)
      except Exception as e:
        sys.stderr.write(f"  worker error {c['iid']} {cond}: {e}\n")

    if not args.analyze_only:
        for rnd in range(args.rounds):
            done.clear(); done.update(load_done())   # refresh successes each round
            incomplete = [(c, cond) for c in cases
                          for cond in ("AGENT", "RESOLVE", "FULL")  # NEG_EVID settled
                          if (c["iid"], cond) not in done]
            if not incomplete:
                print(f"all complete after round {rnd}", flush=True); break
            print(f"round {rnd+1}/{args.rounds}: {len(incomplete)} incomplete "
                  f"({len(done)} done)", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(lambda t: gen_and_judge(*t), incomplete))
        fp.close()

    # analyze
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    from collections import defaultdict
    agg = defaultdict(lambda: {"YES": 0, "PARTIAL": 0, "NO": 0, "none": 0, "n": 0})
    for r in rows:
        a = agg[r["cond"]]; a["n"] += 1
        a[r["verdict"] or "none"] += 1
    print(f"\n=== equivalence vs gold patch ({len(cases)} cases) ===")
    print(f"{'condition':>10}  {'n':>3}  {'YES':>5}  {'PARTIAL':>7}  {'NO':>4}  {'YES%':>6}  {'YES+½P%':>7}")
    for cond in ("AGENT", "RESOLVE", "NEG_EVID"):
        a = agg[cond]; n = a["n"] or 1
        yes_p = a["YES"] / n; soft = (a["YES"] + 0.5 * a["PARTIAL"]) / n
        print(f"{cond:>10}  {a['n']:>3}  {a['YES']:>5}  {a['PARTIAL']:>7}  {a['NO']:>4}  "
              f"{yes_p:>5.1%}  {soft:>6.1%}")
    print("\nAGENT row is the judge sanity check (known-failed; should be low).")
    print("the test: does NEG_EVID clearly exceed AGENT and RESOLVE?")

if __name__ == "__main__":
    main()
