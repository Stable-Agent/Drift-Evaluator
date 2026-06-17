#!/usr/bin/env python
"""Run FULL vs RESOLVE on out-of-distribution (post-cutoff) bugs and judge
against the PR's real source fix. Memorization disambiguator:
  - RESOLVE holds, FULL collapses (vs SWE-bench) -> real code-based fixing
  - both collapse -> the SWE-bench corrector signal was contamination
Resumable multi-round (success-only done), reuses corrector_content primitives.
"""
from __future__ import annotations
import json, pathlib, threading, argparse
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
from corrector_content import call, GEN, GEN_FULL, JUDGE, parse_verdict, GEN_MODEL, JUDGE_MODEL

CASES = pathlib.Path("Drift-Evaluator/reports/ood_cases.jsonl")
OUT = pathlib.Path("Drift-Evaluator/reports/ood_results.jsonl")
CONDS = ("FULL", "RESOLVE")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--round-sleep", type=int, default=0, help="seconds between rounds (ride out the usage cap)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repos", default="", help="comma list of repo substrings to keep")
    ap.add_argument("--resolve-only", action="store_true", help="skip FULL; only RESOLVE")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    global CONDS
    if args.resolve_only:
        CONDS = ("RESOLVE",)

    cases = [json.loads(l) for l in open(CASES) if l.strip()]
    cases = [c for c in cases if c.get("has_test")]      # real tested bug fixes
    if args.repos:
        keep = tuple(s.strip() for s in args.repos.split(","))
        cases = [c for c in cases if any(k in c["iid"] for k in keep)]
    cases.sort(key=lambda c: c["iid"])
    if args.limit: cases = cases[:args.limit]
    print(f"{len(cases)} OOD cases (conds={CONDS})", flush=True)

    def load_done():
        d = {}
        if OUT.exists():
            for l in open(OUT):
                if l.strip():
                    r = json.loads(l)
                    if r.get("verdict"): d[(r["iid"], r["cond"])] = r
        return d
    done = load_done()
    lock = threading.Lock(); fp = OUT.open("a"); cnt = {"n": 0}

    def work(c, cond):
      try:
        if (c["iid"], cond) in done: return
        if cond == "FULL":
            cand = call(GEN_FULL.format(repo=c["repo"], problem=c["problem"]), GEN_MODEL)
        else:
            cand = call(GEN.format(repo=c["repo"], problem=c["problem"], file=c["file"],
                                   code=c["code"], extra=""), GEN_MODEL)
        verdict = None
        if cand:
            verdict = parse_verdict(call(JUDGE.format(repo=c["repo"], gold=c["gold"][:2500],
                                                      cand=cand[:2500]), JUDGE_MODEL))
        with lock:
            fp.write(json.dumps({"iid": c["iid"], "cond": cond, "verdict": verdict,
                                 "gen_ok": cand is not None, "cand": (cand or "")[:3000]}) + "\n")
            fp.flush(); cnt["n"] += 1
            if cnt["n"] % 10 == 0: print(f"  [{cnt['n']}] {c['iid']} {cond} -> {verdict}", flush=True)
      except Exception as e:
        sys.stderr.write(f"  err {c['iid']} {cond}: {e}\n")

    if not args.analyze_only:
        for rnd in range(args.rounds):
            done.clear(); done.update(load_done())
            inc = [(c, k) for c in cases for k in CONDS if (c["iid"], k) not in done]
            if not inc:
                print(f"all complete after round {rnd}", flush=True); break
            print(f"round {rnd+1}/{args.rounds}: {len(inc)} incomplete ({len(done)} done)", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(lambda t: work(*t), inc))
            if args.round_sleep and rnd < args.rounds - 1:
                import time as _t; _t.sleep(args.round_sleep)
        fp.close()

    # analyze
    from collections import defaultdict
    last = {}
    for l in open(OUT):
        if l.strip(): r = json.loads(l); last[(r["iid"], r["cond"])] = r
    SCORE = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
    cnt2 = defaultdict(lambda: defaultdict(int))
    for (i, k), r in last.items():
        if r["verdict"]: cnt2[k][r["verdict"]] += 1
    print("\n=== OOD results vs SWE-bench ===")
    print(f"{'cond':>9}  {'n':>3}  {'YES':>3} {'PART':>4} {'NO':>3}  {'soft':>5}  {'(SWE-bench)':>12}")
    swe = {"FULL": 0.46, "RESOLVE": 0.36}
    for k in CONDS:
        d = cnt2[k]; n = sum(d.values()); soft = (d.get("YES",0)+0.5*d.get("PARTIAL",0))/n if n else 0
        print(f"{k:>9}  {n:>3}  {d.get('YES',0):>3} {d.get('PARTIAL',0):>4} {d.get('NO',0):>3}  {soft:>5.3f}  {swe[k]:>12.2f}")
    # paired
    by = defaultdict(dict)
    for (i, k), r in last.items():
        if r["verdict"]: by[i][k] = r["verdict"]
    pair = [i for i, cv in by.items() if "FULL" in cv and "RESOLVE" in cv]
    if pair:
        sf = sum(SCORE[by[i]["FULL"]] for i in pair)/len(pair)
        sr = sum(SCORE[by[i]["RESOLVE"]] for i in pair)/len(pair)
        rb = sum(1 for i in pair if SCORE[by[i]["RESOLVE"]] > SCORE[by[i]["FULL"]])
        print(f"\npaired n={len(pair)}: FULL={sf:.3f} RESOLVE={sr:.3f}  RESOLVE>FULL in {rb}")

if __name__ == "__main__":
    main()
