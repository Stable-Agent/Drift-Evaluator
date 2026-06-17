#!/usr/bin/env python
"""Recovery pass: re-judge records whose candidate was generated but the judge
call failed (null verdict + non-empty cand). Cheap; no regeneration."""
from __future__ import annotations
import json, pathlib, threading
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import sys
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
from corrector_content import call, JUDGE, parse_verdict, OUT, JUDGE_MODEL

def main():
    last = {}
    for l in open(OUT):
        if l.strip():
            r = json.loads(l); last[(r["iid"], r["cond"])] = r
    todo = [(k, r) for k, r in last.items()
            if not r["verdict"] and (r.get("cand") or "").strip()]
    print(f"re-judging {len(todo)} records (cand present, judge failed)")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    gold = {r["instance_id"]: r["patch"] for r in ds}
    repo = {r["instance_id"]: r["repo"] for r in ds}

    lock = threading.Lock(); fp = open(OUT, "a"); n = {"ok": 0}
    def work(item):
        (iid, cond), r = item
        if iid not in gold: return
        v = parse_verdict(call(JUDGE.format(repo=repo[iid], gold=gold[iid][:3500],
                                            cand=r["cand"][:3500]), JUDGE_MODEL))
        if v:
            with lock:
                rec = dict(r); rec["verdict"] = v
                fp.write(json.dumps(rec) + "\n"); fp.flush(); n["ok"] += 1
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(work, todo))
    fp.close()
    print(f"recovered {n['ok']}/{len(todo)} verdicts")

if __name__ == "__main__":
    main()
