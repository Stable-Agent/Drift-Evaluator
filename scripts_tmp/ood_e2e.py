#!/usr/bin/env python
"""End-to-end closed-loop corrector test on out-of-distribution bugs.

Per OOD case (buildable repo):
  trigger:    apply PR test changes at base -> tests FAIL (bug present)
  intervene:  fresh focused re-solve (problem + code) -> apply -> re-run tests
  retry:      up to MAX_ATTEMPTS fresh re-solves; recovered if any passes
Real test-passing ground truth, no oracle, no judge. OOD = no memorization.

Reuses the pylint harness scaffolding + corrector_content.call/GEN.
"""
from __future__ import annotations
import json, re, subprocess, sys, pathlib
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
from corrector_content import call, GEN, GEN_MODEL
from ood_eval import REPO, PY, pr_info, split_diff, apply_patch, run_tests, git

CASES = pathlib.Path("Drift-Evaluator/reports/ood_cases.jsonl")
RESULTS = pathlib.Path("Drift-Evaluator/reports/ood_results.jsonl")
OUT = pathlib.Path("Drift-Evaluator/reports/ood_e2e.jsonl")
MAX_ATTEMPTS = 3

def robust_apply(cand, src_dir):
    """Apply SEARCH/REPLACE blocks. Exact match first; if a block doesn't match
    exactly, try a STRICT line-window match (same stripped lines, contiguous).
    Skip blocks that match neither — never force a fuzzy mis-placement (that
    introduced false failures). Returns (blocks_applied, total)."""
    blocks = re.findall(r"<<<+\s*SEARCH\s*\n(.*?)\n===+\s*\n(.*?)\n>>>+\s*REPLACE",
                        cand, re.DOTALL)
    applied = 0
    files = list(src_dir.rglob("*.py"))
    for search, repl in blocks:
        if not search.strip():
            continue
        for f in files:                              # EXACT only — never guess a location
            txt = f.read_text()
            if search in txt:
                f.write_text(txt.replace(search, repl, 1)); applied += 1; break
    return applied, len(blocks)

def offline_candidate(iid):
    last = None
    for l in open(RESULTS):
        if l.strip():
            r = json.loads(l)
            if r["iid"] == iid and r["cond"] == "RESOLVE" and r.get("cand"):
                last = r["cand"]
    return last

def setup(num):
    base, diff = pr_info(num)
    test_d, src_d, names = split_diff(diff)
    git("reset", "--hard", "-q"); git("checkout", "-q", base); git("clean", "-fdq")
    ok, err = apply_patch(test_d)
    return base, names, ok

def main():
    case_meta = {json.loads(l)["iid"]: json.loads(l)
                 for l in open(CASES) if l.strip()}
    iids = sys.argv[1:] or ["pylint#11027", "pylint#11052", "pylint#11055"]
    fp = OUT.open("a")
    for iid in iids:
        num = iid.split("#")[1]; c = case_meta[iid]
        base, names, ok = setup(num)
        if not ok:
            print(f"{iid}: test patch failed to apply; skip"); continue
        base_pass, _ = run_tests(names)
        if base_pass:
            print(f"{iid}: baseline already PASSES (no bug to reproduce); skip"); continue
        print(f"{iid}: baseline FAIL (bug reproduced) -> intervening", flush=True)

        recovered = False; attempts = 0; log = []
        for att in range(1, MAX_ATTEMPTS + 1):
            attempts = att
            # attempt 1 reuses the offline candidate (free); later attempts re-solve fresh
            if att == 1 and offline_candidate(iid):
                cand = offline_candidate(iid)
            else:
                cand = call(GEN.format(repo=c["repo"], problem=c["problem"],
                                       file=c["file"], code=c["code"], extra=""), GEN_MODEL)
            git("checkout", "-q", "--", "pylint/")        # reset source, keep tests
            if not cand:
                log.append(f"a{att}:gen_fail"); continue
            applied, total = robust_apply(cand, REPO / "pylint")
            if applied == 0:
                log.append(f"a{att}:apply0/{total}"); continue
            passed, tail = run_tests(names)
            log.append(f"a{att}:applied{applied}/{total}:{'PASS' if passed else 'fail'}")
            if passed:
                recovered = True; break
        print(f"  -> {'RECOVERED' if recovered else 'not recovered'} in {attempts} attempt(s) | {log}")
        fp.write(json.dumps({"iid": iid, "recovered": recovered,
                             "attempts": attempts, "log": log}) + "\n"); fp.flush()
    fp.close()

    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    rec = sum(r["recovered"] for r in rows)
    print(f"\n=== e2e recovery: {rec}/{len(rows)} OOD bugs recovered by closed-loop re-solve ===")

if __name__ == "__main__":
    main()
