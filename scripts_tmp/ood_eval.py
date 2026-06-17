#!/usr/bin/env python
"""Test-passing eval harness (proof-of-concept on pylint functional tests).

For one OOD pylint case:
  1. checkout base commit, apply the PR's TEST changes only
  2. run the affected functional tests -> BASELINE (expect FAIL = bug present)
  3. apply the PR's GOLD source fix -> run (expect PASS = harness sanity)
  4. reset source, apply our RESOLVE CANDIDATE (search/replace) -> run (result)

Turns the fuzzy LLM-judge verdict into real pass/fail.
"""
from __future__ import annotations
import json, re, subprocess, sys, pathlib

REPO = pathlib.Path("/tmp/ood_eval/pylint")
PY = "/tmp/ood_eval/venv/bin/python"
RESULTS = pathlib.Path("Drift-Evaluator/reports/ood_results.jsonl")

def git(*a): return subprocess.run(["git", "-C", str(REPO), *a],
                                   capture_output=True, text=True)

def pr_info(num):
    base = subprocess.run(["gh", "api", f"repos/pylint-dev/pylint/pulls/{num}",
                           "-q", ".base.sha"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["gh", "pr", "diff", f"https://github.com/pylint-dev/pylint/pull/{num}"],
                          capture_output=True, text=True).stdout
    return base, diff

def split_diff(diff):
    """-> (test_diff, src_diff, test_names) by file path."""
    blocks, cur, head = [], [], None
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if cur: blocks.append((head, "".join(cur)))
            cur, head = [line], line
        else:
            cur.append(line)
    if cur: blocks.append((head, "".join(cur)))
    test_d, src_d, names = [], [], set()
    for head, b in blocks:
        m = re.search(r"b/(\S+)", head); path = m.group(1) if m else ""
        if "/doc/" in head or path.startswith("doc/"): continue
        if "tests/" in path:
            test_d.append(b)
            mm = re.search(r"functional/.*/([^/]+)\.(py|txt)", path)
            if mm: names.add(mm.group(1))
        elif path.endswith(".py"):
            src_d.append(b)
    return "".join(test_d), "".join(src_d), sorted(names)

def apply_patch(patch):
    p = subprocess.run(["git", "-C", str(REPO), "apply", "--3way"],
                       input=patch, capture_output=True, text=True)
    return p.returncode == 0, p.stderr

def run_tests(names):
    if not names: return None, "no test names"
    sel = " or ".join(names)
    r = subprocess.run([PY, "-m", "pytest", "tests/test_functional.py", "-k", sel,
                        "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    passed = ("failed" not in out.lower() and "error" not in out.lower()
              and ("passed" in out.lower() or r.returncode == 0))
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return passed, tail

def apply_candidate(cand):
    """Parse <<<SEARCH/===/>>>REPLACE blocks and edit files under pylint/."""
    blocks = re.findall(r"<<<+\s*SEARCH\s*\n(.*?)\n===+\s*\n(.*?)\n>>>+\s*REPLACE",
                        cand, re.DOTALL)
    if not blocks: return False, "no SEARCH/REPLACE blocks parsed"
    applied = 0
    for src_dir in (REPO / "pylint",):
        for f in src_dir.rglob("*.py"):
            txt = f.read_text()
            for search, repl in blocks:
                if search.strip() and search in txt:
                    txt = txt.replace(search, repl, 1); applied += 1
            f.write_text(txt)
    return applied > 0, f"applied {applied}/{len(blocks)} blocks"

def candidate_for(iid):
    last = None
    for l in open(RESULTS):
        if l.strip():
            r = json.loads(l)
            if r["iid"] == iid and r["cond"] == "RESOLVE" and r.get("cand"):
                last = r
    return last

def main():
    iid = sys.argv[1] if len(sys.argv) > 1 else "pylint#11055"
    num = iid.split("#")[1]
    base, diff = pr_info(num)
    test_d, src_d, names = split_diff(diff)
    print(f"{iid}  base={base[:10]}  tests={names}")

    git("reset", "--hard", "-q"); git("checkout", "-q", base); git("clean", "-fdq")
    ok, err = apply_patch(test_d)
    if not ok: print("  test patch failed:", err[:200]); return
    base_pass, base_tail = run_tests(names)
    print(f"  [1] baseline (bug present): {'PASS' if base_pass else 'FAIL'}  | {base_tail}")

    ok, err = apply_patch(src_d)
    gold_pass, gold_tail = run_tests(names) if ok else (None, err[:120])
    print(f"  [2] + gold fix:             {'PASS' if gold_pass else 'FAIL'}  | {gold_tail}")

    # reset source to base+test, apply candidate
    git("checkout", "-q", "--", "pylint/")
    rec = candidate_for(iid)
    if rec:
        cok, cmsg = apply_candidate(rec["cand"])
        cand_pass, cand_tail = run_tests(names) if cok else (None, cmsg)
        print(f"  [3] + RESOLVE candidate:    {'PASS' if cand_pass else 'FAIL'}  | "
              f"judge={rec['verdict']} | {cmsg} | {cand_tail}")
    else:
        print("  [3] no candidate found")

if __name__ == "__main__":
    main()
