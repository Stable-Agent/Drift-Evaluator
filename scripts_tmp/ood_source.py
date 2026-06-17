#!/usr/bin/env python
"""Source out-of-distribution bugs: bug-fix PRs merged AFTER the model's
Jan-2026 training cutoff, so they cannot be memorized.

For each: problem statement (PR title+body), gold patch (source-file changes
only), and original code (reconstructed from the diff). Mirrors the SWE-bench
case schema so the corrector experiment can run unchanged.

Output: reports/ood_cases.jsonl
"""
from __future__ import annotations
import json, re, subprocess, pathlib

OUT = pathlib.Path("Drift-Evaluator/reports/ood_cases.jsonl")
CUTOFF = "2026-02-15"          # margin past the Jan-2026 cutoff
REPOS = ["django/django", "sympy/sympy", "astropy/astropy",
         "scikit-learn/scikit-learn", "matplotlib/matplotlib",
         "pandas-dev/pandas", "sphinx-doc/sphinx", "pytest-dev/pytest",
         "scipy/scipy", "pylint-dev/pylint", "pydata/xarray", "numpy/numpy",
         "sqlalchemy/sqlalchemy", "dask/dask", "scrapy/scrapy"]
PER_REPO = 12

def sh(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    return r.stdout if r.returncode == 0 else ""

def is_test(path): return "test" in path.lower()
def is_src(path):
    return path.endswith(".py") and not is_test(path) and "/doc" not in path.lower()

def parse_diff(diff):
    """-> {file: {'hunks':[lines...], 'is_test':bool}} for .py files."""
    files = {}; cur = None
    for line in diff.splitlines():
        m = re.match(r"^diff --git a/(.+) b/(.+)$", line)
        if m:
            cur = m.group(2); files[cur] = []
            continue
        if cur is not None and (line.startswith((" ", "+", "-")) or line.startswith("@@")):
            files[cur].append(line)
    return files

def gold_and_code(files):
    """gold = unified diff of SOURCE files only; code = reconstructed original."""
    src = [f for f in files if is_src(f)]
    if not src: return None, None, None
    f = src[0]                       # primary source file
    hunks = files[f]
    gold_lines = [f"--- a/{f}", f"+++ b/{f}"] + hunks
    orig = [l[1:] for l in hunks if l and l[0] in (" ", "-") and not l.startswith("@@")]
    return f, "\n".join(gold_lines), "\n".join(orig)

def main():
    seen = set()
    if OUT.exists():
        seen = {json.loads(l)["iid"] for l in open(OUT) if l.strip()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fp = OUT.open("a")
    kept = 0
    for repo in REPOS:
        raw = sh(["gh", "search", "prs", "fix in:title", "--repo", repo,
                  "--merged-at", f">{CUTOFF}", "--limit", "20",
                  "--json", "number,title,url"])
        if not raw:
            print(f"{repo}: no search results"); continue
        prs = json.loads(raw)
        n_repo = 0
        for pr in prs:
            num = pr["number"]; iid = f"{repo.split('/')[-1]}#{num}"
            if iid in seen: continue
            title = pr["title"]
            if re.match(r"(?i)^(backport|merge|\[?backport)", title): continue
            url = pr["url"]
            body = sh(["gh", "pr", "view", url, "--json", "body", "-q", ".body"]).strip()
            diff = sh(["gh", "pr", "diff", url])
            if not diff: continue
            if not re.search(r"(?i)\b(fix|bug|regression|incorrect|error|broken)\b", title):
                continue
            files = parse_diff(diff)
            has_test = any(is_test(f) for f in files)
            if not has_test: continue          # require a test change = real bug fix
            f, gold, code = gold_and_code(files)
            if not f or not gold: continue
            if len(gold) > 6000 or len(code) < 40: continue      # focused fixes only
            problem = (title + "\n\n" + body).strip()
            if len(problem) < 80: continue
            fp.write(json.dumps({
                "iid": iid, "repo": repo, "problem": problem[:1500],
                "gold": gold, "file": f, "code": code[:2500],
                "has_test": has_test, "url": url, "title": title}) + "\n")
            fp.flush(); kept += 1; n_repo += 1
            if n_repo >= 6: break       # cap per repo for diversity
        print(f"{repo}: kept {n_repo}")
    fp.close()
    print(f"\ntotal OOD cases: {kept} new ({len(seen)} pre-existing)")

if __name__ == "__main__":
    main()
