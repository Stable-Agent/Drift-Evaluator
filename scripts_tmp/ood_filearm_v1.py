#!/usr/bin/env python
"""Three-arm corrector experiment per docs/prereg_corrector_file_loc_v1.md.

Per evaluable case (from ood_cases_v2.jsonl, sourced by ood_source_v2.py):
  gate:  checkout base_sha, apply PR TEST diff only
         -> baseline must FAIL (bug reproduced)
         -> + gold src diff must PASS (harness sanity)   [else attrition]
  arms:  FULL (issue only) / FILE (full pre-fix file) / WINDOW (gold-hunk
         window) -> generate SEARCH/REPLACE -> apply (one regen retry iff
         apply fails, identical rule for all arms) -> run the PR's tests.
  judge: Haiku equivalence verdict recorded for every candidate (secondary;
         never decides the primary metric).

Primary metric: execution pass. Resumable: done = (iid, arm) with a recorded
exec result. Gate results cached per iid in the gates file.

Usage:
  python ood_filearm_v1.py --setup            # clone + venv for all repos
  python ood_filearm_v1.py --gate-only        # run evaluability gates only
  python ood_filearm_v1.py                    # gates + arms (resumable)
  python ood_filearm_v1.py --analyze-only
"""
from __future__ import annotations
import argparse, json, math, pathlib, re, subprocess, sys
from collections import defaultdict

sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
from corrector_content import call, GEN, GEN_FULL, JUDGE, parse_verdict, GEN_MODEL, JUDGE_MODEL

CASES = pathlib.Path("Drift-Evaluator/reports/ood_cases_v2.jsonl")
OUT = pathlib.Path("Drift-Evaluator/reports/ood_filearm_v1.jsonl")
GATES = pathlib.Path("Drift-Evaluator/reports/ood_filearm_v1_gates.jsonl")
WORK = pathlib.Path("/tmp/ood_eval_v2")
ARMS = ("FULL", "FILE", "WINDOW")
CLONE_URL = {r.split("/")[-1]: f"https://github.com/{r}.git" for r in
             ["sympy/sympy", "django/django", "pydata/xarray",
              "pylint-dev/pylint", "pytest-dev/pytest", "scrapy/scrapy"]}
EXTRA_TEST_DEPS = {                      # discovered at gate time
    "sympy": ["hypothesis"],
    "xarray": ["pytest-mypy-plugins", "pytest-xdist", "hypothesis"],
}


def sh(args, cwd=None, timeout=600, inp=None):
    r = subprocess.run(args, cwd=cwd, input=inp, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode == 0, (r.stdout + r.stderr)


# ---------- per-repo workspace ----------

def repo_dir(name): return WORK / name
def venv_py(name): return WORK / f"venv_{name}" / "bin" / "python"


def ensure_env(name: str) -> bool:
    """Clone + venv + editable install (best effort), once per repo."""
    d = repo_dir(name)
    if not d.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        ok, out = sh(["git", "clone", "--quiet", CLONE_URL[name], str(d)])
        if not ok:
            print(f"  [{name}] clone failed: {out[-200:]}"); return False
    py = venv_py(name)
    if py.exists():
        return True
    ok, out = sh([sys.executable, "-m", "venv", str(WORK / f"venv_{name}")])
    if not ok:
        print(f"  [{name}] venv failed: {out[-200:]}"); return False
    pip = [str(py), "-m", "pip", "install", "-q"]
    sh(pip + ["--upgrade", "pip"])
    ok, out = sh(pip + ["-e", str(d)], timeout=900)
    if not ok:
        print(f"  [{name}] install -e . failed: {out[-300:]}"); return False
    sh(pip + ["pytest"] + EXTRA_TEST_DEPS.get(name, []))
    for req in d.glob("requirements*test*.txt"):       # pylint-style test deps
        sh(pip + ["-r", str(req)], timeout=900)
    return True


def git(name, *a, inp=None):
    return sh(["git", "-C", str(repo_dir(name)), *a], inp=inp)


def checkout_base(name, sha) -> bool:
    git(name, "reset", "--hard", "-q")
    git(name, "clean", "-fdq")
    ok, _ = git(name, "checkout", "-q", sha)
    if not ok:                                          # sha not in clone yet
        git(name, "fetch", "-q", "origin", sha)
        ok, _ = git(name, "checkout", "-q", sha)
    return ok


# ---------- diffs and tests ----------

def pr_diff(case) -> str:
    cache = WORK / "diffs" / (case["iid"].replace("#", "_") + ".diff")
    if cache.exists():
        return cache.read_text()
    ok, out = sh(["gh", "pr", "diff", case["url"]])
    if ok and out.strip():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out)
        return out
    return ""


def split_diff(diff, test_files):
    """-> (test_diff, src_diff) using the case's recorded test file list."""
    blocks, cur, head = [], [], None
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if cur: blocks.append((head, "".join(cur)))
            cur, head = [line], line
        else:
            cur.append(line)
    if cur: blocks.append((head, "".join(cur)))
    tset = set(test_files)
    test_d, src_d = [], []
    for head, b in blocks:
        m = re.search(r" b/(\S+)", head)
        path = m.group(1) if m else ""
        (test_d if path in tset else src_d).append(b)
    return "".join(test_d), "".join(src_d)


def apply_diff(name, patch) -> bool:
    if not patch.strip():
        return False
    ok, _ = git(name, "apply", "--3way", inp=patch)
    return ok


def run_tests(case) -> tuple[bool | None, str]:
    """Run the PR's changed tests. Verdict is by exit code (string-matching
    pytest summaries miscounts 'xfailed'): 0 = pass, 1 = test failures,
    anything else = harness problem (collection error, no tests) -> None.
    django via runtests.py; pylint functional fixtures via test_functional.py;
    rest via pytest on the changed files."""
    name = case["repo"].split("/")[-1]
    py, d = str(venv_py(name)), repo_dir(name)
    if name == "django":
        labels = [re.sub(r"^tests/(.+)\.py$", r"\1", f).replace("/", ".")
                  for f in case["test_files"] if f.startswith("tests/")]
        if not labels:
            return None, "no django test labels"
        cmd = [py, "tests/runtests.py", "--parallel", "1", *labels]
    else:
        sel = []
        if name == "pylint":
            func = sorted({m for f in case["test_files"]
                           for m in re.findall(r"functional/.*/([^/]+)\.(?:py|txt)", f)})
            if func:
                sel = ["tests/test_functional.py", "-k", " or ".join(func)]
            paths = [f for f in case["test_files"]
                     if "functional/" not in f and f.endswith(".py") and (d / f).exists()]
        else:
            paths = [f for f in case["test_files"] if (d / f).exists()]
        if not sel and not paths:
            return None, "no runnable test files"
        cmd = [py, "-m", "pytest", "-q", "--no-header",
               "-p", "no:cacheprovider", *sel, *paths]
    try:
        r = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "test timeout"
    out = r.stdout + r.stderr
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    if r.returncode == 0:
        return True, tail[:200]
    if r.returncode == 1:
        return False, tail[:200]
    return None, f"rc={r.returncode}: {tail[:180]}"


def apply_candidate(case, cand) -> tuple[int, int]:
    """Exact-match SEARCH/REPLACE. Primary file first, then repo-wide exact.
    Never fuzzy (robust_apply lesson from ood_e2e)."""
    blocks = re.findall(r"<<<+\s*SEARCH\s*\n(.*?)\n===+\s*\n(.*?)\n>>>+\s*REPLACE",
                        cand, re.DOTALL)
    name = case["repo"].split("/")[-1]
    d = repo_dir(name)
    applied = 0
    primary = d / case["file"]
    for search, repl in blocks:
        if not search.strip():
            continue
        if primary.exists() and search in primary.read_text():
            primary.write_text(primary.read_text().replace(search, repl, 1))
            applied += 1
            continue
        for f in d.rglob("*.py"):                       # FULL arm may pick another file
            if ".git" in f.parts or "test" in str(f).lower():
                continue
            txt = f.read_text(errors="replace")
            if search in txt:
                f.write_text(txt.replace(search, repl, 1)); applied += 1; break
    return applied, len(blocks)


def reset_src(name, test_diff):
    """Back to base + test changes only."""
    git(name, "checkout", "-q", "--", ".")
    git(name, "clean", "-fdq")
    apply_diff(name, test_diff)


# ---------- gate + arms ----------

def gate_case(case) -> dict:
    """Evaluability per prereg §5: bug reproduces AND gold passes."""
    name = case["repo"].split("/")[-1]
    if not ensure_env(name):
        return {"iid": case["iid"], "evaluable": False, "reason": "env setup failed"}
    diff = pr_diff(case)
    if not diff:
        return {"iid": case["iid"], "evaluable": False, "reason": "pr diff fetch failed"}
    test_d, src_d = split_diff(diff, case["test_files"])
    if not checkout_base(name, case["base_sha"]):
        return {"iid": case["iid"], "evaluable": False, "reason": "base checkout failed"}
    if not apply_diff(name, test_d):
        return {"iid": case["iid"], "evaluable": False, "reason": "test diff apply failed"}
    base_pass, base_tail = run_tests(case)
    if base_pass is None:
        return {"iid": case["iid"], "evaluable": False, "reason": f"baseline: {base_tail}"}
    if base_pass:
        return {"iid": case["iid"], "evaluable": False, "reason": "bug not reproduced (baseline passes)"}
    if not apply_diff(name, src_d):
        return {"iid": case["iid"], "evaluable": False, "reason": "gold diff apply failed"}
    gold_pass, gold_tail = run_tests(case)
    if not gold_pass:
        return {"iid": case["iid"], "evaluable": False, "reason": f"gold fails in harness: {gold_tail}"}
    return {"iid": case["iid"], "evaluable": True, "reason": ""}


def gen_candidate(case, arm) -> str | None:
    if arm == "FULL":
        return call(GEN_FULL.format(repo=case["repo"], problem=case["problem"]),
                    GEN_MODEL, timeout=540)
    code = case["file_content"] if arm == "FILE" else case["window_code"]
    return call(GEN.format(repo=case["repo"], problem=case["problem"],
                           file=case["file"], code=code, extra=""),
                GEN_MODEL, timeout=540)


def run_arm(case, arm, test_diff) -> dict:
    name = case["repo"].split("/")[-1]
    row = {"iid": case["iid"], "arm": arm, "attempts": 0, "gen_ok": False,
           "applied": 0, "blocks": 0, "exec_pass": False, "judge": None, "tail": ""}
    for att in (1, 2):                                  # one regen retry iff apply fails
        row["attempts"] = att
        cand = gen_candidate(case, arm)
        if cand and "session limit" in cand.lower():    # capped CLI can return the
            cand = None                                 # limit banner as stdout
        if not cand:
            row["tail"] = "gen failed"; continue
        row["gen_ok"] = True
        reset_src(name, test_diff)
        applied, total = apply_candidate(case, cand)
        row["applied"], row["blocks"] = applied, total
        if applied == 0:
            row["tail"] = "apply failed"
            continue                                    # -> retry once
        passed, tail = run_tests(case)
        row["exec_pass"] = bool(passed)
        row["tail"] = tail
        row["judge"] = parse_verdict(call(JUDGE.format(
            repo=case["repo"], gold=case["gold"][:2500], cand=cand[:2500]),
            JUDGE_MODEL))
        break
    return row


# ---------- analysis ----------

def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact sign test."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


# Harness errors = the TEST RUNNER itself couldn't execute, for reasons
# unrelated to the candidate's code. These are attrition (§5), not fails.
# CRITICAL: a candidate that breaks imports ("No module named
# 'django.utils.foo'") or syntax ("IndentationError") is a REAL fail, not a
# harness error — do NOT match those, or you silently drop bad-candidate
# failures and bias the comparison.
_HARNESS_ERR = (
    "no module named pytest", "no module named 'pytest",      # broken venv
    "no module named 'xdist'", "no module named 'execnet'",   # missing plugin
    "no module named hypothesis", "no module named 'hypothesis'",
    "no runnable test", "no django test", "no test names",
    "no test files", "test timeout",
)

def _is_real(r) -> bool:
    """A real verdict ran the tests to a genuine pass/fail. Excludes only:
    cap victims ('gen failed'), unparseable candidates (blocks==0), and true
    harness errors (test runner missing/uninvocable). A passing run, an honest
    test failure ('N failed, M passed'), and a candidate that breaks the build
    are all REAL."""
    if not (r.get("gen_ok") and r.get("blocks")):
        return False
    tail = (r.get("tail") or "").lower()
    if tail == "gen failed":
        return False
    if r.get("exec_pass"):
        return True
    return not any(e in tail for e in _HARNESS_ERR)


def analyze():
    gates = {r["iid"]: r for r in map(json.loads, open(GATES))} if GATES.exists() else {}
    rows = {}
    if OUT.exists():
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); rows[(r["iid"], r["arm"])] = r
    rows = {k: r for k, r in rows.items() if _is_real(r)}      # drop non-verdicts
    ev = sorted(i for i, g in gates.items() if g["evaluable"])
    print(f"\n=== prereg_corrector_file_loc_v1 ===")
    print(f"gated: {len(gates)}  evaluable: {len(ev)}")
    attr = defaultdict(int)
    for g in gates.values():
        if not g["evaluable"]: attr[g["reason"].split(":")[0]] += 1
    for k, v in sorted(attr.items(), key=lambda kv: -kv[1]):
        print(f"  attrition - {k}: {v}")
    incomplete = [i for i in ev if not all((i, a) in rows for a in ARMS)]
    if incomplete:
        print(f"  INCOMPLETE (evaluable, missing a real verdict): {incomplete}")
    print(f"\n{'arm':>7}  {'n':>3}  {'pass':>4}  rate")
    for arm in ARMS:
        got = [rows[(i, arm)] for i in ev if (i, arm) in rows]
        p = sum(r["exec_pass"] for r in got)
        print(f"{arm:>7}  {len(got):>3}  {p:>4}  {p/len(got):.3f}" if got else f"{arm:>7}    0")
    # Pairwise-complete: each comparison uses every case where BOTH its arms
    # have real verdicts — NOT only cases where all three ran. Requiring the
    # third (irrelevant) arm would arbitrarily drop valid two-arm pairs (e.g.
    # an anti-FILE case whose WINDOW arm hit a harness error), biasing the test.
    def pr(a, b):
        both = [i for i in ev if (i, a) in rows and (i, b) in rows]
        w = sum(rows[(i, a)]["exec_pass"] > rows[(i, b)]["exec_pass"] for i in both)
        l = sum(rows[(i, a)]["exec_pass"] < rows[(i, b)]["exec_pass"] for i in both)
        t = len(both) - w - l
        print(f"  {a} vs {b}: +{w}/-{l} (={t} ties, n={len(both)}), "
              f"one-sided sign p={sign_test_p(w, l):.4f}")
    print(f"\npairwise-complete comparisons:"); pr("FILE", "FULL"); pr("WINDOW", "FILE")
    conf = defaultdict(int)
    for r in rows.values():
        if r["judge"]:
            conf[(r["judge"], r["exec_pass"])] += 1
    if conf:
        print("\njudge x execution:")
        for (j, e), n in sorted(conf.items()):
            print(f"  judge={j:<8} exec_pass={e}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="clone+venv all repos, then exit")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repos", default="", help="comma list of repo-name substrings")
    args = ap.parse_args()

    if args.analyze_only:
        analyze(); return
    if args.setup:
        for name in CLONE_URL:
            print(f"setup {name}: {'ok' if ensure_env(name) else 'FAILED'}", flush=True)
        return

    cases = [json.loads(l) for l in open(CASES) if l.strip()]
    if args.repos:
        keep = tuple(s.strip() for s in args.repos.split(","))
        cases = [c for c in cases if any(k in c["iid"] for k in keep)]
    cases.sort(key=lambda c: c["iid"])
    if args.limit:
        cases = cases[:args.limit]

    gates = {r["iid"]: r for r in map(json.loads, open(GATES))} if GATES.exists() else {}
    # gen-failure rows (rate limits etc.) are transient -> retried on rerun;
    # apply-failure rows with their retry consumed are real verdicts per
    # prereg §5. A row whose tail is "gen failed" lost its retry to the cap,
    # so it is NOT done; same for blocks==0 (no parseable candidate).
    done = set()
    if OUT.exists():
        done = {(r["iid"], r["arm"]) for r in map(json.loads, open(OUT))
                if r.get("gen_ok") and r.get("blocks") and r.get("tail") != "gen failed"}

    gf, of = GATES.open("a"), OUT.open("a")
    for c in cases:
        if c["iid"] not in gates:
            g = gate_case(c)
            gates[c["iid"]] = g
            gf.write(json.dumps(g) + "\n"); gf.flush()
            print(f"{c['iid']}: {'EVALUABLE' if g['evaluable'] else 'attrition: ' + g['reason']}",
                  flush=True)
        if args.gate_only or not gates[c["iid"]]["evaluable"]:
            continue
        name = c["repo"].split("/")[-1]
        test_d, _ = split_diff(pr_diff(c), c["test_files"])
        checkout_base(name, c["base_sha"])
        apply_diff(name, test_d)
        for arm in ARMS:
            if (c["iid"], arm) in done:
                continue
            row = run_arm(c, arm, test_d)
            of.write(json.dumps(row) + "\n"); of.flush()
            print(f"  {c['iid']} {arm}: {'PASS' if row['exec_pass'] else 'fail'} "
                  f"(applied {row['applied']}/{row['blocks']}, judge={row['judge']})", flush=True)
    gf.close(); of.close()
    analyze()


if __name__ == "__main__":
    main()
