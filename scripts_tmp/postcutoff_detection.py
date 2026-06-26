#!/usr/bin/env python
"""POST-CUTOFF rigor check for the detection finding: do the frontier judge
(blind P(fail)) and the open white-box feature (answer_entropy) hold on bugs
sonnet COULD NOT have memorized (post-cutoff PRs, ood_cases_v2)?

Removes the public-SWE-bench memorization confound behind frontier's 0.956.

Pipeline (resumable, buildable pip repos):
  A. ensure venvs (reuses ood_filearm_v1.ensure_env)
  B. gate: bug reproduces + GOLD passes (ood_filearm_v1.gate_case)
  C. build a detection set per bug: GOLD patch (label=pass) + qwen-14B candidates
     (FILE arm, TEXT SAVED) execution-labeled pass/fail
  D. score every (bug, patch) with sonnet (blind P(fail)) AND open answer_entropy
     (verdict-token logprobs); AUC vs execution.

Compare to pre-cutoff: frontier 0.956 / open answer_entropy 0.814 (non-empty).
"""
from __future__ import annotations
import argparse, json, math, pathlib, re, sys
import requests
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
import ood_filearm_v1 as fa
import detector_probe as dp

CASES = pathlib.Path("Drift-Evaluator/reports/ood_cases_v2.jsonl")
OUT = pathlib.Path("Drift-Evaluator/reports/postcutoff_detection.jsonl")
OLLAMA = "http://localhost:11434/v1/chat/completions"
OPEN_MODEL = "qwen2.5-coder:14b"
N_CAND = 2                       # qwen candidates per bug

# bug+patch judge (analogous to the trajectory judge, adapted to no-trajectory)
PATCH_JUDGE = """You are reviewing a candidate fix for a bug.

BUG:
{problem}

CANDIDATE PATCH:
{patch}

Did this patch FAIL to correctly fix the bug? Reply with ONLY a probability from
0.00 (definitely a correct fix) to 1.00 (definitely fails). Just the number."""


def open_judge_wb(problem, patch, timeout=120):
    """Open model P(fail) + verdict-token entropy (answer_entropy)."""
    try:
        r = requests.post(OLLAMA, json={
            "model": OPEN_MODEL, "temperature": 0.0, "max_tokens": 10,
            "logprobs": True, "top_logprobs": 5,
            "messages": [{"role": "user", "content": PATCH_JUDGE.format(problem=problem[:1500], patch=patch[:1800])}]},
            timeout=timeout)
        if r.status_code != 200:
            return {}
        ch = r.json()["choices"][0]
        p = dp.parse_prob(ch["message"]["content"])
        toks = (ch.get("logprobs") or {}).get("content") or []
        vt = next((t for t in toks if any(c.isdigit() for c in t.get("token", ""))), toks[0] if toks else None)
        ent = None
        if vt:
            ps = [math.exp(t["logprob"]) for t in (vt.get("top_logprobs") or [])]
            if ps:
                ent = -sum(q * math.log(q + 1e-12) for q in ps)
        return {"open_p_fail": p, "answer_entropy": ent}
    except Exception:
        return {}


def frontier_judge(problem, patch, model="claude-sonnet-4-6"):
    import subprocess, shutil
    cb = shutil.which("claude") or "/opt/homebrew/bin/claude"
    try:
        r = subprocess.run([cb, "-p", "--model", model],
                           input=PATCH_JUDGE.format(problem=problem[:1500], patch=patch[:1800]),
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return dp.parse_prob(r.stdout)
    except Exception:
        pass
    return None


def qwen_candidate(case):
    """Generate a wrong-or-right FILE-arm patch with qwen (SEARCH/REPLACE)."""
    from corrector_content import GEN
    prompt = GEN.format(repo=case["repo"], problem=case["problem"],
                        file=case["file"], code=case["file_content"][:6000], extra="")
    try:
        r = requests.post(OLLAMA, json={
            "model": OPEN_MODEL, "temperature": 0.7, "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}]}, timeout=180)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="django,pylint,pytest,scrapy")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    keep = tuple(args.repos.split(","))

    cases = [json.loads(l) for l in open(CASES) if l.strip()]
    cases = [c for c in cases if c["repo"].split("/")[-1] in keep]
    print(f"{len(cases)} post-cutoff bugs in {keep}", flush=True)

    done = {}
    if OUT.exists():
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); done[(r["iid"], r["kind"])] = r

    if not args.analyze_only:
        of = OUT.open("a")
        for c in cases:
            name = c["repo"].split("/")[-1]
            if not fa.ensure_env(name):
                print(f"  {c['iid']}: env failed", flush=True); continue
            gate = fa.gate_case(c)
            if not gate["evaluable"]:
                print(f"  {c['iid']}: not evaluable ({gate['reason'][:40]})", flush=True); continue
            diff = fa.pr_diff(c); test_d, _ = fa.split_diff(diff, c["test_files"])
            # GOLD = guaranteed-correct positive (gate already verified it passes)
            if (c["iid"], "gold") not in done:
                row = {"iid": c["iid"], "kind": "gold", "exec_pass": True,
                       "patch": c["gold"][:2500]}
                of.write(json.dumps(row) + "\n"); of.flush(); done[(c["iid"], "gold")] = row
            # qwen candidates, execution-labeled, TEXT SAVED
            for j in range(N_CAND):
                k = (c["iid"], f"qwen{j}")
                if k in done: continue
                cand = qwen_candidate(c)
                if not cand:
                    continue
                fa.checkout_base(name, c["base_sha"]); fa.apply_diff(name, test_d)
                applied, total = fa.apply_candidate(c, cand)
                if applied == 0:
                    row = {"iid": c["iid"], "kind": f"qwen{j}", "exec_pass": False,
                           "patch": cand[:2500], "note": "apply_failed"}
                else:
                    passed, tail = fa.run_tests(c)
                    if passed is None:
                        continue   # harness error, skip (don't mislabel)
                    row = {"iid": c["iid"], "kind": f"qwen{j}", "exec_pass": bool(passed),
                           "patch": cand[:2500]}
                of.write(json.dumps(row) + "\n"); of.flush(); done[k] = row
                print(f"  {c['iid']} qwen{j}: {'PASS' if row['exec_pass'] else 'fail'}", flush=True)

        # score every patch with both judges (resumable on score fields)
        prob = {c["iid"]: c["problem"] for c in cases}
        for k, row in list(done.items()):
            if row["iid"] not in prob:
                continue
            changed = False
            if row.get("frontier_p_fail") is None:
                row["frontier_p_fail"] = frontier_judge(prob[row["iid"]], row["patch"]); changed = True
            if row.get("answer_entropy") is None:
                row.update(open_judge_wb(prob[row["iid"]], row["patch"])); changed = True
            if changed:
                of.write(json.dumps(row) + "\n"); of.flush()
        of.close()

    # analyze: reload last-write-wins
    last = {}
    for l in open(OUT):
        if l.strip():
            r = json.loads(l); last[(r["iid"], r["kind"])] = {**last.get((r["iid"], r["kind"]), {}), **r}
    rows = list(last.values())
    labels = [not r["exec_pass"] for r in rows]   # positive = failed
    print(f"\n=== POST-CUTOFF detection (n={len(rows)}, "
          f"{sum(labels)} fail / {sum(not x for x in labels)} pass) ===")
    for d, key in [("frontier (sonnet)", "frontier_p_fail"),
                   ("open p_fail", "open_p_fail"),
                   ("open answer_entropy", "answer_entropy")]:
        a, n = dp.auc([r.get(key) for r in rows], labels)
        print(f"  {d:>22}: AUC {a if a is None else round(a,3)}  (n={n})")
    print("\npre-cutoff baseline: frontier 0.956 | open answer_entropy 0.814 (non-empty)")


if __name__ == "__main__":
    main()
