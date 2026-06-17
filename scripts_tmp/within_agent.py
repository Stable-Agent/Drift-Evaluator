#!/usr/bin/env python
"""Thread 1: within a SINGLE agent, does trajectory structure predict its own
pass/fail BEYOND task difficulty?

Holds model + scaffold constant (all trajectories from livesweagent_opus45),
so the agent-identity confound that killed the matched analysis is removed.
Difficulty is controlled statistically: difficulty_proxy = how many of the
OTHER 3 agents resolved the same instance (0-3; lower = harder).

Test: nested logistic regression.
  - Model A: difficulty_proxy only
  - Model B: difficulty_proxy + 10 structural features
If B's CV AUC > A's, there is within-agent behavioral signal beyond difficulty.
Caveat: failed vs resolved are still different PROBLEMS within this agent, so
the proxy only coarsely controls difficulty. The definitive version needs
same-agent SAME-problem multi-runs (data we'd generate).
"""
from __future__ import annotations
import json, pathlib, warnings, urllib.request
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from drift_detector.features import FEATURES, extract_features

ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
GH = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
AGENT = "livesweagent_opus45"
OTHERS = {
    "20251120_livesweagent_gemini-3-pro-preview",
    "20251205_sonar-foundation-agent_claude-opus-4-5",
    "20251103_sonar-foundation-agent_claude-sonnet-4-5",
}

def load(p):
    obj = json.load(open(p))
    if isinstance(obj, dict):
        return obj.get("messages", []), (obj.get("info", {}) or {}).get("submission", "")
    return obj, ""

def difficulty_map():
    """instance -> count of the 3 OTHER agents that resolved it."""
    cnt = {}
    for sub in OTHERS:
        try:
            res = json.loads(urllib.request.urlopen(
                f"{GH}/{sub}/results/results.json", timeout=30).read().decode())
            for inst in res.get("resolved", []):
                cnt[inst] = cnt.get(inst, 0) + 1
        except Exception as e:
            print(f"  warn: {sub}: {e}")
    return cnt

def collect():
    rows = []
    specs = [(f"{AGENT}_held_out", 0), (f"{AGENT}_resolved", 0),
             (f"{AGENT}_held_out_failed", 1)]
    for dname, y in specs:
        d = ROOT / dname
        if not d.exists(): continue
        for f in d.glob("*.json"):
            msgs, patch = load(f)
            if not msgs: continue
            rows.append({"inst": f.stem, "y": y, "msgs": msgs, "patch": patch})
    return rows

def main():
    print("fetching difficulty proxy (other agents' resolved sets)...")
    diff = difficulty_map()
    rows = collect()
    # dedupe by instance (keep one per instance per label)
    seen = set(); uniq = []
    for r in rows:
        key = (r["inst"], r["y"])
        if key in seen: continue
        seen.add(key); uniq.append(r)
    rows = uniq
    y = np.array([r["y"] for r in rows])
    print(f"  {AGENT}: n={len(rows)}  (failed={int(y.sum())}, resolved={int((y==0).sum())})")

    feats = np.array([[extract_features(r["msgs"], final_patch=r["patch"])[k]
                       for k in FEATURES] for r in rows], dtype=float)
    dproxy = np.array([[diff.get(r["inst"], 0)] for r in rows], dtype=float)

    # sanity: is difficulty proxy itself predictive? (it should be)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    def auc(X):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.5))
        return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

    a_diff = auc(dproxy)
    a_struct = auc(feats)
    a_both = auc(np.hstack([dproxy, feats]))
    print("\n=== within-agent 5-fold CV AUC ===")
    print(f"  difficulty proxy only:        {a_diff:.3f}")
    print(f"  structural features only:     {a_struct:.3f}")
    print(f"  difficulty + structural:      {a_both:.3f}")
    print(f"\n  structural LIFT over difficulty: {a_both - a_diff:+.3f}")
    if a_both - a_diff > 0.03:
        print("  => structure adds signal beyond difficulty within one agent.")
        print("     Justifies generating same-agent same-problem multi-run data.")
    else:
        print("  => structure adds little beyond difficulty even within one agent.")
        print("     Consistent with no recoverable behavioral signal here.")

    # difficulty distribution by outcome (is the proxy doing work?)
    print("\n  difficulty proxy mean: "
          f"resolved={dproxy[y==0].mean():.2f}  failed={dproxy[y==1].mean():.2f}")

if __name__ == "__main__":
    main()
