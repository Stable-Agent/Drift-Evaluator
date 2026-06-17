#!/usr/bin/env python
"""Decisive test: is the within-problem signal drift, or just agent strength?

Split the 87 matched pairs by strength direction (opus = strong; gemini/sonnet
= weaker) and report within-problem concordance per bucket for the bundled
detector and the re-fit 2-feature under-iteration model.

Logic:
  - PURE strength artifact => resolved_stronger bucket high, resolved_weaker
    bucket BELOW 0.5 (model ranks the strong-but-failed run as less failure-y).
  - Real behavioral signal => stays >=0.5 even in the resolved_weaker bucket,
    where the signal must cut AGAINST strength.
"""
from __future__ import annotations
import json, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
from drift_detector import Detector
from drift_detector.features import FEATURES, extract_features

MANIFEST = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs/matched_corpus/manifest.jsonl")
STRONG = {"opus"}  # gemini/sonnet treated as weaker

def load(p):
    obj = json.load(open(p))
    if isinstance(obj, dict):
        return obj.get("messages", []), (obj.get("info", {}) or {}).get("submission", "")
    return obj, ""

# re-fit under-iteration model: failure predicted by FEWER reverts + intermediate diffs
W = {"n_intermediate_diffs": -0.285, "n_reverts": -0.204}
def under_iter_score(msgs, patch):
    f = extract_features(msgs, final_patch=patch)
    return sum(W[k] * f[k] for k in W)

def concord(diffs):
    diffs = np.array(diffs)
    if len(diffs) == 0: return float("nan"), 0
    w = (diffs > 0).sum(); t = (diffs == 0).sum()
    return (w + 0.5 * t) / len(diffs), len(diffs)

def main():
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    det = Detector()
    buckets = {"resolved_stronger": {"b": [], "u": []},
               "same_tier":         {"b": [], "u": []},
               "resolved_weaker":   {"b": [], "u": []}}
    for r in rows:
        rt, ft = r["resolved"]["tier"], r["failed"]["tier"]
        rs, fs = rt in STRONG, ft in STRONG
        if rs and not fs:   key = "resolved_stronger"
        elif fs and not rs: key = "resolved_weaker"
        elif rt == ft:      key = "same_tier"
        else:               key = "same_tier"  # weaker<->weaker (none here)
        fm, fp = load(r["failed"]["path"]); rm, rp = load(r["resolved"]["path"])
        if not fm or not rm: continue
        buckets[key]["b"].append(det.score(fm, fp) - det.score(rm, rp))
        buckets[key]["u"].append(under_iter_score(fm, fp) - under_iter_score(rm, rp))

    print(f"{'bucket':>18}  {'n':>3}  {'bundled':>8}  {'under-iter':>10}")
    print(f"{'(strength dir)':>18}  {'':>3}  {'concord':>8}  {'concord':>10}")
    for key in ["resolved_stronger", "same_tier", "resolved_weaker"]:
        cb, n = concord(buckets[key]["b"])
        cu, _ = concord(buckets[key]["u"])
        print(f"{key:>18}  {n:>3}  {cb:>8.3f}  {cu:>10.3f}")
    # pooled
    allb = sum((buckets[k]["b"] for k in buckets), [])
    allu = sum((buckets[k]["u"] for k in buckets), [])
    print(f"{'ALL':>18}  {len(allb):>3}  {concord(allb)[0]:>8.3f}  {concord(allu)[0]:>10.3f}")
    print("\nread: if 'resolved_weaker' is <0.5, the signal is agent-strength,")
    print("not drift. if it holds >=0.5, real behavioral signal survives.")
    print("(resolved_weaker n=14 is small — directional, not definitive.)")

if __name__ == "__main__":
    main()
