#!/usr/bin/env python
"""Difficulty-controlled AUC over the matched_corpus (built by build_matched_corpus.py).

Each problem has exactly one resolved + one failed trajectory => within-problem
pairing is exact. Reports structural detector, NN tier, and n_msgs baseline on
(a) all matched pairs and (b) the same-agent-tier subset (cleanest: controls
for both difficulty AND agent strength).
"""
from __future__ import annotations
import json, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from drift_detector import Detector

MANIFEST = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs/matched_corpus/manifest.jsonl")

def load_msgs(p):
    obj = json.load(open(p))
    if isinstance(obj, dict):
        return obj.get("messages", []), (obj.get("info", {}) or {}).get("submission", "")
    return obj, ""

def msg_text(m):
    c = m.get("content", "")
    return c if isinstance(c, str) else (json.dumps(c) if isinstance(c, list) else "")

def traj_text(msgs, mx=8000):
    lines = [f"{m.get('role','?')}: {msg_text(m)[:1500]}" for m in msgs if msg_text(m)]
    full = "\n".join(lines); return full[-mx:] if len(full) > mx else full

def report(name, pairs, det, embed=None):
    """pairs: list of dicts with res_score, fail_score, res_nmsg, fail_nmsg, (res_emb, fail_emb)"""
    n = len(pairs)
    # within-problem concordance (structural)
    def concord(key):
        w = t = 0
        for p in pairs:
            d = p[f"fail_{key}"] - p[f"res_{key}"]
            if d > 0: w += 1
            elif d == 0: t += 1
        return (w + 0.5 * t) / n
    # pooled AUC
    y = np.array([0, 1] * n)  # res, fail interleaved
    s = np.array([v for p in pairs for v in (p["res_score"], p["fail_score"])])
    nm = np.array([v for p in pairs for v in (p["res_nmsg"], p["fail_nmsg"])])
    print(f"\n--- {name} (n={n} matched problems) ---")
    print(f"  structural: within-problem concordance={concord('score'):.3f}   "
          f"pooled AUC={roc_auc_score(y, s):.3f}")
    print(f"  n_msgs:     within-problem concordance={concord('nmsg'):.3f}   "
          f"pooled AUC={roc_auc_score(y, nm):.3f}")
    if embed is not None:
        # leave-one-PROBLEM-out NN: ref = all trajs from other problems
        all_emb = np.array([v for p in pairs for v in (p["res_emb"], p["fail_emb"])])
        all_y = y
        prob_of = np.array([i for i in range(n) for _ in range(2)])
        nn = np.full(2 * n, np.nan)
        for idx in range(2 * n):
            mask = prob_of != prob_of[idx]
            sims = all_emb[mask] @ all_emb[idx]
            ref_y = all_y[mask]
            k = min(50, len(ref_y))
            nn[idx] = ref_y[np.argsort(-sims)[:k]].mean()
        # concordance from nn
        w = t = 0
        for i in range(n):
            d = nn[2*i+1] - nn[2*i]
            if d > 0: w += 1
            elif d == 0: t += 1
        print(f"  NN tier:    within-problem concordance={(w+0.5*t)/n:.3f}   "
              f"pooled AUC={roc_auc_score(all_y, nn):.3f}")

def main():
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    print(f"loaded {len(rows)} matched pairs")
    det = Detector()
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    pairs = []
    for r in rows:
        rm, fm = load_msgs(r["resolved"]["path"]), load_msgs(r["failed"]["path"])
        if not rm[0] or not fm[0]:
            continue
        rt = emb_model.encode(traj_text(rm[0]), normalize_embeddings=True)
        ft = emb_model.encode(traj_text(fm[0]), normalize_embeddings=True)
        pairs.append({
            "same_tier": r["resolved"]["tier"] == r["failed"]["tier"],
            "res_score": det.score(rm[0], final_patch=rm[1]),
            "fail_score": det.score(fm[0], final_patch=fm[1]),
            "res_nmsg": len(rm[0]), "fail_nmsg": len(fm[0]),
            "res_emb": rt, "fail_emb": ft,
        })

    report("ALL matched pairs", pairs, det, embed=True)
    same = [p for p in pairs if p["same_tier"]]
    report("SAME-TIER subset (cleanest)", same, det, embed=True)

if __name__ == "__main__":
    main()
