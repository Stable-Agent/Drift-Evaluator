#!/usr/bin/env python
"""Difficulty-controlled (within-problem matched) AUC for the drift detector.

Compares:
  (1) FULL pooled AUC over the near-disjoint resolved/failed pools  [potentially inflated]
  (2) MATCHED pooled AUC over only instances appearing in BOTH classes
  (3) WITHIN-INSTANCE paired concordance (same problem: does the detector rank
      the failed run > the resolved run?) -- pure behavioral signal
for both the structural detector and the NN-retrieval tier.
"""
from __future__ import annotations
import json, pathlib, collections, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from drift_detector import Detector

ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
RESOLVED_SUFFIXES = ["_held_out", "_resolved"]
FAILED_SUFFIXES = ["_held_out_failed"]

def load_msgs(f):
    obj = json.load(f.open())
    if isinstance(obj, dict):
        return obj.get("messages", []), (obj.get("info", {}) or {}).get("submission", "")
    return obj, ""

def collect():
    rows = []  # dict(instance, source, y, messages, patch, path)
    for d in ROOT.iterdir():
        if not d.is_dir() or d.name.endswith("_logs"):
            continue
        y = None
        for suf in FAILED_SUFFIXES:
            if d.name.endswith(suf): y = 1
        for suf in RESOLVED_SUFFIXES:
            if d.name.endswith(suf): y = 0
        if y is None:
            continue
        for f in d.glob("*.json"):
            msgs, patch = load_msgs(f)
            if not msgs:
                continue
            rows.append({"instance": f.stem, "source": d.name, "y": y,
                         "messages": msgs, "patch": patch})
    return rows

def paired_concordance(by_inst):
    """Mann-Whitney AUC restricted to within-instance pairs."""
    wins = ties = total = 0
    diffs = []
    for inst, (res_scores, fail_scores) in by_inst.items():
        if not res_scores or not fail_scores:
            continue
        r = np.mean(res_scores); fdr = np.mean(fail_scores)
        diffs.append(fdr - r)
        total += 1
        if fdr > r: wins += 1
        elif fdr == r: ties += 1
    auc = (wins + 0.5 * ties) / total if total else float("nan")
    return auc, total, np.array(diffs)

def main():
    print("loading trajectories...")
    rows = collect()
    # dedupe: same instance can appear in multiple source dirs w/ same label; keep all (different agents = different trajectories)
    by_class_inst = {0: set(), 1: set()}
    for r in rows:
        by_class_inst[r["y"]].add(r["instance"])
    res_inst, fail_inst = by_class_inst[0], by_class_inst[1]
    matched = res_inst & fail_inst
    print(f"  total trajectories: {len(rows)}")
    print(f"  resolved instances: {len(res_inst)}   failed instances: {len(fail_inst)}")
    print(f"  MATCHED (in both): {len(matched)}")

    print("\nscoring structural detector on every trajectory...")
    det = Detector()
    for r in rows:
        r["score"] = det.score(r["messages"], final_patch=r.get("patch"))
        r["n_msgs"] = len(r["messages"])

    y_all = np.array([r["y"] for r in rows])
    s_all = np.array([r["score"] for r in rows])
    print("\n================= STRUCTURAL DETECTOR =================")
    print(f"(1) FULL pooled AUC (disjoint pools, n={len(rows)}): "
          f"{roc_auc_score(y_all, s_all):.3f}")

    m_rows = [r for r in rows if r["instance"] in matched]
    ym = np.array([r["y"] for r in m_rows]); sm = np.array([r["score"] for r in m_rows])
    print(f"(2) MATCHED pooled AUC (n={len(m_rows)} trajs over {len(matched)} problems): "
          f"{roc_auc_score(ym, sm):.3f}")

    by_inst = {}
    for inst in matched:
        res = [r["score"] for r in rows if r["instance"]==inst and r["y"]==0]
        fail = [r["score"] for r in rows if r["instance"]==inst and r["y"]==1]
        by_inst[inst] = (res, fail)
    auc_p, n_p, diffs = paired_concordance(by_inst)
    print(f"(3) WITHIN-INSTANCE paired concordance ({n_p} matched problems): {auc_p:.3f}")
    print(f"    mean(score_failed - score_resolved) within problem: {diffs.mean():+.3f} "
          f"(>0 means detector ranks failed run higher)")

    # difficulty proxy check: n_msgs
    nm = np.array([r["n_msgs"] for r in rows])
    print(f"\n    [difficulty proxy] n_msgs FULL pool AUC: {roc_auc_score(y_all, nm):.3f}")
    nm_m = np.array([r["n_msgs"] for r in m_rows])
    print(f"    [difficulty proxy] n_msgs MATCHED pool AUC: {roc_auc_score(ym, nm_m):.3f}")

    # ---------- NN tier, restricted to matched, leave-one-INSTANCE-out ----------
    print("\n================= NN-RETRIEVAL TIER =================")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("  sentence-transformers unavailable:", e); return

    def msg_text(m):
        c = m.get("content","")
        if isinstance(c,str): return c
        if isinstance(c,list): return json.dumps(c)
        return ""
    def traj_text(msgs, mx=8000):
        lines=[f"{m.get('role','?')}: {msg_text(m)[:1500]}" for m in msgs if msg_text(m)]
        full="\n".join(lines); return full[-mx:] if len(full)>mx else full

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [traj_text(r["messages"]) for r in m_rows]
    emb = emb_model.encode(texts, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True)
    insts = np.array([r["instance"] for r in m_rows])
    K = 50
    nn_scores = np.full(len(m_rows), np.nan)
    for i in range(len(m_rows)):
        mask = insts != insts[i]            # leave-one-INSTANCE-out
        ref_emb = emb[mask]; ref_y = ym[mask]
        sims = ref_emb @ emb[i]
        k = min(K, len(ref_y))
        top = np.argsort(-sims)[:k]
        nn_scores[i] = ref_y[top].mean()
    print(f"(2) MATCHED pooled NN AUC (leave-instance-out, K={K}): "
          f"{roc_auc_score(ym, nn_scores):.3f}")
    # within-instance paired for NN
    nn_by_inst={}
    for inst in matched:
        idx=[j for j,r in enumerate(m_rows) if r['instance']==inst]
        res=[nn_scores[j] for j in idx if m_rows[j]['y']==0]
        fail=[nn_scores[j] for j in idx if m_rows[j]['y']==1]
        nn_by_inst[inst]=(res,fail)
    aucn,npn,dn=paired_concordance(nn_by_inst)
    print(f"(3) WITHIN-INSTANCE paired NN concordance ({npn} problems): {aucn:.3f}")
    print(f"    mean(nn_failed - nn_resolved) within problem: {dn.mean():+.3f}")

if __name__=="__main__":
    main()
