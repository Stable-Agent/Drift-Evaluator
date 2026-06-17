#!/usr/bin/env python
"""Nearest-neighbor failure-rate predictor.

For each held-out trajectory, find the K most-similar trajectories in
the training set (cosine similarity on sentence embeddings), and return
the failure rate of those neighbors as the prediction.

Compares against:
  - Structural detector (AUC 0.65 held-out)
  - Opus 4.7 judge (AUC 0.74 on n=66 pilot)
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from drift_detector import Detector

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
SOURCES = ["livesweagent_opus45", "livesweagent_gemini3",
           "sonar_opus45", "sonar_sonnet45"]


def msg_text(m):
    if "blocks" in m:
        parts = []
        for b in m.get("blocks") or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("block_type") or b.get("type")
            if bt == "text":
                parts.append(b.get("text") or "")
            elif bt in ("tool_use", "tool"):
                nm = b.get("name") or b.get("tool_name") or ""
                inp = b.get("input") or b.get("arguments") or ""
                parts.append(f"[{nm} {json.dumps(inp)[:300] if not isinstance(inp, str) else inp[:300]}]")
            elif bt in ("tool_result", "tool_output"):
                tr = b.get("content") or b.get("output") or ""
                if isinstance(tr, list):
                    tr = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in tr)
                parts.append(str(tr)[:1000])
        return "\n".join(parts)
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def load_traj(path):
    d = json.load(path.open())
    if isinstance(d, dict) and "messages" in d:
        return d["messages"]
    if isinstance(d, list):
        return d
    return []


def trajectory_text(messages, max_chars=8000):
    """Compact trajectory text for embedding. Keep last `max_chars`
    characters of concatenated role+text — that's where drift signal lives."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        text = msg_text(m)
        if not text:
            continue
        # Truncate per-turn to avoid one huge tool result dominating
        lines.append(f"{role}: {text[:1500]}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def build_dataset():
    rows = []
    for src in SOURCES:
        for suffix, label in [("_held_out", 0), ("_held_out_failed", 1)]:
            d = TRAJ_ROOT / f"{src}{suffix}"
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    msgs = load_traj(f)
                    if not msgs:
                        continue
                    text = trajectory_text(msgs)
                    if len(text) < 200:
                        continue  # too short to embed meaningfully
                    rows.append({
                        "source": src, "instance": f.stem, "y": label,
                        "text": text, "messages": msgs,
                    })
                except Exception as e:
                    print(f"skip {f.name}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    print("loading dataset...")
    rows = build_dataset()
    y = np.array([r["y"] for r in rows])
    print(f"  n={len(rows)}  ({y.sum()} failed, {(y==0).sum()} resolved)")

    print(f"\nloading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model)
    print("encoding trajectories...")
    embeddings = embedder.encode([r["text"] for r in rows],
                                  batch_size=32, show_progress_bar=True,
                                  normalize_embeddings=True)
    print(f"  shape: {embeddings.shape}")

    # INSTANCE-level split (not trajectory-level) to prevent leakage from
    # same-instance-different-agent trajectories appearing in both splits.
    instances = sorted({r["instance"] for r in rows})
    print(f"\nsplitting at instance level: {len(instances)} unique instances")
    import random as _r
    rng = _r.Random(args.seed)
    rng.shuffle(instances)
    n_test_inst = int(len(instances) * args.test_frac)
    test_instances = set(instances[:n_test_inst])
    tr_idx = np.array([i for i, r in enumerate(rows) if r["instance"] not in test_instances])
    te_idx = np.array([i for i, r in enumerate(rows) if r["instance"] in test_instances])
    print(f"train: n={len(tr_idx)} ({len({rows[i]['instance'] for i in tr_idx})} instances)")
    print(f"test:  n={len(te_idx)} ({len({rows[i]['instance'] for i in te_idx})} instances)")

    emb_tr = embeddings[tr_idx]
    emb_te = embeddings[te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    # Cosine similarity (since embeddings are normalized, dot product = cosine)
    sim_te_tr = emb_te @ emb_tr.T  # (n_test, n_train)

    print(f"\n=== K-NN held-out AUC (cosine sim → neighbor failure rate) ===")
    print(f"{'K':>3}  {'AUC':>6}  {'weighted-AUC':>14}")
    auc_per_k = {}
    for K in [1, 3, 5, 10, 20, 50, 100]:
        if K > len(tr_idx):
            continue
        # Top-K neighbors per test row
        top_k_idx = np.argsort(-sim_te_tr, axis=1)[:, :K]
        # Plain mean of neighbor labels
        scores = y_tr[top_k_idx].mean(axis=1)
        auc = roc_auc_score(y_te, scores)
        # Similarity-weighted average
        top_k_sim = np.take_along_axis(sim_te_tr, top_k_idx, axis=1)
        # Convert (-1,1) sim to weights (0,2) range, normalize
        w = top_k_sim - top_k_sim.min(axis=1, keepdims=True) + 1e-6
        w = w / w.sum(axis=1, keepdims=True)
        weighted_scores = (y_tr[top_k_idx] * w).sum(axis=1)
        auc_w = roc_auc_score(y_te, weighted_scores)
        auc_per_k[K] = (auc, auc_w)
        print(f"  {K:>3d}  {auc:.3f}    {auc_w:.3f}")

    # Best K
    best_K = max(auc_per_k.keys(), key=lambda k: auc_per_k[k][0])
    best_auc = auc_per_k[best_K][0]
    print(f"\nbest K = {best_K}, AUC = {best_auc:.3f}")

    # Compare to structural detector on the same test set
    print("\nscoring same test set with structural detector for comparison...")
    det = Detector()
    test_rows = [rows[i] for i in te_idx]
    struct_scores = np.array([det.score(r["messages"]) for r in test_rows])
    auc_struct = roc_auc_score(y_te, struct_scores)
    print(f"  structural AUC on this test split: {auc_struct:.3f}")

    # Combined: average struct + NN
    top_k_idx = np.argsort(-sim_te_tr, axis=1)[:, :best_K]
    nn_scores = y_tr[top_k_idx].mean(axis=1)
    combo = 0.5 * struct_scores + 0.5 * nn_scores
    print(f"  avg(struct, NN K={best_K}) AUC: {roc_auc_score(y_te, combo):.3f}")

    # AND-gate sweep
    print(f"\n=== AND-gate: struct AND NN ===")
    print(f"{'s_thr':>6}  {'nn_thr':>7}  {'TP/n_fired':>11}  {'precision':>10}  {'recall':>7}")
    P = int(y_te.sum())
    for s_thr in [0.5, 0.6, 0.7]:
        for nn_thr in [0.4, 0.5, 0.6, 0.7]:
            fires = (struct_scores > s_thr) & (nn_scores > nn_thr)
            tp = int((fires & (y_te == 1)).sum())
            fp = int((fires & (y_te == 0)).sum())
            n = tp + fp
            if n == 0: continue
            print(f"  {s_thr:.2f}    {nn_thr:.2f}     {tp:>4d}/{n:<6d}   {tp/n:>9.1%}   {tp/max(1,P):>6.1%}")

    # Save predictions for later 3-way combination with Opus judge
    out_path = pathlib.Path("Drift-Evaluator/reports/nn_retrieval.jsonl")
    with out_path.open("w") as fp:
        for i, ti in enumerate(te_idx):
            r = rows[ti]
            fp.write(json.dumps({
                "source": r["source"], "instance": r["instance"], "y": int(r["y"]),
                "struct_score": float(struct_scores[i]),
                "nn_score": float(nn_scores[i]),
            }) + "\n")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
