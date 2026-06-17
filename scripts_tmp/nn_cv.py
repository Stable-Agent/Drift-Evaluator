#!/usr/bin/env python
"""5-fold instance-level cross-validation of the NN retrieval signal."""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
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
    lines = []
    for m in messages:
        text = msg_text(m)
        if not text: continue
        lines.append(f"{m.get('role', '?')}: {text[:1500]}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def build_dataset():
    rows = []
    for src in SOURCES:
        for suffix, label in [("_held_out", 0), ("_held_out_failed", 1)]:
            d = TRAJ_ROOT / f"{src}{suffix}"
            if not d.exists(): continue
            for f in d.glob("*.json"):
                try:
                    msgs = load_traj(f)
                    if not msgs: continue
                    text = trajectory_text(msgs)
                    if len(text) < 200: continue
                    rows.append({"source": src, "instance": f.stem, "y": label,
                                  "text": text, "messages": msgs})
                except Exception as e:
                    print(f"skip {f.name}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("loading dataset...")
    rows = build_dataset()
    y = np.array([r["y"] for r in rows])
    print(f"  n={len(rows)}  ({y.sum()} failed, {(y==0).sum()} resolved)")

    print("\nencoding...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode([r["text"] for r in rows],
                                  batch_size=64, show_progress_bar=False,
                                  normalize_embeddings=True)
    print(f"  shape: {embeddings.shape}")

    # Score struct once on the whole set (it's already fully trained)
    det = Detector()
    struct_scores = np.array([det.score(r["messages"]) for r in rows])

    # Instance-level CV
    instances = sorted({r["instance"] for r in rows})
    import random as _r
    rng = _r.Random(args.seed)
    rng.shuffle(instances)
    fold_size = len(instances) // args.n_folds

    print(f"\n{args.n_folds}-fold instance-level CV ({len(instances)} unique instances):")
    print(f"{'fold':>4}  " + "  ".join(f"K={K:<3}" for K in args.K) + "  struct  combo_K50")
    results = {K: [] for K in args.K}
    struct_aucs = []
    combo_aucs = []
    for fold in range(args.n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < args.n_folds - 1 else len(instances)
        test_inst = set(instances[start:end])
        tr_idx = np.array([i for i, r in enumerate(rows) if r["instance"] not in test_inst])
        te_idx = np.array([i for i, r in enumerate(rows) if r["instance"] in test_inst])
        emb_tr, emb_te = embeddings[tr_idx], embeddings[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sim = emb_te @ emb_tr.T

        line = [f"{fold:>4}"]
        for K in args.K:
            top_k = np.argsort(-sim, axis=1)[:, :K]
            scores = y_tr[top_k].mean(axis=1)
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                line.append("  n/a")
                continue
            a = roc_auc_score(y_te, scores)
            results[K].append(a)
            line.append(f"  {a:.3f}")

        s_struct = struct_scores[te_idx]
        if y_te.sum() > 0 and y_te.sum() < len(y_te):
            a_struct = roc_auc_score(y_te, s_struct)
            struct_aucs.append(a_struct)
            line.append(f"   {a_struct:.3f}")
            top_k50 = np.argsort(-sim, axis=1)[:, :50]
            scores50 = y_tr[top_k50].mean(axis=1)
            combo = 0.5 * s_struct + 0.5 * scores50
            a_combo = roc_auc_score(y_te, combo)
            combo_aucs.append(a_combo)
            line.append(f"   {a_combo:.3f}")
        print("  ".join(line))

    print(f"\n=== Mean ± std across {args.n_folds} folds ===")
    for K in args.K:
        a = np.array(results[K])
        print(f"  NN K={K:<3}  mean={a.mean():.3f}  std={a.std():.3f}  range=[{a.min():.3f}, {a.max():.3f}]")
    s = np.array(struct_aucs)
    c = np.array(combo_aucs)
    print(f"  struct alone   mean={s.mean():.3f}  std={s.std():.3f}  range=[{s.min():.3f}, {s.max():.3f}]")
    print(f"  combo K=50    mean={c.mean():.3f}  std={c.std():.3f}  range=[{c.min():.3f}, {c.max():.3f}]")

    # Operating points at best K
    best_K = max(args.K, key=lambda k: np.mean(results[k]))
    print(f"\n=== Pooled CV operating points at K={best_K} ===")
    print(f"{'s_thr':>6}  {'nn_thr':>7}  {'TP/n_fired':>11}  {'precision':>10}  {'recall':>7}")

    # Pool: for each test fold predictions and stack
    all_y, all_struct, all_nn = [], [], []
    for fold in range(args.n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < args.n_folds - 1 else len(instances)
        test_inst = set(instances[start:end])
        tr_idx = np.array([i for i, r in enumerate(rows) if r["instance"] not in test_inst])
        te_idx = np.array([i for i, r in enumerate(rows) if r["instance"] in test_inst])
        sim = embeddings[te_idx] @ embeddings[tr_idx].T
        top_k = np.argsort(-sim, axis=1)[:, :best_K]
        nn_scores = y[tr_idx][top_k].mean(axis=1)
        all_y.append(y[te_idx])
        all_struct.append(struct_scores[te_idx])
        all_nn.append(nn_scores)
    all_y = np.concatenate(all_y)
    all_struct = np.concatenate(all_struct)
    all_nn = np.concatenate(all_nn)
    P = int(all_y.sum())
    print(f"  pooled n={len(all_y)}  P={P}  N={(all_y==0).sum()}")
    for s_thr in [0.5, 0.6, 0.7]:
        for nn_thr in [0.4, 0.5, 0.6, 0.7]:
            fires = (all_struct > s_thr) & (all_nn > nn_thr)
            tp = int((fires & (all_y == 1)).sum())
            fp = int((fires & (all_y == 0)).sum())
            n = tp + fp
            if n == 0: continue
            print(f"  {s_thr:.2f}    {nn_thr:.2f}     {tp:>4d}/{n:<6d}   {tp/n:>9.1%}   {tp/max(1,P):>6.1%}")


if __name__ == "__main__":
    main()
