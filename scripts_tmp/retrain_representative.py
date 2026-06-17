#!/usr/bin/env python
"""Retrain the L1 logistic on a representative held-out sample.

Uses the new *_held_out/ (resolved) and *_held_out_failed/ (failed)
trajectories — each ~100 per source, random sample, NOT in the
original training set. Stratified 80/20 split by (source, class).
Reports held-out AUC, FP rate, precision/recall per threshold.

If the new AUC is meaningful (e.g., > 0.6), saves the new model to
Drift-Detector/src/drift_detector/data/non_llm_detector_v0.pkl
(overwrites in place; existing copy is preserved in git history).
"""

from __future__ import annotations
import argparse
import json
import pathlib
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drift_detector import FEATURES, extract_features

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
SOURCES = ["livesweagent_opus45", "livesweagent_gemini3",
           "sonar_opus45", "sonar_sonnet45"]
MODEL_OUT = pathlib.Path("Drift-Detector/src/drift_detector/data/non_llm_detector_v0.pkl")


def load_trajectory(path: pathlib.Path) -> tuple[list[dict], str]:
    d = json.load(path.open())
    msgs = d["messages"] if isinstance(d, dict) and "messages" in d else d
    final_patch = (d.get("info") or {}).get("submission") if isinstance(d, dict) else ""
    return (msgs if isinstance(msgs, list) else []), (final_patch or "")


def build_dataset():
    rows = []  # (source, instance, y, features dict)
    for src in SOURCES:
        for suffix, label in [("_held_out", 0), ("_held_out_failed", 1)]:
            d = TRAJ_ROOT / f"{src}{suffix}"
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    msgs, fp = load_trajectory(f)
                    if not msgs:
                        continue
                    feats = extract_features(msgs, final_patch=fp)
                    rows.append({"source": src, "instance": f.stem, "y": label, **feats})
                except Exception as e:
                    print(f"  skip {f.name}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-model", action="store_true",
                    help="overwrite the bundled model file (be sure first!)")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build_dataset()
    print(f"loaded {len(rows)} trajectories")
    y = np.array([r["y"] for r in rows])
    print(f"  class balance: {y.sum()} failed / {(y == 0).sum()} resolved")
    src_arr = np.array([r["source"] for r in rows])
    for src in SOURCES:
        m = src_arr == src
        if m.sum() == 0:
            print(f"  {src}: 0 (no data)")
            continue
        print(f"  {src}: {m.sum()} (failed: {y[m].sum()}, resolved: {(y[m] == 0).sum()})")

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)

    # Stratified split — by (source, class) so each split is representative
    strata = np.array([f"{r['source']}_{r['y']}" for r in rows])
    X_tr, X_te, y_tr, y_te, strata_tr, strata_te = train_test_split(
        X, y, strata, test_size=0.2, random_state=args.seed, stratify=strata
    )
    print(f"\ntrain: n={len(X_tr)}  test: n={len(X_te)}")

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=args.C,
                                  penalty="l1", solver="liblinear")),
    ])
    pipe.fit(X_tr, y_tr)

    # Held-out evaluation
    test_scores = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, test_scores)
    print(f"\n=== Held-out test set (n={len(X_te)}) ===")
    print(f"  AUC: {auc:.3f}")

    # Per-threshold FP / precision / recall
    print(f"\n{'thr':>6}  {'TP':>4}/{'P':<4}  {'recall':>7}  {'FP':>4}/{'N':<4}  {'precision':>10}  {'FP rate':>8}")
    P = int(y_te.sum())
    N = int((y_te == 0).sum())
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pred = (test_scores > thr).astype(int)
        tp = int(((pred == 1) & (y_te == 1)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        recall = tp / max(1, P)
        precision = tp / max(1, tp + fp)
        fp_rate = fp / max(1, N)
        print(f"  {thr:.2f}  {tp:>4d}/{P:<4d}  {recall:>6.1%}   {fp:>4d}/{N:<4d}  {precision:>9.1%}   {fp_rate:>7.1%}")

    # Per-source AUC on test
    print(f"\n=== Per-source test AUC ===")
    for src in SOURCES:
        m = strata_te == f"{src}_1"  # this is a heuristic; need raw source array
    # Build aligned source array for X_te
    rows_te_idx = []  # not directly available; recompute by mapping back
    # Easier: just split by index pattern
    src_te = np.array([s.split("_")[0] + "_" + s.split("_")[1] for s in strata_te])  # imperfect
    # Just print per-stratum
    for stratum in sorted(set(strata_te)):
        m = strata_te == stratum
        if m.sum() < 5:
            continue
        ys = y_te[m]; ss = test_scores[m]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        # Within stratum, all one class, so AUC undefined per stratum.
        # Instead show per-source by aggregating both classes:
        pass
    # Just per-source aggregate
    # Reconstruct source from strata_te ("<src>_0" or "<src>_1")
    src_te = np.array(["_".join(s.split("_")[:-1]) for s in strata_te])
    for src in SOURCES:
        m = src_te == src
        if m.sum() == 0:
            continue
        ys = y_te[m]; ss = test_scores[m]
        if ys.sum() == 0 or ys.sum() == len(ys):
            print(f"  {src}: n={m.sum()} no class variance")
            continue
        a = roc_auc_score(ys, ss)
        print(f"  {src}: n={m.sum()} (P={ys.sum()}, N={(ys==0).sum()})  AUC={a:.3f}")

    # Coefficients
    coefs = pipe.named_steps["lr"].coef_[0]
    print(f"\n=== Coefficients (standardized) ===")
    for f, c in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
        direction = "→ failure" if c > 0 else "→ success"
        print(f"  {f:32s} {c:>+8.3f}  {direction}")

    if args.save_model:
        # Refit on ALL data (train + test) for the saved model
        pipe.fit(X, y)
        bundle = {
            "pipeline": pipe,
            "features": FEATURES,
            "training_n": len(X),
            "in_sample_auc": float(roc_auc_score(y, pipe.predict_proba(X)[:, 1])),
            "trained_on": "representative held_out + held_out_failed sample (2026-05-26)",
            "held_out_auc": float(auc),
        }
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_OUT.open("wb") as f:
            pickle.dump(bundle, f)
        print(f"\nsaved model: {MODEL_OUT}")
    else:
        print(f"\n(not saving — pass --save-model to overwrite the bundled model)")


if __name__ == "__main__":
    main()
