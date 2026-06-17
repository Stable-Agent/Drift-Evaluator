#!/usr/bin/env python
"""Step 0: re-fit the detector on the matched corpus, NN tier dropped.

Design: 1:1 matched pairs (same SWE-bench problem, one resolved + one failed
trajectory). The correct model for matched case-control is CONDITIONAL logistic
regression on within-pair feature differences Δ = x_failed - x_resolved. This
differences-out the problem-level intercept (difficulty), so the model cannot
relearn the selection bias documented in project_matched_auc_bias.md.

Reports:
  - leave-one-problem-out within-problem concordance of the re-fit model
    (honest achievable signal), vs the bundled detector and n_msgs
  - which of the 10 structural features survive L1 and their signs
  - whether a global-threshold operating point is viable, or if the signal is
    only usable per-problem-relative
"""
from __future__ import annotations
import json, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from drift_detector import Detector
from drift_detector.features import FEATURES, extract_features

MANIFEST = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs/matched_corpus/manifest.jsonl")

def load(p):
    obj = json.load(open(p))
    if isinstance(obj, dict):
        return obj.get("messages", []), (obj.get("info", {}) or {}).get("submission", "")
    return obj, ""

def featvec(msgs, patch):
    f = extract_features(msgs, final_patch=patch)
    return np.array([f[k] for k in FEATURES], dtype=float)

def concordance(deltas_signed):
    """deltas_signed: per-pair model score for failed minus for resolved."""
    w = (deltas_signed > 0).sum(); t = (deltas_signed == 0).sum()
    return (w + 0.5 * t) / len(deltas_signed)

def main():
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    X_fail, X_res, same_tier, nmsg_d = [], [], [], []
    det = Detector()
    bundled_d = []
    for r in rows:
        fm, fp = load(r["failed"]["path"]); rm, rp = load(r["resolved"]["path"])
        if not fm or not rm:
            continue
        xf, xr = featvec(fm, fp), featvec(rm, rp)
        X_fail.append(xf); X_res.append(xr)
        same_tier.append(r["resolved"]["tier"] == r["failed"]["tier"])
        nmsg_d.append(len(fm) - len(rm))
        bundled_d.append(det.score(fm, fp) - det.score(rm, rp))
    X_fail = np.array(X_fail); X_res = np.array(X_res)
    same_tier = np.array(same_tier); nmsg_d = np.array(nmsg_d)
    bundled_d = np.array(bundled_d)
    n = len(X_fail)
    Delta = X_fail - X_res                       # (n, 10)
    print(f"matched pairs: n={n}  (same-tier={int(same_tier.sum())})")

    # Scale Δ by std WITHOUT centering: the systematic directional offset of
    # each feature's difference (e.g. failed runs have larger diff-growth) IS
    # the signal; centering would erase it. with_mean=False keeps it.
    scaler = StandardScaler(with_mean=False)
    Dz = scaler.fit_transform(Delta)

    # Conditional logistic: stack (+Δ -> 1, -Δ -> 0), no intercept.
    Xc = np.vstack([Dz, -Dz]); yc = np.concatenate([np.ones(n), np.zeros(n)])

    def fit(Cval, train_idx=None):
        if train_idx is None:
            Xt, yt = Xc, yc
        else:
            Xt = np.vstack([Dz[train_idx], -Dz[train_idx]])
            yt = np.concatenate([np.ones(len(train_idx)), np.zeros(len(train_idx))])
        m = LogisticRegression(penalty="l1", solver="liblinear", C=Cval,
                               fit_intercept=False, max_iter=2000)
        m.fit(Xt, yt); return m

    # Leave-one-problem-out CV concordance across a few C values
    print("\n=== leave-one-problem-out within-problem concordance (re-fit model) ===")
    print(f"{'C':>8}  {'all-87':>8}  {'same-tier':>10}  {'nnz feats':>9}")
    best = None
    for C in [0.05, 0.1, 0.25, 0.5, 1.0]:
        loo = np.zeros(n)
        for i in range(n):
            tr = np.array([j for j in range(n) if j != i])
            m = fit(C, tr)
            loo[i] = m.decision_function(Dz[i:i+1])[0]   # score for +Δ (failed>res)
        c_all = concordance(loo)
        c_st = concordance(loo[same_tier])
        nnz = int((np.abs(fit(C).coef_[0]) > 1e-8).sum())
        print(f"{C:>8.2f}  {c_all:>8.3f}  {c_st:>10.3f}  {nnz:>9d}")
        if best is None or c_all > best[1]:
            best = (C, c_all, c_st)

    C_best = best[0]
    print(f"\nbest C={C_best}  all-87 concordance={best[1]:.3f}  same-tier={best[2]:.3f}")

    # Baselines for context
    print("\n=== baselines (within-problem concordance) ===")
    print(f"  re-fit conditional-LR (LOO):   {best[1]:.3f}")
    print(f"  bundled detector (shipped):    {concordance(bundled_d):.3f}")
    print(f"  n_msgs delta alone:            {concordance(nmsg_d.astype(float)):.3f}")

    # Final model coefficients on full data
    m = fit(C_best)
    coefs = m.coef_[0]
    print(f"\n=== surviving features (C={C_best}, standardized Δ coefs; + => predicts FAILURE) ===")
    order = np.argsort(-np.abs(coefs))
    for j in order:
        if abs(coefs[j]) > 1e-8:
            print(f"  {FEATURES[j]:26s} {coefs[j]:+.3f}")
    dropped = [FEATURES[j] for j in range(len(FEATURES)) if abs(coefs[j]) <= 1e-8]
    if dropped:
        print(f"  (L1-dropped: {', '.join(dropped)})")

    # Operating-point reality check: can a GLOBAL threshold work?
    print("\n=== operating-point reality check ===")
    # Score each trajectory absolutely with the re-fit model (apply to standardized
    # ABSOLUTE features, not differences) to see if a global cut separates classes.
    Xall = np.vstack([X_fail, X_res])
    yall = np.concatenate([np.ones(n), np.zeros(n)])
    sc_abs = StandardScaler().fit(Xall)
    abs_scores = sc_abs.transform(Xall) @ coefs            # linear score, abs features
    pooled_auc = roc_auc_score(yall, abs_scores)
    print(f"  pooled AUC of re-fit model on ABSOLUTE features: {pooled_auc:.3f}")
    print(f"  (vs within-problem concordance {best[1]:.3f})")
    if pooled_auc < best[1] - 0.04:
        print("  => signal is largely WITHIN-problem; a global fixed threshold")
        print("     leaves most of it on the table. Hook needs per-problem-relative")
        print("     scoring or it will mostly fire on hard problems, not on drift.")
    else:
        print("  => global threshold captures most of the signal; hook is viable.")

if __name__ == "__main__":
    main()
