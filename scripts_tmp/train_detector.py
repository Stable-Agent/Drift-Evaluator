#!/usr/bin/env python
"""Train a drift detector on white-box + structural features, evaluated
HELD-OUT BY INSTANCE (train on some problems, test on unseen ones — the honest
generalization test). The question: does a learned combination beat the best
single feature (answer_entropy 0.814), and does it hold on unseen instances?

Features (all available for the full 300-rollout corpus):
  white-box: answer_entropy, margin, mean_logprob, p_fail (the verdict)
  structural: n_steps, stuck, rc_err_frac, max_diff_lines, final_patch_lines
Target: failed (execution-grounded).

Eval: GroupKFold by instance_id -> held-out pooled AUC + within-cluster
concordance, on FULL corpus and the NON-EMPTY (non-trivial) subset.
Reports the trained detector vs the best single feature and (if available)
the frontier judge baseline on the same rows.
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
import detector_probe as dp

WB_CACHE = "Drift-Evaluator/reports/whitebox_probe_cache.jsonl"
FR_CACHE = "Drift-Evaluator/reports/detector_probe_cache.jsonl"
WB_FEATS = ["answer_entropy", "margin", "mean_logprob", "p_fail"]
ST_FEATS = ["n_steps", "stuck", "rc_err_frac", "max_diff_lines", "final_patch_lines"]
FEATS = WB_FEATS + ST_FEATS


def load():
    rows = dp.load_corpus()
    wb = {(c["iid"], c["seed"]): c for c in
          (json.loads(l) for l in open(WB_CACHE)) if c.get("answer_entropy") is not None}
    fr = {}
    try:
        fr = {(c["iid"], c["seed"]): c.get("frontier_judge")
              for c in (json.loads(l) for l in open(FR_CACHE))}
    except FileNotFoundError:
        pass
    out = []
    for r in rows:
        k = (r["iid"], r["seed"]); w = wb.get(k)
        if not w:
            continue
        feat = {**{f: w.get(f) for f in WB_FEATS}, **{f: r["struct"][f] for f in ST_FEATS}}
        if any(feat[f] is None for f in FEATS):
            continue
        out.append({"iid": r["iid"], "failed": int(r["failed"]),
                    "empty": r["struct"]["empty_patch"], "feat": feat,
                    "frontier": fr.get(k)})
    return out


def held_out_scores(data):
    """GroupKFold by instance: out-of-fold predicted P(fail) for each row."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    X = np.array([[d["feat"][f] for f in FEATS] for d in data], float)
    y = np.array([d["failed"] for d in data])
    groups = np.array([d["iid"] for d in data])
    oof = np.full(len(data), np.nan)
    n_groups = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l1", solver="liblinear", C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof, X, y, groups


def auc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    pos = [s for s, l in pairs if l]; neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return None, len(pairs)
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg)), len(pairs)


def within_cluster(scores, data):
    from collections import defaultdict
    by = defaultdict(list)
    for s, d in zip(scores, data):
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            by[d["iid"]].append((s, d["failed"]))
    tot = w = inst = 0
    for vs in by.values():
        f = [s for s, y in vs if y]; p = [s for s, y in vs if not y]
        if not f or not p:
            continue
        inst += 1
        for a in f:
            for b in p:
                tot += 1; w += (a > b) + 0.5 * (a == b)
    return (w / tot if tot else None), tot, inst


def report(name, scores, data):
    sub = data
    a, n = auc(scores, [d["failed"] for d in sub])
    wc, pairs, inst = within_cluster(scores, sub)
    ne = [(s, d) for s, d in zip(scores, data) if not d["empty"]]
    ane, nne = auc([s for s, _ in ne], [d["failed"] for _, d in ne])
    wcne, pne, ine = within_cluster([s for s, _ in ne], [d for _, d in ne])
    f = lambda x: "None" if x is None else f"{x:.3f}"
    print(f"{name:>26}  all={f(a)}(n{n})  nonempty={f(ane)}(n{nne})  "
          f"wc_all={f(wc)}({pairs}p)  wc_nonempty={f(wcne)}({pne}p)")


def main():
    data = load()
    print(f"corpus with full features: {len(data)} "
          f"({sum(d['failed'] for d in data)} failed / {sum(1-d['failed'] for d in data)} resolved); "
          f"non-empty {sum(not d['empty'] for d in data)}\n")

    oof, X, y, groups = held_out_scores(data)
    print(f"{'detector':>26}  {'pooled+nonempty AUC, within-cluster (held-out by instance)'}")
    report("TRAINED (wb+struct, L1)", list(oof), data)
    # best single features for comparison
    for f in ["answer_entropy", "p_fail"]:
        report(f"single: {f}", [d["feat"][f] for d in data], data)
    # frontier baseline on rows that have it
    fr = [d["frontier"] for d in data]
    if any(x is not None for x in fr):
        report("frontier (sonnet)", fr, data)
    else:
        print("\n(frontier baseline still populating — re-run when detector_probe finishes)")

    # feature importances from a full-data fit (descriptive)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0).fit(Xs, y)
    coefs = sorted(zip(FEATS, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("\nL1 coefficients (full-data fit, standardized; +=>failure):")
    for fname, c in coefs:
        if abs(c) > 1e-6:
            print(f"  {fname:>20}: {c:+.3f}")


if __name__ == "__main__":
    main()
