#!/usr/bin/env python
"""Repetition-signal feature extraction + AND-gate analysis.

Three new features per trajectory:
  - max_diff_similarity:       max Jaccard similarity between consecutive
                               intermediate diffs (high = stuck on same patch)
  - max_file_read_repeats:     max times the agent re-read the same file
                               (high = uncertainty / not making progress)
  - max_hunk_header_repeats:   max times the same @@ hunk header appears
                               across intermediate diffs (high = revising
                               same location repeatedly)

Tests whether these are uncorrelated enough with the structural-score to
make an AND-gate work (one signal at p=0.78 AND another at p=0.85 →
joint precision ~95% if near-independent).
"""

from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
from drift_detector import Detector

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
SOURCES = ["livesweagent_opus45", "livesweagent_gemini3",
           "sonar_opus45", "sonar_sonnet45"]

# Match each `diff --git` block
RE_DIFF = re.compile(r"diff --git a/(\S+) b/.+?(?=(?:diff --git|</output>|\Z))", re.DOTALL)
RE_HUNK = re.compile(r"^@@ -(\d+),\d+ \+\d+,\d+ @@", re.MULTILINE)
# File-read patterns: cat, less, head, tail, view targeting a path with /
RE_FILE_READ = re.compile(r"\b(?:cat|less|head|tail|view|sed -n)\s+([^\s|;><]+/[^\s|;><]+)")


def msg_text(m):
    """Handle 3 trajectory formats:
    - {'content': str}           — livesweagent / mini-swe-agent
    - {'content': [blocks]}      — Claude-style block list
    - {'blocks': [block dicts]}  — sonar-foundation-agent
    """
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
                parts.append(f"[{nm} {json.dumps(inp)[:500] if not isinstance(inp, str) else inp[:500]}]")
            elif bt in ("tool_result", "tool_output"):
                tr = b.get("content") or b.get("output") or ""
                if isinstance(tr, list):
                    tr = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in tr)
                parts.append(str(tr)[:2000])
            elif bt == "thinking":
                continue  # don't count thinking for surface signals
        return "\n".join(parts)
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def extract_diff_blocks(messages):
    """Yield (msg_idx, full_diff_text) for each diff --git block in tool results."""
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        text = msg_text(m)
        if "diff --git" not in text:
            continue
        for match in RE_DIFF.finditer(text):
            yield i, match.group(0).strip()


def jaccard_lines(a, b):
    """Jaccard similarity of unique non-trivial lines."""
    sa = {l.strip() for l in a.splitlines() if l.strip() and not l.startswith("@@")}
    sb = {l.strip() for l in b.splitlines() if l.strip() and not l.startswith("@@")}
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def repetition_features(messages):
    diffs = [d for _, d in extract_diff_blocks(messages)]
    # max consecutive-pair similarity
    max_sim = 0.0
    for a, b in zip(diffs[:-1], diffs[1:]):
        max_sim = max(max_sim, jaccard_lines(a, b))

    # file-read repeats: count "cat path/x" etc. across all assistant turns
    file_reads = Counter()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for path in RE_FILE_READ.findall(msg_text(m)):
            file_reads[path] += 1
    max_repeats = max(file_reads.values(), default=0)

    # hunk header repeats across intermediate diffs
    hunk_counter = Counter()
    for d in diffs:
        for hdr in RE_HUNK.findall(d):
            hunk_counter[hdr] += 1
    max_hunk_repeats = max(hunk_counter.values(), default=0)

    return {
        "max_diff_similarity": max_sim,
        "max_file_read_repeats": max_repeats,
        "max_hunk_header_repeats": max_hunk_repeats,
    }


def load_dataset():
    """Same dataset retrain_representative.py uses."""
    rows = []
    for src in SOURCES:
        for suffix, label in [("_held_out", 0), ("_held_out_failed", 1)]:
            d = TRAJ_ROOT / f"{src}{suffix}"
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    traj = json.load(f.open())
                    if isinstance(traj, dict) and "messages" in traj:
                        msgs = traj["messages"]
                        final_patch = (traj.get("info") or {}).get("submission") or ""
                    elif isinstance(traj, list):
                        msgs = traj
                        final_patch = ""
                    else:
                        continue
                    if not isinstance(msgs, list):
                        continue
                    rep = repetition_features(msgs)
                    rows.append({
                        "source": src, "instance": f.stem, "y": label,
                        "final_patch": final_patch,
                        "messages": msgs,
                        **rep,
                    })
                except Exception as e:
                    print(f"  skip {f.name}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    rows = load_dataset()
    print(f"loaded {len(rows)} trajectories ({sum(r['y'] for r in rows)} failed, {sum(1 for r in rows if r['y']==0)} resolved)")

    # Univariate signal of each repetition feature
    print(f"\n=== Univariate signal of repetition features ===")
    print(f"{'feature':32s} {'mean(resolved)':>15s} {'mean(failed)':>15s} {'AUC':>6}")
    from sklearn.metrics import roc_auc_score
    y = np.array([r["y"] for r in rows])
    for feat in ["max_diff_similarity", "max_file_read_repeats", "max_hunk_header_repeats"]:
        vals = np.array([r[feat] for r in rows], dtype=float)
        if vals.std() == 0:
            print(f"  {feat:32s} {vals[y==0].mean():>15.3f} {vals[y==1].mean():>15.3f}  {'(constant)':>6}")
            continue
        auc = roc_auc_score(y, vals)
        print(f"  {feat:32s} {vals[y==0].mean():>15.3f} {vals[y==1].mean():>15.3f}  {auc:>6.3f}")

    # Combine with structural score: AND-gate at various thresholds
    det = Detector()
    print(f"\nscoring all {len(rows)} trajectories with the structural detector...")
    struct_scores = np.array([det.score(r["messages"], final_patch=r["final_patch"])
                              for r in rows])
    print(f"  done. mean(resolved)={struct_scores[y==0].mean():.3f} "
          f"mean(failed)={struct_scores[y==1].mean():.3f}")

    # Correlations
    print(f"\n=== Correlations (rep features vs struct score) ===")
    for feat in ["max_diff_similarity", "max_file_read_repeats", "max_hunk_header_repeats"]:
        vals = np.array([r[feat] for r in rows], dtype=float)
        if vals.std() == 0: continue
        corr = np.corrcoef(vals, struct_scores)[0, 1]
        print(f"  {feat:32s} corr with struct_score: {corr:+.3f}")

    # AND-gate sweep — for each operating point (struct_thr, rep_thr, rep_feat),
    # measure precision/recall on the WHOLE dataset (no train/test split since
    # we're just analyzing the joint distribution)
    print(f"\n=== AND-gate precision/recall sweep ===")
    print(f"{'struct_thr':>10s} {'rep_feat':32s} {'rep_thr':>8s} {'TP/n_fired':>11s} {'precision':>10s} {'recall':>7s}")
    n_failed = int(y.sum())
    for struct_thr in [0.5, 0.6, 0.7]:
        struct_fires = struct_scores > struct_thr
        struct_alone_tp = int((struct_fires & (y == 1)).sum())
        struct_alone_fp = int((struct_fires & (y == 0)).sum())
        n_alone = struct_alone_tp + struct_alone_fp
        prec_alone = struct_alone_tp / max(1, n_alone)
        rec_alone = struct_alone_tp / max(1, n_failed)
        print(f"  {struct_thr:>8.2f}  {'(structural alone)':32s} {'-':>8s} {struct_alone_tp:>4d}/{n_alone:<6d} {prec_alone:>9.1%}  {rec_alone:>6.1%}")
        for feat in ["max_diff_similarity", "max_file_read_repeats", "max_hunk_header_repeats"]:
            vals = np.array([r[feat] for r in rows], dtype=float)
            if vals.std() == 0: continue
            # Try several thresholds for the repetition feature
            for rep_thr in sorted(set(np.percentile(vals, [50, 75, 90, 95]))):
                joint = struct_fires & (vals > rep_thr)
                tp = int((joint & (y == 1)).sum())
                fp = int((joint & (y == 0)).sum())
                n = tp + fp
                if n == 0:
                    continue
                prec = tp / n
                rec = tp / max(1, n_failed)
                print(f"  {struct_thr:>8.2f}  {feat:32s} {rep_thr:>8.2f} {tp:>4d}/{n:<6d} {prec:>9.1%}  {rec:>6.1%}")

    # Save dataset with all features for downstream use
    out = pathlib.Path("Drift-Evaluator/reports/repetition_signals_v0.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r, s in zip(rows, struct_scores):
            f.write(json.dumps({
                "source": r["source"], "instance": r["instance"], "y": r["y"],
                "struct_score": float(s),
                "max_diff_similarity": r["max_diff_similarity"],
                "max_file_read_repeats": r["max_file_read_repeats"],
                "max_hunk_header_repeats": r["max_hunk_header_repeats"],
            }) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
