#!/usr/bin/env python
"""Score every held-out RESOLVED trajectory and report FP rate per threshold.

Uses the shipped drift_detector package directly (the one in the
Drift-Detector pypi package). This validates the FP claim from
project_multi_signal_classifier_v0 on FRESH resolved trajectories the
model has never seen.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from drift_detector import Detector

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
SOURCES = ["livesweagent_opus45", "livesweagent_gemini3", "sonar_opus45", "sonar_sonnet45"]


def load_trajectory(path: pathlib.Path) -> tuple[list[dict], str]:
    d = json.load(path.open())
    msgs = d["messages"] if isinstance(d, dict) and "messages" in d else d
    final_patch = (d.get("info") or {}).get("submission") if isinstance(d, dict) else ""
    return (msgs if isinstance(msgs, list) else []), (final_patch or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Drift-Evaluator/reports/fp_validation_held_out.json")
    args = ap.parse_args()

    det = Detector()
    print(f"loaded detector (in-sample AUC {det._bundle.get('in_sample_auc', '?'):.3f})")

    rows = []  # (source, instance_id, score)
    for short in SOURCES:
        held_dir = TRAJ_ROOT / f"{short}_held_out"
        if not held_dir.exists():
            print(f"  {short}: no held-out dir, skipping")
            continue
        files = sorted(held_dir.glob("*.json"))
        print(f"  {short}: scoring {len(files)} held-out resolved trajectories...")
        for f in files:
            try:
                msgs, final_patch = load_trajectory(f)
                if not msgs:
                    continue
                score = det.score(msgs, final_patch=final_patch)
            except Exception as e:
                print(f"    skip {f.name}: {e}", file=sys.stderr)
                continue
            rows.append({"source": short, "instance_id": f.stem, "score": score})

    if not rows:
        sys.exit("no trajectories scored")

    scores = np.array([r["score"] for r in rows])
    n = len(scores)
    print(f"\n=== FP rate on {n} held-out resolved trajectories ===")
    print(f"{'threshold':>10}  {'FP':>4}/{'n':<4}  {'rate':>6}  {'95% CI':>14}")
    summary = {"n": n, "by_threshold": {}, "by_source": {}}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        fp = int((scores > thr).sum())
        rate = fp / n
        # Wilson 95% CI for proportion
        z = 1.96
        denom = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denom
        half = z * ((rate * (1 - rate) / n + z**2 / (4 * n * n)) ** 0.5) / denom
        ci_lo = max(0, center - half)
        ci_hi = min(1, center + half)
        print(f"  {thr:>8.2f}  {fp:>4d}/{n:<4d}  {rate:>5.1%}  [{ci_lo:>5.1%}, {ci_hi:>5.1%}]")
        summary["by_threshold"][f"{thr:.2f}"] = {
            "fp": fp, "n": n, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
        }

    print(f"\n=== Per-source breakdown (at threshold=0.5) ===")
    for src in SOURCES:
        src_scores = np.array([r["score"] for r in rows if r["source"] == src])
        if len(src_scores) == 0:
            print(f"  {src}: no data")
            continue
        fp = int((src_scores > 0.5).sum())
        print(f"  {src}: {fp}/{len(src_scores)} = {fp/len(src_scores):.1%}")
        summary["by_source"][src] = {"n": len(src_scores), "fp_at_0.5": fp}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
