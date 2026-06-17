#!/usr/bin/env python
"""Pull held-out resolved SWE-bench trajectories for FP validation.

For each of the 4 sources we trained on, downloads N resolved trajectories
that are NOT already in our local set. Lands in
`datasets/swebench_trajs/<short>_held_out/`.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time
import urllib.request

SOURCES = {
    # Each entry: (submission, traj_path_template) where template has {inst}
    "livesweagent_opus45": ("20251215_livesweagent_claude-opus-4-5",
                            "trajs/{inst}/{inst}.traj.json"),
    "livesweagent_gemini3": ("20251120_livesweagent_gemini-3-pro-preview",
                             "trajs/{inst}/{inst}.traj.json"),
    "sonar_opus45": ("20251205_sonar-foundation-agent_claude-opus-4-5",
                     "trajs/{inst}.json"),
    "sonar_sonnet45": ("20251103_sonar-foundation-agent_claude-sonnet-4-5",
                       "trajs/{inst}.json"),
}

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com/verified"
GITHUB_BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"


def fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=100,
                    help="how many held-out resolved trajectories per source")
    ap.add_argument("--pace-seconds", type=float, default=0.3)
    args = ap.parse_args()

    grand_total = 0
    for short, (submission, traj_template) in SOURCES.items():
        local_failed = {p.stem for p in (TRAJ_ROOT / short).glob("*.json")} \
            if (TRAJ_ROOT / short).exists() else set()
        local_resolved = {p.stem for p in (TRAJ_ROOT / f"{short}_resolved").glob("*.json")} \
            if (TRAJ_ROOT / f"{short}_resolved").exists() else set()
        local_all = local_failed | local_resolved

        results_url = f"{GITHUB_BASE}/{submission}/results/results.json"
        try:
            results = json.loads(fetch(results_url).decode())
        except Exception as e:
            print(f"{short}: results.json fetch failed: {e}", file=sys.stderr)
            continue
        held_out = sorted(set(results.get("resolved", [])) - local_all)
        targets = held_out[:args.per_source]

        out_dir = TRAJ_ROOT / f"{short}_held_out"
        log_dir = TRAJ_ROOT / f"{short}_held_out_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {short}: {len(targets)} trajectories to fetch ===", flush=True)
        n_ok, n_fail = 0, 0
        for i, inst in enumerate(targets, 1):
            traj_path = out_dir / f"{inst}.json"
            report_path = log_dir / f"{inst}__report.json"
            if traj_path.exists() and report_path.exists():
                n_ok += 1
                continue
            try:
                if not traj_path.exists():
                    traj_url = f"{S3_BASE}/{submission}/" + traj_template.format(inst=inst)
                    traj = fetch(traj_url)
                    traj_path.write_bytes(traj)
                if not report_path.exists():
                    report = fetch(f"{S3_BASE}/{submission}/logs/{inst}/report.json")
                    report_path.write_bytes(report)
                n_ok += 1
            except Exception as e:
                print(f"  FAIL {inst}: {e}", file=sys.stderr)
                n_fail += 1
            if i % 20 == 0:
                print(f"  ... {i}/{len(targets)} (ok={n_ok}, fail={n_fail})", flush=True)
            time.sleep(args.pace_seconds)
        print(f"  done: ok={n_ok} fail={n_fail}", flush=True)
        grand_total += n_ok

    print(f"\nGRAND TOTAL: {grand_total} held-out trajectories downloaded")


if __name__ == "__main__":
    main()
