#!/usr/bin/env python
"""Pull held-out FAILED SWE-bench trajectories for retraining.

failed = (all instance IDs in trajs/) - resolved - no_generation.
Lands in `datasets/swebench_trajs/<short>_held_out_failed/`.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

SOURCES = {
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


def list_all_instances(submission: str, traj_template: str) -> list[str]:
    """List every instance ID in this submission's trajs/."""
    if "{inst}/" in traj_template:
        # Pattern: trajs/<inst>/<inst>.traj.json — list prefixes
        url = f"{S3_BASE.replace('/verified', '')}/?list-type=2&prefix=verified/{submission}/trajs/&delimiter=/&max-keys=1000"
        body = fetch(url, timeout=30).decode()
        prefixes = re.findall(r"<Prefix>verified/[^/]+/trajs/([^/]+)/</Prefix>", body)
        return sorted(prefixes)
    else:
        # Pattern: trajs/<inst>.json — list keys
        url = f"{S3_BASE.replace('/verified', '')}/?list-type=2&prefix=verified/{submission}/trajs/&max-keys=1000"
        body = fetch(url, timeout=30).decode()
        keys = re.findall(r"<Key>verified/[^/]+/trajs/([^<]+)</Key>", body)
        return sorted({k.replace(".json", "") for k in keys if k.endswith(".json")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--pace-seconds", type=float, default=0.3)
    args = ap.parse_args()

    grand_total = 0
    for short, (submission, traj_template) in SOURCES.items():
        local_failed = {p.stem for p in (TRAJ_ROOT / short).glob("*.json")} \
            if (TRAJ_ROOT / short).exists() else set()
        local_resolved = {p.stem for p in (TRAJ_ROOT / f"{short}_resolved").glob("*.json")} \
            if (TRAJ_ROOT / f"{short}_resolved").exists() else set()
        local_held_out = {p.stem for p in (TRAJ_ROOT / f"{short}_held_out").glob("*.json")} \
            if (TRAJ_ROOT / f"{short}_held_out").exists() else set()
        local_held_out_failed = {p.stem for p in (TRAJ_ROOT / f"{short}_held_out_failed").glob("*.json")} \
            if (TRAJ_ROOT / f"{short}_held_out_failed").exists() else set()
        local_all = local_failed | local_resolved | local_held_out | local_held_out_failed

        try:
            results = json.loads(fetch(f"{GITHUB_BASE}/{submission}/results/results.json").decode())
            all_insts = list_all_instances(submission, traj_template)
        except Exception as e:
            print(f"{short}: setup fetch failed: {e}", file=sys.stderr)
            continue
        resolved = set(results.get("resolved", []))
        no_gen = set(results.get("no_generation", []))
        failed_pool = set(all_insts) - resolved - no_gen
        held_out_failed = sorted(failed_pool - local_all)
        targets = held_out_failed[:args.per_source]

        out_dir = TRAJ_ROOT / f"{short}_held_out_failed"
        log_dir = TRAJ_ROOT / f"{short}_held_out_failed_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {short}: {len(failed_pool)} failed available, "
              f"fetching {len(targets)} held-out ===", flush=True)
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
                    traj_path.write_bytes(fetch(traj_url))
                if not report_path.exists():
                    report_path.write_bytes(fetch(
                        f"{S3_BASE}/{submission}/logs/{inst}/report.json"))
                n_ok += 1
            except Exception as e:
                print(f"  FAIL {inst}: {e}", file=sys.stderr)
                n_fail += 1
            if i % 20 == 0:
                print(f"  ... {i}/{len(targets)} (ok={n_ok}, fail={n_fail})", flush=True)
            time.sleep(args.pace_seconds)
        print(f"  done: ok={n_ok} fail={n_fail}", flush=True)
        grand_total += n_ok

    print(f"\nGRAND TOTAL: {grand_total} failed trajectories")


if __name__ == "__main__":
    main()
