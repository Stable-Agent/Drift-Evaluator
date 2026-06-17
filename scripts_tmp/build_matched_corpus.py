#!/usr/bin/env python
"""Build a difficulty-controlled MATCHED corpus of SWE-bench trajectories.

A matched instance is one SWE-bench Verified problem that was *resolved by at
least one agent* AND *failed by at least one agent*. For each, we pull one
resolved trajectory and one failed trajectory. Same problem => difficulty,
repo, and test gate are held constant, so any signal separating the two runs
is behavioral (the bias documented in project_matched_auc_bias.md is removed
by construction).

Avoids openhands (sanitized trajectories — see reference_swebench_trajs.md).

Usage:
  # Cheap: only fetch results.json + listings, report achievable matched size
  python build_matched_corpus.py --plan

  # Download up to N matched pairs (resumable; skips existing)
  python build_matched_corpus.py --limit 150
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time, urllib.request
from collections import defaultdict

# (short, submission, traj_template, agent_tier)
SOURCES = [
    ("livesweagent_opus45",  "20251215_livesweagent_claude-opus-4-5",      "trajs/{inst}/{inst}.traj.json", "opus"),
    ("livesweagent_gemini3", "20251120_livesweagent_gemini-3-pro-preview", "trajs/{inst}/{inst}.traj.json", "gemini"),
    ("sonar_opus45",         "20251205_sonar-foundation-agent_claude-opus-4-5",   "trajs/{inst}.json", "opus"),
    ("sonar_sonnet45",       "20251103_sonar-foundation-agent_claude-sonnet-4-5", "trajs/{inst}.json", "sonnet"),
]

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
OUT_ROOT = TRAJ_ROOT / "matched_corpus"
MANIFEST = OUT_ROOT / "manifest.jsonl"
S3 = "https://swe-bench-submissions.s3.amazonaws.com"
GH = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"


def fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def list_instances(submission: str, template: str) -> set[str]:
    """Every instance id with a trajectory under this submission."""
    if "{inst}/" in template:  # trajs/<inst>/<inst>.traj.json — directory prefixes
        url = f"{S3}/?list-type=2&prefix=verified/{submission}/trajs/&delimiter=/&max-keys=1000"
        body = fetch(url).decode()
        return set(re.findall(r"<Prefix>verified/[^/]+/trajs/([^/]+)/</Prefix>", body))
    url = f"{S3}/?list-type=2&prefix=verified/{submission}/trajs/&max-keys=1000"
    body = fetch(url).decode()
    keys = re.findall(r"<Key>verified/[^/]+/trajs/([^<]+)</Key>", body)
    return {k[:-5] for k in keys if k.endswith(".json")}


def survey() -> dict:
    """inst -> {'resolved': [(short, tier, template, submission)...],
                'failed':   [...]}"""
    inst_map: dict[str, dict] = defaultdict(lambda: {"resolved": [], "failed": []})
    for short, submission, template, tier in SOURCES:
        try:
            results = json.loads(fetch(f"{GH}/{submission}/results/results.json").decode())
            attempted = list_instances(submission, template)
        except Exception as e:
            print(f"  {short}: survey fetch failed: {e}", file=sys.stderr)
            continue
        resolved = set(results.get("resolved", []))
        no_gen = set(results.get("no_generation", []))
        failed = attempted - resolved - no_gen
        meta = (short, tier, template, submission)
        for inst in attempted & resolved:
            inst_map[inst]["resolved"].append(meta)
        for inst in failed:
            inst_map[inst]["failed"].append(meta)
        print(f"  {short:22s} attempted={len(attempted):3d} "
              f"resolved={len(attempted & resolved):3d} failed={len(failed):3d}")
    return inst_map


def pick_pair(rec):
    """Choose (resolved_meta, failed_meta), preferring same agent tier to
    avoid 'resolved=stronger model' becoming a new confound."""
    res, fail = rec["resolved"], rec["failed"]
    for rm in res:
        for fm in fail:
            if rm[1] == fm[1]:            # same tier
                return rm, fm
    return res[0], fail[0]                 # fall back to cross-tier


def dl_traj(meta, inst, dest: pathlib.Path) -> bool:
    short, tier, template, submission = meta
    if dest.exists():
        return True
    url = f"{S3}/verified/{submission}/" + template.format(inst=inst)
    try:
        dest.write_bytes(fetch(url))
        return True
    except Exception as e:
        print(f"    FAIL {short}/{inst}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="only survey + report achievable matched size; no downloads")
    ap.add_argument("--limit", type=int, default=200, help="max matched pairs to pull")
    ap.add_argument("--pace-seconds", type=float, default=0.3)
    args = ap.parse_args()

    print("surveying agent results (4 usable sources; openhands excluded)...")
    inst_map = survey()
    matched = {i: r for i, r in inst_map.items() if r["resolved"] and r["failed"]}

    tier_pairs = defaultdict(int)
    same_tier = 0
    for r in matched.values():
        rm, fm = pick_pair(r)
        tier_pairs[(rm[1], fm[1])] += 1
        if rm[1] == fm[1]:
            same_tier += 1
    print(f"\n=== achievable matched set: {len(matched)} problems ===")
    print(f"    same-tier pairs (cleanest): {same_tier}")
    print("    resolved_tier -> failed_tier counts:")
    for (rt, ft), n in sorted(tier_pairs.items(), key=lambda kv: -kv[1]):
        print(f"      {rt:7s} -> {ft:7s}  {n}")

    if args.plan:
        print("\n--plan: no downloads. Re-run without --plan to pull.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        done = {json.loads(l)["instance"] for l in MANIFEST.open() if l.strip()}
    targets = [i for i in sorted(matched) if i not in done][:args.limit]
    print(f"\ndownloading {len(targets)} pairs (skip {len(done)} already done)...")

    n_ok = 0
    with MANIFEST.open("a") as mf:
        for i, inst in enumerate(targets, 1):
            rm, fm = pick_pair(matched[inst])
            d = OUT_ROOT / inst
            d.mkdir(exist_ok=True)
            rp = d / f"resolved__{rm[0]}.json"
            fp = d / f"failed__{fm[0]}.json"
            ok_r = dl_traj(rm, inst, rp)
            time.sleep(args.pace_seconds)
            ok_f = dl_traj(fm, inst, fp)
            time.sleep(args.pace_seconds)
            if ok_r and ok_f:
                mf.write(json.dumps({
                    "instance": inst,
                    "resolved": {"path": str(rp), "source": rm[0], "tier": rm[1]},
                    "failed":   {"path": str(fp), "source": fm[0], "tier": fm[1]},
                }) + "\n")
                mf.flush()
                n_ok += 1
            if i % 20 == 0:
                print(f"  ... {i}/{len(targets)} (ok={n_ok})", flush=True)
    print(f"\ndone: {n_ok} matched pairs in {OUT_ROOT}")
    print("manifest:", MANIFEST)
    print("Next: point matched_auc.py at the manifest to re-run at the larger n.")


if __name__ == "__main__":
    main()
