#!/usr/bin/env python
"""Thread #2, Phase 1 (framing): localize against the SWE-bench gold patch.

Establishes the gap a live truth-site predictor must close:
  - AGENT sites (where the failing agent actually edited) vs gold patch = the floor
  - ANNOTATOR truth_sites (hindsight) vs gold patch = the ceiling / metric sanity

Localization ground truth = files + new-side line ranges parsed from the gold
unified diff. A site "file-hits" if its file matches a gold file; "line-hits"
if its approx_line is within +/-TOL of a gold hunk for that file.
"""
from __future__ import annotations
import json, re, pathlib
from datasets import load_dataset

TOL = 15
GOLD = pathlib.Path("Drift-Evaluator/gold/coding_v0_opus_labeled.jsonl")

def parse_patch(patch: str) -> dict[str, list[tuple[int, int]]]:
    """gold diff -> {new_file_path: [(start,end) new-side line ranges]}."""
    files: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m:
            cur = m.group(1); files.setdefault(cur, [])
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m and cur is not None:
            start = int(m.group(1)); length = int(m.group(2) or 1)
            files[cur].append((start, start + max(length, 1)))
    return files

def norm(path: str) -> str:
    # strip leading testbed/repo prefixes for comparison
    p = path.strip()
    for pre in ("/testbed/", "testbed/", "a/", "b/"):
        if p.startswith(pre): p = p[len(pre):]
    return p

def file_hit(pred_file, gold_files):
    pf = norm(pred_file)
    gf = {norm(g) for g in gold_files}
    if pf in gf: return True
    # basename fallback (annotator sometimes abbreviates)
    pb = pf.rsplit("/", 1)[-1]
    return any(pb == g.rsplit("/", 1)[-1] for g in gf)

def line_hit(pred_file, pred_line, gold):
    if pred_line is None: return False
    pf = norm(pred_file); pb = pf.rsplit("/", 1)[-1]
    for gfile, ranges in gold.items():
        if norm(gfile) == pf or norm(gfile).rsplit("/", 1)[-1] == pb:
            for (s, e) in ranges:
                if s - TOL <= pred_line <= e + TOL:
                    return True
    return False

def eval_sites(sites, gold):
    """returns (any_file_hit, any_line_hit) over a list of sites."""
    fh = any(file_hit(s.get("file", ""), gold.keys()) for s in sites)
    lh = any(line_hit(s.get("file", ""), s.get("approx_line"), gold) for s in sites)
    return fh, lh

def main():
    recs = [json.loads(l) for l in open(GOLD) if l.strip()]
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    patch_by_id = {r["instance_id"]: r["patch"] for r in ds}

    n = 0
    agent_fh = agent_lh = ts_fh = ts_lh = 0
    gold_files_count = []
    for r in recs:
        iid = r["trajectory_id"]
        if iid not in patch_by_id:
            continue
        gold = parse_patch(patch_by_id[iid])
        if not gold:
            continue
        n += 1
        gold_files_count.append(len(gold))
        sc = r.get("scope", {})
        afh, alh = eval_sites(sc.get("agent_sites", []), gold)
        tfh, tlh = eval_sites(sc.get("truth_sites", []), gold)
        agent_fh += afh; agent_lh += alh
        ts_fh += tfh; ts_lh += tlh

    print(f"evaluated {n} failed trajectories against gold patches (TOL=+/-{TOL} lines)")
    print(f"gold patch files/instance: mean={sum(gold_files_count)/n:.2f} "
          f"single-file={sum(1 for c in gold_files_count if c==1)}/{n}\n")
    print(f"{'sites':>22}  {'file-hit':>9}  {'line-hit':>9}")
    print(f"{'AGENT (failing edit)':>22}  {agent_fh/n:>8.1%}  {agent_lh/n:>8.1%}")
    print(f"{'ANNOTATOR truth_sites':>22}  {ts_fh/n:>8.1%}  {ts_lh/n:>8.1%}")
    print(f"\nfloor = agent (these all FAILED); ceiling = annotator hindsight.")
    print(f"a live predictor must climb from ~agent toward ~annotator without")
    print(f"seeing the gold patch. that headroom is the #2 research target.")

if __name__ == "__main__":
    main()
