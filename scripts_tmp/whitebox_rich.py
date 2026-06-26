"""Richer white-box signal: CONTINUATION ENTROPY. Instead of reading the model's
verdict-token uncertainty, measure how UNCERTAIN the open model is about what the
agent should do next, given the trajectory so far. A drifted/confused trajectory
leaves the model unsure how to continue -> high-entropy continuation. This is a
different (and in-flight-capable) white-box signal than answer_entropy, and one a
frontier API can't expose.

ollama gives logprobs on GENERATED tokens (not prompt tokens), so we measure the
entropy of the model's own generated next-action, not perplexity over the agent's
action (which would need echo/prompt logprobs ollama lacks).

Features per trajectory (1 generation call, logprobs+top_logprobs):
  cont_entropy_mean : mean per-token entropy of the generated next-action
  cont_entropy_max  : max per-token entropy (a confusion spike)
  cont_perplexity   : exp(mean -logprob) of the generation
Scored on the NON-EMPTY subset (the real task). Resumable.
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import requests
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
import detector_probe as dp

OLLAMA = "http://localhost:11434/v1/chat/completions"
CACHE = pathlib.Path("Drift-Evaluator/reports/whitebox_rich_cache.jsonl")

CONT_PROMPT = """You are an expert software engineer supervising an agent fixing a bug.
Here is the bug and the agent's actions so far:

{text}

What is the single best next shell command for the agent to make real progress
toward a correct fix? Output ONLY the command, nothing else."""


def continuation_features(text, model, timeout=120):
    try:
        r = requests.post(OLLAMA, json={
            "model": model, "temperature": 0.0, "max_tokens": 40,
            "logprobs": True, "top_logprobs": 5,
            "messages": [{"role": "user", "content": CONT_PROMPT.format(text=text)}]},
            timeout=timeout)
        if r.status_code != 200:
            return {}
        ch = r.json()["choices"][0]
        toks = (ch.get("logprobs") or {}).get("content") or []
        if not toks:
            return {}
        ents, lps = [], []
        for t in toks:
            lps.append(t["logprob"])
            tops = t.get("top_logprobs") or []
            ps = [math.exp(x["logprob"]) for x in tops]
            if ps:
                ents.append(-sum(p * math.log(p + 1e-12) for p in ps))
        if not ents:
            return {}
        mean_lp = sum(lps) / len(lps)
        return {"cont_entropy_mean": sum(ents) / len(ents),
                "cont_entropy_max": max(ents),
                "cont_perplexity": math.exp(-mean_lp)}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    rows = dp.load_corpus()
    nonempty = [r for r in rows if not r["struct"]["empty_patch"]]
    print(f"scoring continuation entropy on {len(nonempty)} non-empty trajectories", flush=True)

    cache = {}
    if CACHE.exists():
        for l in open(CACHE):
            if l.strip():
                c = json.loads(l); cache[(c["iid"], c["seed"])] = c

    if not args.analyze_only:
        cf = CACHE.open("a")
        for i, r in enumerate(nonempty):
            key = (r["iid"], r["seed"])
            if cache.get(key, {}).get("cont_entropy_mean") is not None:
                continue
            feats = continuation_features(r["text"], args.model)
            feats.update({"iid": r["iid"], "seed": r["seed"]})
            cache[key] = feats
            cf.write(json.dumps(feats) + "\n"); cf.flush()
            if (i + 1) % 15 == 0:
                print(f"  {i+1}/{len(nonempty)}", flush=True)
        cf.close()

    for r in nonempty:
        r.update({k: cache.get((r["iid"], r["seed"]), {}).get(k)
                  for k in ("cont_entropy_mean", "cont_entropy_max", "cont_perplexity")})

    print(f"\n=== richer white-box ({args.model}) on NON-EMPTY (n={len(nonempty)}) vs FAILURE ===")
    print(f"{'feature':>20} {'AUC':>7} {'within-cluster':>15}")
    for feat in ("cont_entropy_mean", "cont_entropy_max", "cont_perplexity"):
        a, n = dp.auc([r.get(feat) for r in nonempty], [r["failed"] for r in nonempty])
        wc, pairs, inst = dp.within_cluster_concordance(nonempty, feat)
        f = lambda x: "None" if x is None else f"{x:.3f}"
        print(f"{feat:>20} {f(a):>7} {f(wc):>10} ({pairs}p, {inst}i)")
    print("\nbaseline to beat: answer_entropy non-empty AUC 0.814, wc 0.775")


if __name__ == "__main__":
    main()
