#!/usr/bin/env python
"""White-box signal check: does the OPEN model's token-level CONFIDENCE separate
failing from passing trajectories, even though its flatlined ANSWER doesn't?

The detector probe showed base qwen-14B's direct P(fail) is chance (it says
~0.0 to everything). But white-box self-consistency (0.649) beat that. This
extracts richer white-box features the open model can give but a frontier API
cannot, and checks whether ANY of them carry signal — a go/no-go before
investing in corpus expansion + training a detector.

Features per trajectory (open model judging it, logprobs=True):
  - p_fail        : the P(fail) value (baseline, flatlines)
  - answer_entropy: entropy over top_logprobs of the answer's first token
                    (the model's UNCERTAINTY about its own verdict)
  - mean_logprob  : mean token logprob of the generated answer (fluency/confidence)
  - margin        : prob(top token) - prob(2nd token) on the verdict token

Reuses the detector_probe corpus + execution labels. Cheap, resumable.
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import requests
sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
import detector_probe as dp

OLLAMA = "http://localhost:11434/v1/chat/completions"
CACHE = pathlib.Path("Drift-Evaluator/reports/whitebox_probe_cache.jsonl")


def judge_with_logprobs(text, model, temp=0.0, timeout=120):
    """Return (p_fail, answer_entropy, mean_logprob, margin) or Nones."""
    try:
        r = requests.post(OLLAMA, json={
            "model": model, "temperature": temp, "max_tokens": 10,
            "logprobs": True, "top_logprobs": 5,
            "messages": [{"role": "user", "content": dp.JUDGE_PROMPT.format(text=text)}]},
            timeout=timeout)
        if r.status_code != 200:
            return {}
        ch = r.json()["choices"][0]
        p_fail = dp.parse_prob(ch["message"]["content"])
        content_lp = (ch.get("logprobs") or {}).get("content") or []
        if not content_lp:
            return {"p_fail": p_fail}
        # find the first token that looks numeric (the verdict digit)
        verdict_tok = None
        for t in content_lp:
            if any(c.isdigit() for c in t.get("token", "")):
                verdict_tok = t; break
        verdict_tok = verdict_tok or content_lp[0]
        tops = verdict_tok.get("top_logprobs") or []
        probs = [math.exp(t["logprob"]) for t in tops]
        ent = -sum(p * math.log(p + 1e-12) for p in probs) if probs else None
        margin = (probs[0] - probs[1]) if len(probs) >= 2 else None
        mean_lp = sum(t["logprob"] for t in content_lp) / len(content_lp)
        return {"p_fail": p_fail, "answer_entropy": ent,
                "mean_logprob": mean_lp, "margin": margin}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    rows = dp.load_corpus()
    print(f"corpus: {len(rows)} trajectories", flush=True)
    cache = {}
    if CACHE.exists():
        for l in open(CACHE):
            if l.strip():
                c = json.loads(l); cache[(c["iid"], c["seed"])] = c

    if not args.analyze_only:
        cf = CACHE.open("a")
        for i, r in enumerate(rows):
            key = (r["iid"], r["seed"])
            if cache.get(key, {}).get("answer_entropy") is not None:
                continue
            feats = judge_with_logprobs(r["text"], args.model)
            feats.update({"iid": r["iid"], "seed": r["seed"]})
            cache[key] = feats
            cf.write(json.dumps(feats) + "\n"); cf.flush()
            if (i + 1) % 15 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
        cf.close()

    for r in rows:
        r.update({k: cache.get((r["iid"], r["seed"]), {}).get(k)
                  for k in ("p_fail", "answer_entropy", "mean_logprob", "margin")})

    nonempty = [r for r in rows if not r["struct"]["empty_patch"]]
    print(f"\n=== white-box features ({args.model}) vs FAILURE ===")
    print(f"{'feature':>16} {'AUC(all)':>9} {'AUC(non-empty)':>15} {'within-cluster':>15}")
    for feat in ("p_fail", "answer_entropy", "mean_logprob", "margin"):
        a_all, _ = dp.auc([r.get(feat) for r in rows], [r["failed"] for r in rows])
        a_ne, n_ne = dp.auc([r.get(feat) for r in nonempty], [r["failed"] for r in nonempty])
        wc, pairs, insts = dp.within_cluster_concordance(rows, feat)
        # AUC < 0.5 means the feature is inversely related; report |dist from 0.5|
        def fmt(a): return "None" if a is None else f"{a:.3f}"
        print(f"{feat:>16} {fmt(a_all):>9} {fmt(a_ne):>15} (n={n_ne}) {fmt(wc):>10} ({pairs}p)")
    print("\nNote: AUC far from 0.5 in EITHER direction = signal. <0.5 = inverse "
          "(e.g. high uncertainty -> failure would give answer_entropy AUC>0.5).")


if __name__ == "__main__":
    main()
