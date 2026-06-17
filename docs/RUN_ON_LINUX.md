# Running the multi-seed pilot on a Linux host

The harness needs Linux x86_64. On an arm64 Mac, swebench images run under
x86 emulation and OOM-kill at Docker's default memory (see
`reference-arm64-swebench-docker`). On native Linux x86, swebench pulls
**prebuilt** images from Docker Hub — no building, no emulation, no OOM — so
the whole pilot Just Runs.

## What to provision

- Linux x86_64 box (bare metal or cloud VM).
- Docker.
- **Disk:** prebuilt swebench images are ~1–2 GB each; budget ~100–150 GB for
  a 120-instance pilot (`--cache_level env` reuses env layers across seeds).
- **RAM:** ≥16 GB.
- **GPU (recommended):** qwen2.5-coder:14b is the agent model; it runs on CPU
  but a single 16–24 GB GPU makes the ~400 agent rollouts tractable.
- Network: pulls images from Docker Hub + the model from ollama.

## Files to copy to the host

Only three, plus the repo skeleton for output paths:
- `Drift-Evaluator/scripts_tmp/gen_multiseed.py`  (self-contained)
- `Drift-Evaluator/scripts_tmp/setup_host.sh`
- `Drift-Evaluator/docs/spec_multiseed_detector_v1.md` (reference)

Outputs land under `Drift-Evaluator/datasets/multiseed_v1/` (runs + per-step
checkpoint sidecars) and `.../reports/frontier_v1.json`.

## Steps

```bash
# from repo root on the Linux host
bash Drift-Evaluator/scripts_tmp/setup_host.sh        # installs deps, pulls model, warms one image
. .venv-linux/bin/activate

# 1) generate (resumable): K=4 seeds x 120 easy instances, agent in-container
python Drift-Evaluator/scripts_tmp/gen_multiseed.py --n 120 --seeds 4 --max-steps 15

# 2) evaluate all seeds with the official swebench harness + frontier report
python Drift-Evaluator/scripts_tmp/gen_multiseed.py --eval --max-workers 8

# 3) reprint frontier analysis anytime
python Drift-Evaluator/scripts_tmp/gen_multiseed.py --frontier-only
```

`MULTISEED_NAMESPACE` controls image source: default `swebench` (pull
prebuilt). Set `MULTISEED_NAMESPACE=""` only to force slow local builds.

## The decision gate (spec §7)

The pilot exists to answer one question: **does qwen2.5-coder:14b land enough
problems in the [0.25, 0.75] pass-rate band to give within-cluster signal?**
`--frontier-only` prints the pass-rate histogram and the in-band count.

- **≥30 in band** → proceed: bump to 8 seeds on the in-band set, freeze
  `prereg_multiseed_detector_v1.md`, run the three within-cluster analyses
  (divergence-from-success-manifold, per-step counterfactual value, stall
  label) per the spec.
- **<30 in band, mostly all-fail** → the model is too weak; swap to a stronger
  agent model (the harness is model-agnostic — change `MODEL`/backend) and
  re-pilot.
- **<30, mostly all-pass** → problems too easy; widen the difficulty filter in
  `load_pool` upward.

Do not start the full 8-seed run or write conclusions until the gate is met
and the prereg is frozen.
```
