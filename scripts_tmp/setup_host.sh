#!/usr/bin/env bash
# Set up a Linux x86_64 GPU host to run the multi-seed pilot FAST with the
# SWE-tuned 32B agent. Native x86 => swebench uses PREBUILT Docker Hub images
# (no builds, no arm attrition — every instance works). Speed comes from:
#   - vLLM serving (continuous batching of concurrent agent calls)
#   - concurrent rollouts (--workers / MULTISEED_WORKERS)
#   - full-VRAM quant (model resident on GPU, zero swap — the local killer)
#
# Assumes: Ubuntu x86_64, sudo, NVIDIA GPU(s). Run from the repo root.
#   VRAM guide for SWE-bench/SWE-agent-LM-32B:
#     fp16  ~64GB -> 1xA100-80G  or 2x 48G (TP=2)
#     awq/fp8 ~20GB -> 1x 24-48G (set MODEL to an AWQ repo, e.g. a community
#       SWE-agent-LM-32B-AWQ); cheaper, slightly lower quality.
set -euo pipefail

MODEL="${MULTISEED_MODEL:-SWE-bench/SWE-agent-LM-32B}"
TP="${VLLM_TP:-1}"                 # tensor-parallel = number of GPUs
PORT="${VLLM_PORT:-8000}"
WORKERS="${MULTISEED_WORKERS:-8}"  # concurrent rollouts (vLLM batches them)

echo "== docker =="
command -v docker >/dev/null || { curl -fsSL https://get.docker.com | sh; sudo usermod -aG docker "$USER" || true; echo "re-login for docker, then re-run"; }
docker run --rm hello-world >/dev/null && echo "docker ok"

echo "== python venv + deps =="
python3 -m venv .venv-linux
. .venv-linux/bin/activate
pip install -q --upgrade pip
pip install -q "swebench>=4.1" datasets requests docker vllm

echo "== serve $MODEL via vLLM on :$PORT (TP=$TP) =="
pkill -f "vllm.entrypoints" 2>/dev/null || true
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  >/tmp/vllm.log 2>&1 &
echo "  waiting for vLLM to load (watch /tmp/vllm.log)..."
for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "  vLLM up"; break; }
  sleep 10
done

echo "== warm one prebuilt image =="
MULTISEED_NAMESPACE=swebench python - <<'PY'
import os; os.environ["MULTISEED_NAMESPACE"]="swebench"
import sys; sys.path.insert(0,"Drift-Evaluator/scripts_tmp")
from datasets import load_dataset
import gen_multiseed as g
ds=load_dataset("princeton-nlp/SWE-bench_Verified",split="test")
inst=[r for r in ds if r["repo"]=="django/django" and str(r.get("difficulty","")).startswith("<15")][0]
g.ensure_instance_image(inst); print("PREBUILT PULL OK —", g.docker_image_for(inst))
PY

cat <<EOF

== ready: run the FAST 32B pilot ==
  . .venv-linux/bin/activate
  export MULTISEED_BASE_URL=http://localhost:$PORT
  export MULTISEED_MODEL="$MODEL"
  export MULTISEED_NAMESPACE=swebench      # prebuilt x86 images, all instances
  export MULTISEED_ARCH=x86_64

  # generate: 120 instances x 4 seeds, $WORKERS concurrent rollouts (vLLM batches)
  python Drift-Evaluator/scripts_tmp/gen_multiseed.py --n 120 --seeds 4 --workers $WORKERS
  # evaluate (parallel) + frontier
  python Drift-Evaluator/scripts_tmp/gen_multiseed.py --eval --max-workers 16

Speed knobs: --workers (more concurrent rollouts), VLLM_TP (more GPUs),
--max-workers (eval parallelism). vLLM batching is the main win.
EOF
