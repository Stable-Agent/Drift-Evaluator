#!/usr/bin/env bash
# Set up a Linux x86_64 host to run the multi-seed pilot (gen_multiseed.py).
# Native x86 => swebench uses PREBUILT Docker Hub images (no local builds, no
# emulation, no OOM — the wall we hit on the arm64 Mac). See
# docs/spec_multiseed_detector_v1.md and reference-arm64-swebench-docker.
#
# Assumes: Ubuntu/Debian-like, sudo, x86_64, ideally an NVIDIA GPU for ollama
# (qwen2.5-coder:14b runs on CPU but slowly). Run from the repo root.
set -euo pipefail

echo "== sanity =="
arch="$(uname -m)"; [ "$arch" = "x86_64" ] || { echo "WARNING: arch=$arch, not x86_64 — prebuilt images may not match"; }

echo "== docker =="
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
  echo "NOTE: log out/in (or 'newgrp docker') so docker works without sudo, then re-run."
fi
docker run --rm hello-world >/dev/null && echo "docker ok"

echo "== python venv + deps =="
python3 -m venv .venv-linux
. .venv-linux/bin/activate
pip install -q --upgrade pip
pip install -q "swebench>=4.1" datasets requests docker

echo "== ollama + model =="
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
# start server if not already up
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  (ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 5
fi
ollama pull qwen2.5-coder:14b

echo "== warm one prebuilt image (validates the pull path) =="
. .venv-linux/bin/activate
python - <<'PY'
import os; os.environ["MULTISEED_NAMESPACE"]="swebench"
import sys; sys.path.insert(0,"Drift-Evaluator/scripts_tmp")
from datasets import load_dataset
import gen_multiseed as g
ds=load_dataset("princeton-nlp/SWE-bench_Verified",split="test")
inst=[r for r in ds if r["repo"]=="django/django" and str(r.get("difficulty","")).startswith("<15")][0]
print("pulling prebuilt image for",inst["instance_id"],"...",flush=True)
g.ensure_instance_image(inst)
print("PREBUILT PULL OK —",g.docker_image_for(inst))
PY

cat <<'EOF'

== ready ==
Activate the venv and run the pilot (prebuilt images are the default):

  . .venv-linux/bin/activate
  # 1) generate: K=4 seeds x 120 easy instances (resumable)
  python Drift-Evaluator/scripts_tmp/gen_multiseed.py --n 120 --seeds 4 --max-steps 15
  # 2) evaluate every seed via the official harness + frontier report
  python Drift-Evaluator/scripts_tmp/gen_multiseed.py --eval --max-workers 8
  # 3) re-print frontier analysis anytime
  python Drift-Evaluator/scripts_tmp/gen_multiseed.py --frontier-only

Gate (spec §7): need >=30 instances in the [0.25,0.75] band to proceed to the
full 8-seed run + freeze prereg_multiseed_detector_v1.md.
EOF
