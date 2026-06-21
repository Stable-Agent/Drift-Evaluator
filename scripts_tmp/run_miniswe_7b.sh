#!/usr/bin/env bash
# Run SWE-agent-LM-7B in its NATIVE mini-swe-agent harness (the format it was
# fine-tuned for) on the same frontier instances as the qwen pilot.
# Uses prebuilt x86 swebench images (emulated on Mac). Run AFTER the qwen pilot
# to avoid resource contention. Multi-seed via --shuffle + repeated runs, or use
# SWE-bench eval on the predictions afterward.
set -euo pipefail
cd /Users/tbrady/code/Stable-Agent
OUT=Drift-Evaluator/datasets/miniswe_7b
INSTANCES="${1:-django__django-10914|astropy__astropy-14309|django__django-10880}"
.venv/bin/python -m minisweagent.run.benchmarks.swebench \
  --subset verified --split test \
  --filter "($INSTANCES)" \
  -m "ollama_chat/swe-agent-lm-7b" \
  -w 1 -o "$OUT"
echo "predictions in $OUT ; eval with swebench harness or native grading"
