#!/usr/bin/env bash
# Auto-chain after the corpus-expansion generation finishes:
#   1. wait for gen_multiseed to exit (avoid Docker/GPU contention)
#   2. native-eval all rollouts (x86 prebuilt, matches generation arch) -> labels
#   3. extract white-box confidence features for the full corpus
#   4. print the white-box detection signal on the FULL labeled corpus
# Resumable at every stage. Logs to /tmp/auto_eval_chain.log.
set -uo pipefail
cd /Users/tbrady/code/Stable-Agent
LOG=/tmp/auto_eval_chain.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

say "waiting for generation (gen_multiseed) to finish..."
while pgrep -f gen_multiseed.py >/dev/null; do sleep 120; done
say "generation done. corpus size: $(wc -l < Drift-Evaluator/datasets/multiseed_v1/runs.jsonl) rollouts"

say "STAGE 2: native eval (x86 prebuilt) — labels the new rollouts (resumable)"
MULTISEED_ARCH=x86_64 MULTISEED_NAMESPACE=swebench \
  .venv/bin/python Drift-Evaluator/scripts_tmp/gen_multiseed.py --eval --max-workers 2 \
  >>"$LOG" 2>&1
say "eval done. labels: $(wc -l < Drift-Evaluator/datasets/multiseed_v1/reports/eval_native.jsonl)"

say "STAGE 3: white-box feature extraction over full corpus (resumable)"
.venv/bin/python Drift-Evaluator/scripts_tmp/whitebox_probe.py --model qwen2.5-coder:14b \
  >>"$LOG" 2>&1

say "STAGE 4: white-box detection signal on FULL labeled corpus"
.venv/bin/python Drift-Evaluator/scripts_tmp/whitebox_probe.py --model qwen2.5-coder:14b --analyze-only 2>&1 \
  | grep -viE 'warning|httpx|HTTP Request' | tee -a "$LOG"
say "CHAIN COMPLETE — review white-box AUCs above (esp. answer_entropy non-empty vs the n=22 baseline 0.702)"
