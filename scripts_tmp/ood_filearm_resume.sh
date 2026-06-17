#!/bin/zsh
# Resume wrapper for ood_filearm_v1.py across usage-cap windows.
# Probes the claude CLI (haiku, ~free) every 15 min; when live, reruns the
# experiment (resumable: gen-failed rows are retried, verdicts are kept).
# Stops when no transient gen failures remain or after 10 rounds.
cd /Users/tbrady/code/Stable-Agent || exit 1
LOG=Drift-Evaluator/reports/ood_filearm_v1_run.log
for round in {1..10}; do
  until echo ok | claude -p --model claude-haiku-4-5-20251001 >/dev/null 2>&1; do
    echo "$(date '+%F %T') round $round: CLI capped; sleeping 15m" >> "$LOG"
    sleep 900
  done
  echo "$(date '+%F %T') round $round: CLI live; running" >> "$LOG"
  .venv/bin/python Drift-Evaluator/scripts_tmp/ood_filearm_v1.py >> "$LOG" 2>&1
  left=$(.venv/bin/python -c "
import json
rows={}
for l in open('Drift-Evaluator/reports/ood_filearm_v1.jsonl'):
    if l.strip():
        r=json.loads(l); rows[(r['iid'],r['arm'])]=r
print(sum(1 for r in rows.values()
          if not (r.get('gen_ok') and r.get('blocks') and r.get('tail') != 'gen failed')))")
  echo "$(date '+%F %T') round $round done; transient gen-failures left: $left" >> "$LOG"
  [ "$left" -eq 0 ] && break
done
echo "$(date '+%F %T') resume wrapper finished" >> "$LOG"
