#!/usr/bin/env bash
# Status monitor for the formal seed42 run (spec §19/§20/§25 checks).
ROOT=experiments/runs/compose_ucit_formal_seed42
echo "== run status $(date '+%F %T') =="
echo "-- stages --"
for t in 0 1 2 3 4 5; do
  done_count=$(ls "$ROOT/task$t/stages/"*.done 2>/dev/null | wc -l)
  markers=$(ls "$ROOT/task$t/stages/" 2>/dev/null | tr '\n' ' ')
  echo "task$t: $done_count/13 stages  [$markers]"
done
echo "-- tmux alive --"
tmux has-session -t compose_formal42 2>/dev/null && echo "session RUNNING" || echo "session DEAD"
echo "-- gpus 4-7 --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '5,8p'
echo "-- NaN/Inf scan (word-boundary) --"
for f in $(find "$ROOT/task"* -name "*.log" 2>/dev/null | head -60); do
  if grep -qiE "\bnan\b|\binf\b|nan=|inf=" "$f" 2>/dev/null; then echo "NAN/INF in $f"; fi
done
echo "-- kappa 0.25 scan --"
for f in $(find "$ROOT/task"* -name "rms_summary.json" 2>/dev/null); do
  grep -q "kappa" "$f" && echo "rms present: $f"
done
echo "== end =="
