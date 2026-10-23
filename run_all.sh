#!/usr/bin/env bash
# S6：airline 全部 50 个任务，4 个变体 × 3 种任务顺序。同一顺序下 4 个变体并行，顺序之间串行。
cd /home/claude/chime
PY=/home/claude/refs/tau2/.venv/bin/python
mkdir -p runs logs
for seed in 0 1 2; do
  for v in no_plan plan_only chime chime_nogate; do
    if [ -f runs/airline_${v}_s${seed}/summary.json ]; then continue; fi
    $PY run_stream.py --domain airline --variant $v --order-seed $seed --eval-workers 3 --out runs \
      > logs/airline_${v}_s${seed}.log 2>&1 &
  done
  wait
done
echo ALL_DONE
