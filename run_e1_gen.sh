#!/usr/bin/env bash
# E1 故障样本生成：四个条件并行，各自输出目录，可断点续跑
cd /home/claude/chime
mkdir -p mas_runs/e1/logs
for c in clean plan exec mem; do
  ./gm.sh /home/claude/chime/run_faults.py --conds $c --budget-cny 0.8 --out /home/claude/chime/mas_runs/e1/$c >> mas_runs/e1/logs/$c.log 2>&1 &
done
wait
echo E1_GEN_ALL_DONE
