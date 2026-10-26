#!/usr/bin/env bash
# 危害实验（单智能体）：orig / oracle × 任务顺序 s0 / s1，执行器加固定提示
cd /home/claude/chime
mkdir -p mas_runs/harm/logs
for s in 0 1; do for c in orig oracle; do
  ./gm.sh /home/claude/chime/run_harm.py --mode single_chime --cond $c --order-seed $s --act-hint --budget-cny 1.0 >> mas_runs/harm/logs/single_${c}_s$s.log 2>&1 &
done; done
wait
echo HARM_SINGLE_DONE
