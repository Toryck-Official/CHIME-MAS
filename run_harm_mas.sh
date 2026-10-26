#!/usr/bin/env bash
# 危害实验（多智能体朴素 CHIME，DyLAN 2×2 + critic）：orig / oracle，任务顺序 s0，提议者与定稿者加固定提示
cd /home/claude/chime
mkdir -p mas_runs/harm/logs
for c in orig oracle; do
  ./gm.sh /home/claude/chime/run_harm.py --mode naive --cond $c --order-seed 0 --act-hint --budget-cny 4.5 >> mas_runs/harm/logs/naive_${c}_s0.log 2>&1 &
done
wait
echo HARM_MAS_DONE
