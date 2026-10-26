#!/usr/bin/env bash
# E1'：干净执行器（固定提示：直接输出动作）下的故障样本生成，四个条件并行
cd /home/claude/chime
mkdir -p mas_runs/e1c/logs
for c in clean plan exec mem; do
  ./gm.sh /home/claude/chime/run_faults.py --conds $c --act-hint --budget-cny 0.6 --out /home/claude/chime/mas_runs/e1c/$c >> mas_runs/e1c/logs/$c.log 2>&1 &
done
wait
echo E1C_GEN_ALL_DONE
