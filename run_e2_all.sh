#!/usr/bin/env bash
# E2：三个条件并行，gripper 20 题，任务顺序 s0
cd /home/claude/chime
mkdir -p mas_runs/e2/logs
./gm.sh /home/claude/chime/run_e2.py --mode single_chime --budget-cny 0.5 >> mas_runs/e2/logs/single_chime.log 2>&1 &
./gm.sh /home/claude/chime/run_e2.py --mode naive --budget-cny 1.8 >> mas_runs/e2/logs/naive.log 2>&1 &
./gm.sh /home/claude/chime/run_e2.py --mode central --budget-cny 1.8 >> mas_runs/e2/logs/central.log 2>&1 &
wait
echo E2_ALL_DONE
