#!/usr/bin/env bash
# E2b：barman 20 题，植入与模型误解一致、下游可补救的误导记忆
cd /home/claude/chime
mkdir -p mas_runs/e2/logs
./gm.sh /home/claude/chime/run_e2.py --game barman --mode single_chime --budget-cny 0.6 >> mas_runs/e2/logs/barman_single_chime.log 2>&1 &
./gm.sh /home/claude/chime/run_e2.py --game barman --mode naive --budget-cny 1.8 >> mas_runs/e2/logs/barman_naive.log 2>&1 &
./gm.sh /home/claude/chime/run_e2.py --game barman --mode central --budget-cny 1.8 >> mas_runs/e2/logs/barman_central.log 2>&1 &
wait
echo E2B_ALL_DONE
