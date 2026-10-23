#!/usr/bin/env bash
# 多智能体 pilot：PDDL 60 题，4 个变体 × 3 种任务顺序。同一顺序 4 个变体并行，顺序之间串行。
# 每组花费到 4 元自动停；累计花费超过 45 元则不再启动新一组。
cd /home/claude/chime
OUT=/home/claude/chime/mas_runs
mkdir -p $OUT/logs
total_cost() {
  python3 -I -c "
import json,glob
t=0
for f in glob.glob('$OUT/pddl_dylan_*/episodes.jsonl'):
    lines=open(f).read().strip().splitlines()
    if lines: t+=json.loads(lines[-1])['usage']['cost_cny_offpeak']
print(round(t,3))"
}
for seed in 0 1 2; do
  c=$(total_cost)
  if python3 -c "import sys; sys.exit(0 if $c > 45 else 1)"; then echo "TOTAL_BUDGET_STOP before seed $seed: $c CNY"; break; fi
  for m in empty naive root_only team; do
    if [ -f $OUT/pddl_dylan_${m}_s${seed}/summary.json ]; then continue; fi
    ./gm.sh /home/claude/chime/run_mas.py --mode $m --order-seed $seed --budget-cny 4 --out $OUT \
      > $OUT/logs/${m}_s${seed}.log 2>&1 &
  done
  wait
  echo "seed $seed done, cumulative $(total_cost) CNY"
done
echo ALL_DONE
