#!/usr/bin/env bash
# 多智能体 pilot：PDDL 60 题，4 个变体 × 至多 3 种任务顺序。同一顺序 4 个变体并行，顺序之间串行。
# 每组花费到 8 元自动停（保证评估段能跑完）；每种顺序开跑前，按上一种顺序的实际花费预估，
# 累计会超过 48 元就不再启动。
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
last_seed_cost=20
for seed in 0 1 2; do
  before=$(total_cost)
  if python3 -c "import sys; sys.exit(0 if $before + $last_seed_cost > 48 else 1)"; then
    echo "TOTAL_BUDGET_STOP before seed $seed: spent $before, next ~$last_seed_cost CNY"; break
  fi
  for m in empty naive root_only team; do
    if [ -f $OUT/pddl_dylan_${m}_s${seed}/summary.json ]; then continue; fi
    ./gm.sh /home/claude/chime/run_mas.py --mode $m --order-seed $seed --budget-cny 8 --out $OUT \
      > $OUT/logs/${m}_s${seed}.log 2>&1 &
  done
  wait
  after=$(total_cost)
  last_seed_cost=$(python3 -c "print(max(round($after - $before, 3), 1))")
  echo "seed $seed done, this seed $last_seed_cost CNY, cumulative $after CNY"
done
echo ALL_DONE
