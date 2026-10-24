#!/usr/bin/env bash
# 单智能体 vs 多智能体对比（PDDL，任务顺序 s0，同一框架、同一参数，只有团队结构不同）。
# 多智能体无记忆基线沿用 pilot 中同配置的 pddl_dylan_empty_s0。
cd /home/claude/chime
OUT=/home/claude/chime/mas_runs/cmp
for m in single_nomem single_chime naive; do
  tag=$([ "${m#single}" != "$m" ] && echo "pddl_single_${m}_s0" || echo "pddl_dylan_${m}_s0")
  if [ -f $OUT/$tag/summary.json ]; then continue; fi
  ./gm.sh /home/claude/chime/run_mas.py --mode $m --order-seed 0 --budget-cny 8 --out $OUT > $OUT/logs/${m}_s0.log 2>&1 &
done
wait
echo CMP_DONE
