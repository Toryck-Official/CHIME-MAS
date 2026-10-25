"""用经典规划器（pyperplan A* + LM-cut，可采纳启发式，结果最优）计算 PDDL 60 题的最优步数。
gripper 用闭式公式（与 gate_audit.py 相同），其余游戏跑规划器，单题超时记为 None。
输出 mas_runs/e1/optimal_steps.json：{game: {problem_index: steps}}
"""
import json, os, re, subprocess, sys
sys.path.insert(0, "/home/claude/chime")
from gate_audit import gripper_optimal
D = "/home/claude/refs/gmemory/tasks/envs/pddl_env/pddlgym/pddl/"
PY = "/home/claude/refs/gmemory/.venv/bin/python"
out = {}
for game in ("gripper", "blockworld", "tyreworld", "barman"):
    files = sorted(f for f in os.listdir(D + game) if f.endswith(".pddl"))
    out[game] = {}
    for i, f in enumerate(files):
        if game == "gripper":
            out[game][i] = gripper_optimal(D + game + "/" + f); continue
        try:
            p = subprocess.run([PY, "-m", "pyperplan", "-s", "astar", "-H", "lmcut", D + game + ".pddl", D + game + "/" + f],
                               capture_output=True, text=True, timeout=240)
            m = re.search(r"Plan length: (\d+)", p.stdout + p.stderr)
            out[game][i] = int(m.group(1)) if m else None
        except subprocess.TimeoutExpired:
            out[game][i] = None
        print(game, i, f, out[game][i], flush=True)
        for s in os.listdir(D + game):
            if s.endswith(".soln"):
                os.remove(D + game + "/" + s)
json.dump(out, open("/home/claude/chime/mas_runs/e1/optimal_steps.json", "w"), indent=1)
print("OPT_DONE")
