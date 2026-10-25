"""离线核查（不调用 API）：用 PDDL 可验证的事实，检查 CHIME 归因门在单智能体与多智能体轨迹上的判断。

两类可自动判定的真相：
  barman：域定义中原料只能经 fill-shot 进入量杯，再由 pour-shot-to-clean-shaker 倒进摇壶。
          计划写成"从分配器直接把原料加入摇壶"即为不可执行的计划步骤（计划错误）。
  gripper：按两爪容量算出最优步数；最优步数 > 30 的题，即使完美执行也无法在步数上限内完成，
          失败的真实原因是环境预算（CHIME 标签应为 none），不是执行。
用法：python3 -I gate_audit.py
"""
import json
import math
import os
import re

GRIPPER_DIR = "/home/claude/refs/gmemory/tasks/envs/pddl_env/pddlgym/pddl/gripper/"
SINGLE = "mas_runs/cmp/pddl_single_single_chime_s0"
MAS = "mas_runs/cmp/pddl_dylan_naive_s0"
DIRECT_TO_SHAKER = re.compile(
    r"(add|fill)[^|\n]{0,60}(ingredient\d|part ?\d)[^|\n]{0,40}(to|into|in) shaker"
    r"|(fill|add to) shaker\d? with ingredient|pour ingredient\d into shaker|fill shaker")
PLAN_SOUND = re.compile(r"plan (was|is) (sound|correct|reasonable|adequate|fine)|plan correctly")


def gripper_optimal(path):
    txt = open(path).read()
    init, goal = txt.split(":goal")
    start = re.search(r"\(at-robby (\w+)\)", init).group(1)
    ini = dict(re.findall(r"\(at (ball\d+) (\w+)\)", init))
    gl = dict(re.findall(r"\(at (ball\d+) (\w+)\)", goal))
    a = sum(ini[b] == "rooma" and gl[b] == "roomb" for b in gl)
    b = sum(ini[x] == "roomb" and gl[x] == "rooma" for x in gl)
    ca, cb = math.ceil(a / 2), math.ceil(b / 2)
    cross = min(x + y for x in range(ca, 20) for y in range(cb, 20)
                if ((x - y) in (0, 1) if start == "rooma" else (y - x) in (0, 1)))
    return 2 * (a + b) + cross


def load(run):
    return [json.loads(l) for l in open(os.path.join(run, "episodes.jsonl"))]


def main():
    files = sorted(os.listdir(GRIPPER_DIR))
    out = {"single": {"barman_invalid_plan_failures": [], "barman_invalid_plan_successes": [],
                      "gripper_infeasible_failures": []},
           "mas": {"gripper_infeasible_failures": []}}
    for r in load(SINGLE):
        if r["phase"] != "train":
            continue
        g = r["chime"]["gate"]
        if r["game"] == "barman" and DIRECT_TO_SHAKER.search(r["chime"]["plan"].lower()):
            rec = {"k": r["k"], "attribution": g.get("attribution"), "confidence": g.get("confidence"),
                   "says_plan_sound": bool(PLAN_SOUND.search((g.get("reason") or "").lower())),
                   "reason": g.get("reason")}
            out["single"]["barman_invalid_plan_successes" if r["success"] else "barman_invalid_plan_failures"].append(rec)
        if r["game"] == "gripper" and not r["success"]:
            opt = gripper_optimal(GRIPPER_DIR + files[r["problem"]])
            if opt > 30:
                out["single"]["gripper_infeasible_failures"].append(
                    {"k": r["k"], "optimal_steps": opt, "attribution": g.get("attribution"),
                     "confidence": g.get("confidence"), "reason": g.get("reason")})
    central = {}
    cpath = os.path.join(MAS, "attr_compare.jsonl")
    if os.path.exists(cpath):
        central = {json.loads(l)["k"]: json.loads(l)["central"] for l in open(cpath)}
    for r in load(MAS):
        if r["phase"] != "train" or r["game"] != "gripper" or r["success"]:
            continue
        opt = gripper_optimal(GRIPPER_DIR + files[r["problem"]])
        if opt > 30:
            gates = r["chime"]["gates"]
            out["mas"]["gripper_infeasible_failures"].append(
                {"k": r["k"], "optimal_steps": opt,
                 "local_self": sorted(ro for ro, g in gates.items()
                                      if str((g.get("raw") or {}).get("attribution", "")).lower() == "self"),
                 "central": central.get(r["k"]),
                 "lessons": {ro: (g.get("raw") or {}).get("experience") for ro, g in gates.items()}})
    s = out["single"]
    single_fail = sum(1 for r in load(SINGLE) if r["phase"] == "train" and not r["success"])
    out["summary"] = {
        "single_train_failures": single_fail,
        "barman_invalid_plan_failures": len(s["barman_invalid_plan_failures"]),
        "of_which_blamed_on_execution_with_plan_called_sound": sum(
            x["attribution"] == "execution" and x["says_plan_sound"] for x in s["barman_invalid_plan_failures"]),
        "barman_invalid_plan_successes_credited_to_planning": sum(
            x["attribution"] in ("planning", "both") for x in s["barman_invalid_plan_successes"]),
        "gripper_infeasible_failures": len(s["gripper_infeasible_failures"]),
        "of_which_labelled_none": sum(x["attribution"] == "none" for x in s["gripper_infeasible_failures"]),
        "mas_gripper_infeasible_failures": len(out["mas"]["gripper_infeasible_failures"]),
        "mas_local_or_central_none": sum(not x["local_self"] or not x["central"]
                                         for x in out["mas"]["gripper_infeasible_failures"]),
    }
    with open("mas_runs/cmp/gate_audit.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
