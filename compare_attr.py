"""同一批多智能体轨迹上，对比两种归因视角，量化"单智能体 CHIME 搬到多智能体"后信用分配差在哪里。

  局部式（朴素多智能体 CHIME，已在线运行）：每个角色只看自己的输出 + 执行结果，各自判断 self / none
  集中式（单智能体 CHIME 的做法，本脚本离线补做）：一个归因门看完整团队轨迹，指出该担功劳/责任的角色集合

用法（经 gm.sh，使用同一骨干与计费）：
  /home/claude/chime/gm.sh /home/claude/chime/compare_attr.py mas_runs/cmp/pddl_dylan_naive_s0
输出 <run>/attr_compare.jsonl（逐题）与 attr_compare_summary.json。
"""
import json
import os
import sys

sys.path.insert(0, "/home/claude/chime")
from mas.llm import GPTChat, Message, get_usage

from chime.llm import parse_json
from chime.mas_memory import CENTRAL_GATE, ROLE_DESC, ROLES, render_team_trace, steps_from_json

def task_goals() -> dict:
    """按题号取回每道题的目标描述（与运行时 set_env 返回的 task_main 相同）。"""
    import yaml
    import envs
    cfg = yaml.safe_load(open("tasks/env_configs/pddl_config.yaml"))
    env = envs.get_env("pddl", cfg, 30)
    goals = {}
    for i, t in enumerate(envs.get_task("pddl")):
        main_, _ = env.set_env(dict(t))
        goals[i] = main_
    return goals


def main(run_dir):
    llm = GPTChat("deepseek-chat")
    goals = task_goals()
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    rows = [r for r in rows if r["phase"] == "train" and "steps" in (r.get("chime") or {})]
    out_path = os.path.join(run_dir, "attr_compare.jsonl")
    done = {json.loads(l)["k"] for l in open(out_path)} if os.path.exists(out_path) else set()
    f = open(out_path, "a")
    for r in rows:
        if r["k"] in done:
            continue
        c = r["chime"]
        steps = steps_from_json(c["steps"])
        outcome = "SUCCESS (all goal conditions satisfied)" if r["success"] else "FAILURE (goal not reached within the step limit)"
        prompt = CENTRAL_GATE.format(task=goals[r["task_index"]],
                                     trace=render_team_trace(steps), outcome=outcome, **ROLE_DESC)
        raw = llm([Message("user", prompt)], temperature=0.0, max_tokens=800)
        try:
            d = parse_json(raw)
        except Exception:
            d = {"attributed": [], "_unparsed": raw[:300]}
        central = sorted({x for x in (d.get("attributed") or []) if x in ROLES})
        local = sorted(role for role, g in (c.get("gates") or {}).items()
                       if str((g.get("raw") or {}).get("attribution", "")).lower() == "self")
        rec = {"k": r["k"], "success": r["success"], "game": r["game"], "central": central, "local": local,
               "central_reason": d.get("reason"), "central_conf": d.get("confidence")}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(f"k={r['k']} success={r['success']} central={central} local={local}", flush=True)
    summarize(run_dir)
    print("USAGE", json.dumps(get_usage()))


def summarize(run_dir):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "attr_compare.jsonl"))]
    out = {}
    for name, sel in [("all", recs), ("success", [x for x in recs if x["success"]]),
                      ("failure", [x for x in recs if not x["success"]])]:
        n = len(sel)
        if not n:
            continue
        over = sum(len(set(x["local"]) - set(x["central"])) for x in sel)   # 局部自认、集中不认：错记功 / 错写教训
        under = sum(len(set(x["central"]) - set(x["local"])) for x in sel)  # 集中认定、局部没认：漏掉的责任
        out[name] = {
            "n": n,
            "local_mean_roles": round(sum(len(x["local"]) for x in sel) / n, 2),
            "central_mean_roles": round(sum(len(x["central"]) for x in sel) / n, 2),
            "local_all_three_pct": round(100 * sum(len(x["local"]) == 3 for x in sel) / n, 1),
            "exact_agreement_pct": round(100 * sum(x["local"] == x["central"] for x in sel) / n, 1),
            "over_claims_per_episode": round(over / n, 2),
            "under_claims_per_episode": round(under / n, 2),
            "per_role": {r: {"local_claims": sum(r in x["local"] for x in sel),
                             "central_attributed": sum(r in x["central"] for x in sel),
                             "local_only": sum(r in x["local"] and r not in x["central"] for x in sel),
                             "central_only": sum(r in x["central"] and r not in x["local"] for x in sel)}
                         for r in ROLES},
        }
    with open(os.path.join(run_dir, "attr_compare_summary.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
