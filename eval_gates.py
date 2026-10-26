"""E1 第二步：在真实原因已知的失败样本上，比较不同归因门的判断。

样本来源
  plan / exec / mem   run_faults.py 注入后失败的重跑（strict = 同一题干净重跑成功，反事实成立）
  env                 干净重跑成功的轨迹截短到 T = min(L-1, 最优步数-1) 步：完美执行也无法在 T 步内完成，真相 none
  natural             单智能体 CHIME 训练段：barman 不可执行计划（真相 planning）、gripper 超出步数上限（真相 none）
  mas_env / mas_nat   多智能体朴素 CHIME 的成功轨迹按同样规则截短，以及 4 道 gripper 超限失败；真相：没有角色有责任

单智能体归因门条件
  global     CHIME 原版归因门（同骨干，看任务、计划、完整轨迹、检索记忆）
  plan_view  只给规划侧信息：任务、计划、规划记忆、结果，不给执行轨迹
  exec_view  只给执行侧信息：任务、收到的计划、轨迹，不给规划记忆（只在 mem 样本上与 global 不同）
  self       规划者、执行者各自判断"是否是我的责任"（多智能体朴素 CHIME 的做法），合成阶段标签
  grounded   global + 环境事实（步数上限、已用步数、无效动作数、think 次数、经典规划器给出的最优步数）
  pro        global，换用 DeepSeek-V4-Pro（低推理强度）
  grounded_pro  grounded，换用 DeepSeek-V4-Pro
  global_rep global 的重复调用（温度 0），用于估计同一归因门的自身波动
多智能体条件：local（三个角色各自判断）/ central（一个归因门看全程）/ central_grounded
用法：python3 -I eval_gates.py --conds global,plan_view,exec_view,self,grounded --budget-cny 3
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading

sys.path.insert(0, "/home/claude/chime")
from chime.gate import GATE_SYSTEM, GATE_USER, format_memories
from chime.llm import USAGE, chat, parse_json
from chime.mas_memory import CENTRAL_GATE, NAIVE_GATE, ROLE_DESC, ROLES, render_role_view, render_team_trace, steps_from_json
from chime.memory import MemoryItem
from chime.faults import planted_item


E1 = "/home/claude/chime/mas_runs/e1"  # 最优步数、任务目标等共用文件
SAMPLES = os.environ.get("E1_SAMPLES", E1)  # 故障样本与评测结果所在目录（干净执行器版本为 mas_runs/e1c）
SINGLE = "/home/claude/chime/mas_runs/cmp/pddl_single_single_chime_s0/episodes.jsonl"
MAS = "/home/claude/chime/mas_runs/cmp/pddl_dylan_naive_s0/episodes.jsonl"
AUDIT = "/home/claude/chime/mas_runs/cmp/gate_audit.json"
OUT = os.path.join(SAMPLES, "gate_results.jsonl")
LOCK = threading.Lock()


def clip(t, n=200):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def render_traj(steps):
    return "\n".join(f"[step {s['idx']}] executed: {clip(s['action'])} | observation: {clip(s['observation'])}"
                     for s in steps)


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def optimal():
    d = json.load(open(os.path.join(E1, "optimal_steps.json")))
    return {g: {int(k): v for k, v in m.items()} for g, m in d.items()}


def facts(s):
    steps = s["steps"]
    inv = sum("not valid" in st["observation"].lower() for st in steps)
    th = sum(st["action"].lower().startswith("think") for st in steps)
    cv = sum(st["action"].lower().startswith("check valid") for st in steps)
    opt = s.get("optimal")
    return ("# Environment facts (computed by the environment and an optimal classical planner)\n"
            f"- Step limit: {s['budget']}; steps used: {len(steps)}\n"
            f"- Invalid actions: {inv}; 'think' actions: {th}; 'check valid actions' calls: {cv}\n"
            f"- Minimum number of steps an optimal solver needs for this task: {opt if opt else 'unknown'}\n")


# ---------------- 样本 ----------------
def single_samples():
    opt = optimal()
    S = []
    clean = {r["task_index"]: r for r in load(os.path.join(SAMPLES, "clean", "episodes.jsonl"))}
    for cond in ("plan", "mem"):  # 故障被执行器补救、任务仍成功：真相是只有执行该记功
        for r in load(os.path.join(SAMPLES, cond, "episodes.jsonl")):
            if not r["success"] or os.environ.get("E1_SUCC", "0") != "1":
                continue
            S.append({"id": f"{cond}_succ-{r['task_index']}", "kind": f"{cond}_succ", "game": r["game"],
                      "task": r["task_main"], "plan": r["run"]["plan"], "steps": r["run"]["steps"], "budget": 30,
                      "optimal": opt[r["game"]].get(r["problem"]), "truth": "execution",
                      "truth_mis": ["PLANT_BAD"] if cond == "mem" else [],
                      "retrieved": ["PLANT_BAD"] if cond == "mem" else [], "strict": True, "success": True})
    for cond, truth in (("plan", "planning"), ("exec", "execution"), ("mem", "planning")):
        for r in load(os.path.join(SAMPLES, cond, "episodes.jsonl")):
            if r["success"]:
                continue
            c = clean.get(r["task_index"])
            S.append({"id": f"{cond}-{r['task_index']}", "kind": cond, "game": r["game"], "task": r["task_main"],
                      "plan": r["run"]["plan"], "steps": r["run"]["steps"], "budget": 30,
                      "optimal": opt[r["game"]].get(r["problem"]), "truth": truth,
                      "truth_mis": ["PLANT_BAD"] if cond == "mem" else [],
                      "retrieved": ["PLANT_BAD"] if cond == "mem" else [],
                      "strict": bool(c and c["success"])})
    for ti, r in clean.items():
        if not r["success"]:
            continue
        L, o = len(r["run"]["steps"]), opt[r["game"]].get(r["problem"])
        if not o:
            continue
        T = min(L - 1, o - 1)
        if T < 3:
            continue
        S.append({"id": f"env-{ti}", "kind": "env", "game": r["game"], "task": r["task_main"],
                  "plan": r["run"]["plan"], "steps": r["run"]["steps"][:T], "budget": T, "optimal": o,
                  "truth": "none", "truth_mis": [], "retrieved": [], "strict": True})
    audit = json.load(open(AUDIT))["single"]
    goals = {int(k): v for k, v in json.load(open(os.path.join(E1, "task_goals.json"))).items()}
    src = {r["k"]: r for r in load(SINGLE)}
    for key, truth in (("barman_invalid_plan_failures", "planning"), ("gripper_infeasible_failures", "none")):
        for x in audit[key]:
            r = src[x["k"]]
            S.append({"id": f"nat-{r['task_index']}", "kind": "natural", "game": r["game"], "task": goals[r["task_index"]],
                      "plan": r["chime"]["plan"], "steps": r["chime"]["steps"], "budget": 30,
                      "optimal": opt[r["game"]].get(r["problem"]), "truth": truth, "truth_mis": [],
                      "retrieved": [], "strict": True})
    return S


def mas_samples():
    opt = optimal()
    S = []
    audit = json.load(open(AUDIT))["mas"]
    goals = {int(k): v for k, v in json.load(open(os.path.join(E1, "task_goals.json"))).items()}
    infeasible = {x["k"] for x in audit["gripper_infeasible_failures"]}
    for r in load(MAS):
        st = (r.get("chime") or {}).get("steps")
        if not st:
            continue
        o = opt[r["game"]].get(r["problem"])
        if r["success"] and o:
            T = min(len(st) - 1, o - 1)
            if T >= 3:
                S.append({"id": f"mas_env-{r['task_index']}", "kind": "mas_env", "game": r["game"],
                          "task": goals[r["task_index"]], "steps": st[:T], "budget": T, "optimal": o})
        elif r["phase"] == "train" and r["k"] in infeasible:
            S.append({"id": f"mas_nat-{r['task_index']}", "kind": "mas_nat", "game": r["game"],
                      "task": goals[r["task_index"]], "steps": st, "budget": 30, "optimal": o})
    return S


# ---------------- 归因门 ----------------
def mems(s, stage=None):
    items = [planted_item(s["game"], "plan", "bad")] if s["retrieved"] else []
    return [m for m in items if stage is None or m.stage == stage]


def ask(prompt_msgs, model="deepseek-chat"):
    if model == "deepseek-v4-pro":  # 推理模型：低推理强度，留足输出长度，避免推理耗尽 max_tokens 后正文为空
        raw = chat(prompt_msgs, model=model, max_tokens=4000, reasoning_effort="low")
    else:
        raw = chat(prompt_msgs, model=model, max_tokens=1500)
    try:
        return parse_json(raw)
    except Exception:
        return {"_unparsed": raw[:300]}


def label_of(d):
    lab = str(d.get("attribution", "none")).strip().lower()
    return lab if lab in ("planning", "execution", "both", "none") else "none"


def num(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def gate_global(s, cond):
    outcome = "SUCCESS (reward=1.0)" if s.get("success") else "FAILURE (reward=0.0)"
    traj = render_traj(s["steps"])
    plan_mem = mems(s)
    if cond == "plan_view":
        traj = "(not visible: you only know the plan and the final outcome)"
    if cond == "exec_view":
        plan_mem = []
    user = GATE_USER.format(task=s["task"], plan=s["plan"], trajectory=traj, outcome=outcome,
                            memories=format_memories(plan_mem))
    if cond in ("grounded", "grounded_pro"):
        user = user.replace("# Memories retrieved for this episode", facts(s) + "\n# Memories retrieved for this episode")
    d = ask([{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}],
            model="deepseek-v4-pro" if cond in ("pro", "grounded_pro") else "deepseek-chat")
    return {"label": label_of(d), "conf": num(d.get("confidence")),
            "mis": [i for i in d.get("misleading_memory_ids") or [] if i == "PLANT_BAD"], "raw": d}


SELF_ROLE = {"planner": "writes the plan for the task before execution; you do not act",
             "executor": "carries out the plan step by step with environment actions"}


def gate_self(s):
    out = {}
    for role in ("planner", "executor"):
        view = (f"Your plan:\n{s['plan']}\n\nExecuted actions and observations:\n" if role == "planner"
                else f"The plan you received:\n{s['plan']}\n\nYour actions and observations:\n") + render_traj(s["steps"])
        prompt = NAIVE_GATE.format(role=role, role_desc=SELF_ROLE[role], task=s["task"], view=view,
                                   outcome=("SUCCESS (all goal conditions satisfied)" if s.get("success")
                                            else "FAILURE (goal not reached within the step limit)"),
                                   memories=format_memories(mems(s) if role == "planner" else []))
        out[role] = ask([{"role": "user", "content": prompt}])
    p = str(out["planner"].get("attribution", "")).lower() == "self"
    e = str(out["executor"].get("attribution", "")).lower() == "self"
    lab = "both" if p and e else "planning" if p else "execution" if e else "none"
    mis = [i for i in out["planner"].get("misleading_memory_ids") or [] if i == "PLANT_BAD"]
    return {"label": lab, "conf": (num(out["planner"].get("confidence")) + num(out["executor"].get("confidence"))) / 2,
            "mis": mis, "raw": out}


def gate_mas(s, cond):
    outcome = "FAILURE (goal not reached within the step limit)"
    steps = steps_from_json(s["steps"])
    if cond == "local":
        res = {}
        for role in ROLES:
            prompt = NAIVE_GATE.format(role=role, role_desc=ROLE_DESC[role], task=s["task"],
                                       view=render_role_view(steps, role), outcome=outcome, memories="(none)")
            res[role] = ask([{"role": "user", "content": prompt}])
        blamed = sorted(r for r in ROLES if str(res[r].get("attribution", "")).lower() == "self")
        lessons = sum(bool(str(res[r].get("experience", "")).strip()) for r in blamed)
        return {"blamed": blamed, "lessons": lessons, "raw": res}
    prompt = CENTRAL_GATE.format(task=s["task"], trace=render_team_trace(steps), outcome=outcome, **ROLE_DESC)
    if cond == "central_grounded":
        prompt = prompt.replace("# Outcome", facts(s) + "\n# Outcome")
    d = ask([{"role": "user", "content": prompt}])
    blamed = sorted({x for x in (d.get("attributed") or []) if x in ROLES})
    return {"blamed": blamed, "lessons": len(blamed), "raw": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="global,plan_view,exec_view,self,grounded")
    ap.add_argument("--mas-conds", default="local,central,central_grounded")
    ap.add_argument("--kinds", default="", help="只评测这些样本类型（逗号分隔），默认全部")
    ap.add_argument("--budget-cny", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--strict-only", action="store_true", help="只评测 strict 样本（干净重跑成功的题）")
    a = ap.parse_args()
    done = {(r["id"], r["cond"]) for r in load(OUT)}
    jobs = []
    kinds = set(a.kinds.split(",")) if a.kinds else None
    for s in single_samples():
        if (kinds and s["kind"] not in kinds) or (a.strict_only and not s["strict"]):
            continue
        for c in [x for x in a.conds.split(",") if x]:
            if c == "exec_view" and s["kind"] != "mem":
                continue
            if (s["id"], c) not in done:
                jobs.append((s, c, "single"))
    for s in mas_samples():
        if kinds and s["kind"] not in kinds:
            continue
        for c in [x for x in a.mas_conds.split(",") if x]:
            if (s["id"], c) not in done:
                jobs.append((s, c, "mas"))
    print(f"{len(jobs)} gate jobs", flush=True)
    f = open(OUT, "a")
    stop = threading.Event()

    def run(job):
        s, c, kind = job
        if stop.is_set():
            return
        res = gate_mas(s, c) if kind == "mas" else gate_self(s) if c == "self" else gate_global(s, c)
        rec = {"id": s["id"], "cond": c, "kind": s["kind"], "game": s["game"], "truth": s.get("truth", "none"),
               "truth_mis": s.get("truth_mis", []), "strict": s.get("strict", True), **res}
        with LOCK:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if USAGE["cost_cny_offpeak"] > a.budget_cny:
                stop.set()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    print("USAGE", json.dumps(USAGE), "STOPPED" if stop.is_set() else "E1_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
