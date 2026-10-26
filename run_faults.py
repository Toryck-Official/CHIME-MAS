"""E1 第一步：在单智能体成功轨迹上做故障注入重跑，得到真实原因已知的失败样本。

条件（每条源轨迹各跑一次）：
  clean  使用原计划重跑，不注入（测重跑本身的随机性，作为标签噪声对照）
  plan   计划中改错一步后重跑（真相：planning）
  exec   原计划，第 L//3 步起连续 4 步强制随机合法动作（真相：execution）
  mem    规划库植入一条误导记忆，规划器据此重写计划（真相：planning，误导记忆 = PLANT_BAD）
源轨迹：mas_runs/cmp/pddl_single_single_chime_s0 中成功的题（训练段与评估段）。
用法：/home/claude/chime/gm.sh /home/claude/chime/run_faults.py --budget-cny 2.5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/claude/chime")
import run as R
from mas.llm import Message, get_usage
from prompts import get_dataset_system_prompt, get_task_few_shots

from chime.embed_qwen import qwen_embed
from chime.faults import FaultSingle, planted_item
from chime.llm import parse_json
from chime.mas_memory import dumps

SRC = "/home/claude/chime/mas_runs/cmp/pddl_single_single_chime_s0/episodes.jsonl"
CORRUPT = """Below is a plan written for an executor in a text-based planning environment.
Rewrite it so that it contains exactly ONE plausible but wrong planning decision that would make an executor who
follows it faithfully fail: e.g. a wrong ordering of two subgoals, a missing required step, a wrong target object or
location, or a wrong precondition. Everything else must stay the same, in the same format and length.

# Goal
{task}

# Plan
{plan}

Answer with ONE JSON object: {{"plan": "<the full rewritten plan>", "change": "<one sentence: what you changed>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="clean,plan,exec,mem")
    ap.add_argument("--budget-cny", type=float, default=2.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/home/claude/chime/mas_runs/e1")
    ap.add_argument("--act-hint", action="store_true", help="执行器加固定提示：直接输出动作、不写 think")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R.WORKING_DIR = os.path.join("./.db", "chime_mas", "e1")
    os.makedirs(R.WORKING_DIR, exist_ok=True)

    tm = R.build_task("pddl", "dylan", "empty", 30)
    tm.mas_config.update(node_num=1, round_num=1, use_critic=False, successful_topk=0, failed_topk=0,
                         insights_topk=0, threshold=0.0, use_projector=False)
    tm.mem_config.update(working_dir=R.WORKING_DIR, hop=1)
    R.build_mas(tm, "io", "empty", "deepseek-chat")
    llm = tm.mas.meta_memory.llm_model

    def llm_call(msgs, max_tokens=1500):
        return llm([Message(m["role"], m["content"]) for m in msgs], temperature=0.0, max_tokens=max_tokens)

    fs = FaultSingle(embed=qwen_embed, llm_call=llm_call)
    fs.act_hint = a.act_hint
    tm.mas.chime = fs

    tasks = {i: t for i, t in enumerate(tm.tasks)}
    src = [json.loads(l) for l in open(SRC)]
    src = [r for r in src if r["success"]]
    if a.limit:
        src = src[: a.limit]
    ep_path = os.path.join(a.out, "episodes.jsonl")
    done = set()
    base = {}
    if os.path.exists(ep_path):
        prev = [json.loads(l) for l in open(ep_path)]
        done = {(r["task_index"], r["cond"]) for r in prev}
        base = dict(prev[-1]["usage"]) if prev else {}
    log = open(ep_path, "a")

    def usage_now():
        u = get_usage()
        return {k: round(u[k] + base.get(k, 0), 4) for k in u}

    conds = a.conds.split(",")
    for cond in conds:  # 按条件分批，预算不够时至少每个条件都有样本
        for r in src:
            ti = r["task_index"]
            if (ti, cond) in done:
                continue
            cfg = dict(tasks[ti])
            cfg["task_index"] = ti
            game = cfg.get("game_name")
            orig_plan = r["chime"]["plan"]
            L = len(r["chime"]["steps"])
            t0 = time.time()
            task_main, task_description = tm.mas.env.set_env(cfg)
            few_shots = get_task_few_shots(dataset="pddl", task_config=cfg,
                                           few_shots_num=R.CONFIG.get("pddl").get("few_shots_num", 0))
            cfg.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
            instruction = get_dataset_system_prompt("pddl", task_config=cfg)
            for agent in tm.mas.agents_team.values():
                agent.add_task_instruction(instruction)
            fs.context = task_description
            meta = {}
            if cond == "clean":
                fs.configure(forced_plan=orig_plan)
            elif cond == "plan":
                raw = llm_call([{"role": "user", "content": CORRUPT.format(task=task_main, plan=orig_plan)}])
                try:
                    d = parse_json(raw)
                    bad_plan, change = str(d["plan"]).strip(), str(d.get("change", ""))
                except Exception:
                    print(f"corrupt parse failed for task {ti}", flush=True)
                    continue
                meta = {"change": change, "orig_plan": orig_plan}
                fs.configure(forced_plan=bad_plan)
            elif cond == "exec":
                sb = {"start": max(1, L // 3), "n": 4, "seed": ti}
                meta = {"sabotage_cfg": sb}
                fs.configure(forced_plan=orig_plan, sabotage=sb)
            elif cond == "mem":
                fs.configure(planted=[planted_item(game, "plan", "bad")])
            reward, ok = tm.mas.schedule(cfg)
            rec = {"cond": cond, "task_index": ti, "src_k": r["k"], "src_phase": r["phase"], "game": game,
                   "problem": cfg.get("problem_index"), "task_main": task_main, "src_steps": L,
                   "success": bool(ok), "progress": float(reward), "secs": round(time.time() - t0, 1),
                   "run": fs.finish(bool(ok), str(ti)), **meta}
            rec["usage"] = usage_now()
            log.write(dumps(rec) + "\n")
            log.flush()
            print(f"[{cond}] task {ti} {game} success={ok} progress={reward:.2f} "
                  f"cost={rec['usage']['cost_cny_offpeak']:.3f}", flush=True)
            if rec["usage"]["cost_cny_offpeak"] > a.budget_cny:
                print("BUDGET_STOP", flush=True)
                return
    print("E1_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
