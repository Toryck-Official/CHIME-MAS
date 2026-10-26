"""H1 / H3 pilot：在 DyLAN 2×2 + critic 团队上运行三世界反事实探针（chime/probe.py）。

记忆库：危害实验多智能体原版（pddl_dylan_naive_orig_hint_s0）训练结束时的三个角色记忆库，冻结不更新。
任务：同一任务顺序 s0 的评估段题目（与该记忆库的训练段不重叠），取前 --n-tasks 题；提议者与定稿者加固定提示。
在第 1、4、7、10 步（--probe-steps，0 起计）运行探针。
用法：/home/claude/chime/gm.sh /home/claude/chime/run_probe.py --n-tasks 12 --budget-cny 3
输出：mas_runs/probe/probes.jsonl（每条探针一行）、episodes.jsonl（每题一行）
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "/home/claude/chime")
import run as R
from mas.llm import Message, get_usage
from prompts import get_dataset_system_prompt, get_task_few_shots

from chime.embed_qwen import qwen_embed
from chime.mas_memory import dumps
from chime.probe import ProbeMAS

MEM = "/home/claude/chime/mas_runs/harm/pddl_dylan_naive_orig_hint_s0/memory.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--start", type=int, default=0, help="从评估段第几题开始（并行分片用）")
    ap.add_argument("--probe-steps", default="0,3,6,9")
    ap.add_argument("--budget-cny", type=float, default=3.0)
    ap.add_argument("--out", default="/home/claude/chime/mas_runs/probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R.WORKING_DIR = os.path.join("./.db", "chime_probe")
    os.makedirs(R.WORKING_DIR, exist_ok=True)

    tm = R.build_task("pddl", "dylan", "empty", 30)
    tm.mas_config.update(use_critic=True, successful_topk=0, failed_topk=0, insights_topk=0, threshold=0.0,
                         use_projector=False)
    tm.mem_config.update(working_dir=R.WORKING_DIR, hop=1)
    R.build_mas(tm, "io", "empty", "deepseek-chat")

    ep_path = os.path.join(a.out, "episodes.jsonl")
    done = [json.loads(l) for l in open(ep_path)] if os.path.exists(ep_path) else []
    base = dict(done[-1]["usage"]) if done else {}

    def usage_now():
        u = get_usage()
        return {k: round(u[k] + base.get(k, 0), 4) for k in u}

    pm = ProbeMAS(embed=qwen_embed, llm_call=lambda msgs: "", out_path=os.path.join(a.out, "probes.jsonl"),
                  probe_steps=[int(x) for x in a.probe_steps.split(",")],
                  budget_check=lambda: usage_now()["cost_cny_offpeak"] > a.budget_cny)
    pm.mem.load_json(open(MEM).read())
    pm.act_hint = True
    tm.mas.chime = pm

    tasks = [dict(t, task_index=i) for i, t in enumerate(tm.tasks)]
    random.Random(0).shuffle(tasks)
    cut = int(round(len(tasks) * 0.7))
    tasks = tasks[cut:][a.start: a.start + a.n_tasks]
    log = open(ep_path, "a")
    for k, cfg in enumerate(tasks):
        if k < len(done):
            continue
        t0 = time.time()
        task_main, task_description = tm.mas.env.set_env(cfg)
        few_shots = get_task_few_shots(dataset="pddl", task_config=cfg,
                                       few_shots_num=R.CONFIG.get("pddl").get("few_shots_num", 0))
        cfg.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
        instruction = get_dataset_system_prompt("pddl", task_config=cfg)
        for agent in tm.mas.agents_team.values():
            agent.add_task_instruction(instruction)
        tm.mas.env.max_trials = 30
        pm.task_index = cfg["task_index"]
        reward, ok = tm.mas.schedule(cfg)
        c = pm.finish(bool(ok), task_id=str(cfg["task_index"]))
        rec = {"k": k, "task_index": cfg["task_index"], "game": cfg["game_name"], "success": bool(ok),
               "progress": float(reward), "secs": round(time.time() - t0, 1), "retrieved": c["retrieved"],
               "steps": c["steps"], "usage": usage_now()}
        log.write(dumps(rec) + "\n")
        log.flush()
        print(f"[probe {k+1}/{len(tasks)}] {cfg['game_name']}#{cfg['problem_index']} success={ok} "
              f"cost={rec['usage']['cost_cny_offpeak']:.3f} {rec['secs']:.0f}s", flush=True)
        if rec["usage"]["cost_cny_offpeak"] > a.budget_cny:
            print("BUDGET_STOP", flush=True)
            break
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
