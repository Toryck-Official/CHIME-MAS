"""危害实验：从无责失败中学习，会不会伤害 CHIME？（由 run_mas.py 改写）

训练段随机抽约 1/3 的题（只抽最优步数已知且 ≥ 4 的题），把步数上限设为"最优步数 − 1"，使其任何策略都无法完成；
其余训练题和全部评估题的步数上限为 30。两种条件使用同一批不可行题（同一随机种子），配对比较：
  orig    CHIME 原版：无责失败照常交给归因门
  oracle  可行性预言机（经典规划器给出的最优步数 > 步数上限）判定为无责失败时，不调用归因门、不更新记忆
记录每题的步数上限、是否无责、写入的记忆条数，用于统计"来自无责失败的教训"占比。
用法：/home/claude/chime/gm.sh /home/claude/chime/run_harm.py --mode single_chime --cond oracle --order-seed 0 --act-hint

（以下为原 run_mas.py 说明）多智能体 CHIME 实验：G-Memory 的 DyLAN（2×2，开启 critic）+ PDDL。

用法（经 gm.sh 在 G-Memory 目录运行）：
  /home/claude/chime/gm.sh /home/claude/chime/run_mas.py --mode team --order-seed 0
mode: empty | naive | root_only | team（DyLAN 2×2 + critic）；single_nomem | single_chime（同框架 1×1，先规划后执行）
训练段（前 70%）顺序执行并更新记忆；评估段冻结记忆。每题记录成功与否、进度分、累计花费。
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "/home/claude/chime")
import run as R  # G-Memory 的 tasks/run.py（经 PYTHONPATH=.:tasks 导入）
from mas.llm import Message, get_usage
from prompts import get_dataset_system_prompt, get_task_few_shots

from chime.embed_qwen import qwen_embed
from chime.mas_memory import ChimeMAS, ChimeSingle, dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["empty", "naive", "root_only", "team", "single_nomem", "single_chime"])
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--budget-cny", type=float, default=0, help="累计花费超过此值即停止（0 表示不限）")
    ap.add_argument("--out", default="/home/claude/chime/mas_runs/harm")
    ap.add_argument("--cond", required=True, choices=["orig", "oracle"])
    ap.add_argument("--frac-infeasible", type=float, default=0.34)
    ap.add_argument("--act-hint", action="store_true")
    a = ap.parse_args()
    opt = {g: {int(k): v for k, v in m.items()} for g, m in
           json.load(open("/home/claude/chime/mas_runs/e1/optimal_steps.json")).items()}

    single = a.mode.startswith("single")
    tag = f"pddl_{'single' if single else 'dylan'}_{a.mode}_{a.cond}{'_hint' if a.act_hint else ''}_s{a.order_seed}"
    outdir = os.path.join(a.out, tag)
    os.makedirs(outdir, exist_ok=True)
    R.WORKING_DIR = os.path.join("./.db", "chime_mas", tag)
    os.makedirs(R.WORKING_DIR, exist_ok=True)

    tm = R.build_task("pddl", "dylan", "empty", a.max_steps)
    if single:  # 同一框架缩成一个 agent：1 个节点、1 轮、无 critic
        tm.mas_config.update(node_num=1, round_num=1)
    tm.mas_config.update(use_critic=not single, successful_topk=0, failed_topk=0, insights_topk=0, threshold=0.0,
                         use_projector=False)
    tm.mem_config.update(working_dir=R.WORKING_DIR, hop=1)
    R.build_mas(tm, "io", "empty", "deepseek-chat")

    llm = tm.mas.meta_memory.llm_model

    def llm_call(msgs):
        return llm([Message(m["role"], m["content"]) for m in msgs], temperature=0.0, max_tokens=1500)

    def merger(old, scenario, experience):
        out = llm_call([{"role": "user", "content":
                         "Merge two lessons about the same situation into ONE concise, general, actionable guideline "
                         f"(max 3 sentences).\nSituation: {old.scenario}\nA: {old.experience}\nB: {experience}\n"
                         "Return only the merged guideline."}])
        return old.scenario, (out or old.experience).strip()

    chime = None
    if single:
        chime = ChimeSingle(a.mode, embed=qwen_embed, llm_call=llm_call, merger=merger)
        tm.mas.chime = chime
    elif a.mode != "empty":
        chime = ChimeMAS(a.mode, embed=qwen_embed, llm_call=llm_call, merger=merger)
        tm.mas.chime = chime
    if chime is not None:
        chime.act_hint = a.act_hint

    tasks = list(tm.tasks)
    for i, t in enumerate(tasks):
        t["task_index"] = i
    random.Random(a.order_seed).shuffle(tasks)
    if a.limit:
        tasks = tasks[: a.limit]
    cut = int(round(len(tasks) * 0.7))
    cand = [k for k, t in enumerate(tasks[:cut]) if (opt[t["game_name"]].get(t["problem_index"]) or 0) >= 4]
    infeasible = set(random.Random(1000 + a.order_seed).sample(cand, int(round(a.frac_infeasible * cut))))

    # 断点续跑：读回已完成的题目、最近一次保存的记忆和累计花费
    ep_path = os.path.join(outdir, "episodes.jsonl")
    done_recs = [json.loads(l) for l in open(ep_path)] if os.path.exists(ep_path) else []
    base_usage = dict(done_recs[-1]["usage"]) if done_recs else {}
    if done_recs and chime is not None and os.path.exists(os.path.join(outdir, "memory.json")):
        chime.mem.load_json(open(os.path.join(outdir, "memory.json")).read())
    if done_recs:
        print(f"RESUME from task {len(done_recs)} (cost so far {base_usage.get('cost_cny_offpeak', 0):.3f} CNY)", flush=True)

    def usage_now():
        u = get_usage()
        return {k: round(u[k] + base_usage.get(k, 0), 4) for k in u}

    log = open(ep_path, "a")
    results = {"train": [r for r in done_recs if r["phase"] == "train"],
               "eval": [r for r in done_recs if r["phase"] == "eval"]}
    for k, cfg in enumerate(tasks):
        if k < len(done_recs):
            continue
        phase = "train" if k < cut else "eval"
        if chime is not None:
            chime.frozen = phase == "eval"
        t0 = time.time()
        task_main, task_description = tm.mas.env.set_env(cfg)
        few_shots = get_task_few_shots(dataset="pddl", task_config=cfg,
                                       few_shots_num=R.CONFIG.get("pddl").get("few_shots_num", 0))
        cfg.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
        instruction = get_dataset_system_prompt("pddl", task_config=cfg)
        for agent in tm.mas.agents_team.values():
            agent.add_task_instruction(instruction)
        if single:
            chime.context = task_description
        o = opt[cfg["game_name"]].get(cfg["problem_index"])
        budget = (o - 1) if k in infeasible else a.max_steps
        tm.mas.env.max_trials = budget
        no_fault = bool(o and o > budget)
        if chime is not None:
            chime.skip_update = a.cond == "oracle" and no_fault and phase == "train"
        reward, done = tm.mas.schedule(cfg)
        rec = {"phase": phase, "k": k, "task_index": cfg["task_index"], "task_main": task_main, "game": cfg.get("game_name"),
               "problem": cfg.get("problem_index"), "difficulty": cfg.get("difficulty"),
               "success": bool(done), "progress": float(reward), "secs": round(time.time() - t0, 1),
               "budget": budget, "optimal": o, "no_fault": no_fault and not done}
        if chime is not None:
            rec["chime"] = chime.finish(bool(done), task_id=str(cfg["task_index"]))
            with open(os.path.join(outdir, "memory.json"), "w") as f:
                f.write(chime.mem.to_json())
        rec["usage"] = usage_now()
        log.write(dumps(rec) + "\n")
        log.flush()
        results[phase].append(rec)
        print(f"[{phase} {k+1}/{len(tasks)}] {cfg.get('game_name')}#{cfg.get('problem_index')} "
              f"success={done} progress={reward:.2f} cost={rec['usage']['cost_cny_offpeak']:.3f} {rec['secs']:.0f}s",
              flush=True)
        if a.budget_cny and rec["usage"]["cost_cny_offpeak"] > a.budget_cny:
            print(f"BUDGET_STOP at {rec['usage']['cost_cny_offpeak']:.3f} CNY", flush=True)
            break

    def acc(rs, key):
        return round(100 * sum(r[key] for r in rs) / len(rs), 2) if rs else None

    summ = {"tag": tag, "mode": a.mode, "order_seed": a.order_seed, "n_train": len(results["train"]),
            "n_eval": len(results["eval"]),
            "train_success": acc(results["train"], "success"), "eval_success": acc(results["eval"], "success"),
            "train_progress": acc(results["train"], "progress"), "eval_progress": acc(results["eval"], "progress"),
            "memory_size": {r: len(b) for r, b in chime.mem.banks.items()} if chime else None,
            "usage": usage_now()}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
