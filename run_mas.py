"""多智能体 CHIME 实验：G-Memory 的 DyLAN（2×2，开启 critic）+ PDDL。

用法（经 gm.sh 在 G-Memory 目录运行）：
  /home/claude/chime/gm.sh /home/claude/chime/run_mas.py --mode team --order-seed 0
mode: empty | naive | root_only | team
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
from chime.mas_memory import ChimeMAS, dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["empty", "naive", "root_only", "team"])
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--budget-cny", type=float, default=0, help="累计花费超过此值即停止（0 表示不限）")
    ap.add_argument("--out", default="/home/claude/chime/mas_runs")
    a = ap.parse_args()

    tag = f"pddl_dylan_{a.mode}_s{a.order_seed}"
    outdir = os.path.join(a.out, tag)
    os.makedirs(outdir, exist_ok=True)
    R.WORKING_DIR = os.path.join("./.db", "chime_mas", tag)
    os.makedirs(R.WORKING_DIR, exist_ok=True)

    tm = R.build_task("pddl", "dylan", "empty", a.max_steps)
    tm.mas_config.update(use_critic=True, successful_topk=0, failed_topk=0, insights_topk=0, threshold=0.0,
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
    if a.mode != "empty":
        chime = ChimeMAS(a.mode, embed=qwen_embed, llm_call=llm_call, merger=merger)
        tm.mas.chime = chime

    tasks = list(tm.tasks)
    for i, t in enumerate(tasks):
        t["task_index"] = i
    random.Random(a.order_seed).shuffle(tasks)
    if a.limit:
        tasks = tasks[: a.limit]
    cut = int(round(len(tasks) * 0.7))

    log = open(os.path.join(outdir, "episodes.jsonl"), "w")
    results = {"train": [], "eval": []}
    for k, cfg in enumerate(tasks):
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
        reward, done = tm.mas.schedule(cfg)
        rec = {"phase": phase, "k": k, "task_index": cfg["task_index"], "game": cfg.get("game_name"),
               "problem": cfg.get("problem_index"), "difficulty": cfg.get("difficulty"),
               "success": bool(done), "progress": float(reward), "secs": round(time.time() - t0, 1)}
        if chime is not None:
            rec["chime"] = chime.finish(bool(done), task_id=str(cfg["task_index"]))
            with open(os.path.join(outdir, "memory.json"), "w") as f:
                f.write(chime.mem.to_json())
        rec["usage"] = get_usage()
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
            "usage": get_usage()}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
