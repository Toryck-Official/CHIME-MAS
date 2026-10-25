"""E2：植入一条误导记忆（PLANT_BAD）和一条无用记忆（PLANT_INERT），在 gripper 任务流里追踪它们的命运。

模式
  single_chime  单智能体 CHIME（DyLAN 1×1），植入规划库，CHIME 原版归因门
  naive         多智能体（DyLAN 2×2 + critic），植入提议者库，各角色各自归因
  central       多智能体，植入提议者库，一个归因门看完整团队轨迹（CHIME 原版归因门的推广）
全程训练（记忆持续更新），每题记录：植入记忆是否被检索、是否被标为误导、v / n、是否被剪枝、文本是否被合并改写，
以及本题新写入的记忆全文（用于统计补偿性记忆）。
用法：/home/claude/chime/gm.sh /home/claude/chime/run_e2.py --mode central --order-seed 0 --budget-cny 1.6
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
from chime.faults import planted_item
from chime.mas_memory import ChimeMAS, ChimeSingle, dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["single_chime", "naive", "central"])
    ap.add_argument("--game", default="gripper")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--budget-cny", type=float, default=1.6)
    ap.add_argument("--out", default="/home/claude/chime/mas_runs/e2")
    a = ap.parse_args()
    single = a.mode == "single_chime"
    tag = f"{a.game}_{a.mode}_s{a.order_seed}"
    outdir = os.path.join(a.out, tag)
    os.makedirs(outdir, exist_ok=True)
    R.WORKING_DIR = os.path.join("./.db", "chime_e2", tag)
    os.makedirs(R.WORKING_DIR, exist_ok=True)

    tm = R.build_task("pddl", "dylan", "empty", 30)
    if single:
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

    chime = (ChimeSingle("single_chime", embed=qwen_embed, llm_call=llm_call, merger=merger) if single
             else ChimeMAS(a.mode, embed=qwen_embed, llm_call=llm_call, merger=merger))
    tm.mas.chime = chime
    bank = "plan" if single else "proposer"

    tasks = [dict(t, task_index=i) for i, t in enumerate(tm.tasks)]
    tasks = [t for t in tasks if t.get("game_name") == a.game]
    random.Random(a.order_seed).shuffle(tasks)

    ep_path = os.path.join(outdir, "episodes.jsonl")
    done = [json.loads(l) for l in open(ep_path)] if os.path.exists(ep_path) else []
    base = dict(done[-1]["usage"]) if done else {}
    if done:
        chime.mem.load_json(open(os.path.join(outdir, "memory.json")).read())
    else:
        chime.mem.banks[bank] += [planted_item(a.game, bank, "bad"), planted_item(a.game, bank, "inert")]

    def usage_now():
        u = get_usage()
        return {k: round(u[k] + base.get(k, 0), 4) for k in u}

    def planted_state():
        out = {}
        for kind in ("BAD", "INERT"):
            m = chime.mem.get(f"PLANT_{kind}")
            out[kind] = None if m is None else {"v": round(m.value, 3), "n": m.n, "text": m.experience}
        return out

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
        if single:
            chime.context = task_description
        reward, ok = tm.mas.schedule(cfg)
        c = chime.finish(bool(ok), task_id=str(cfg["task_index"]))
        ins = []
        for u in ([c.get("update")] if "update" in c else [g["update"] for g in (c.get("gates") or {}).values()]):
            for x in (u or {}).get("inserted", []):
                m = chime.mem.get(x["id"])
                if m:
                    ins.append({"stage": m.stage, "id": m.id, "scenario": m.scenario, "experience": m.experience})
        rec = {"k": k, "task_index": cfg["task_index"], "problem": cfg.get("problem_index"), "task_main": task_main,
               "success": bool(ok), "progress": float(reward), "secs": round(time.time() - t0, 1),
               "chime": c, "planted": planted_state(), "inserted_text": ins, "usage": usage_now()}
        with open(os.path.join(outdir, "memory.json"), "w") as f:
            f.write(chime.mem.to_json())
        log.write(dumps(rec) + "\n")
        log.flush()
        print(f"[{a.mode} {k+1}/{len(tasks)}] #{cfg.get('problem_index')} success={ok} prog={reward:.2f} "
              f"planted={json.dumps({x: (y and [y['v'], y['n']]) for x, y in rec['planted'].items()})} "
              f"cost={rec['usage']['cost_cny_offpeak']:.3f}", flush=True)
        if rec["usage"]["cost_cny_offpeak"] > a.budget_cny:
            print("BUDGET_STOP", flush=True)
            break
    print("E2_DONE", flush=True)


if __name__ == "__main__":
    main()
