"""S5/S6：在 τ² 一个域上跑 CHIME 任务流。

训练段（前 70%）顺序执行：每个任务结束 → 归因门 → 更新记忆；
评估段（后 30%）冻结记忆，只检索不更新（论文 Test-time protocol）。

用法（在装好 tau2 的环境里）：
  python run_stream.py --domain airline --variant chime --order-seed 0 --limit 4    # 冒烟
  python run_stream.py --domain airline --variant no_plan --order-seed 0
变体：no_plan | plan_only | chime | chime_nogate（归因门换成"失败写 both、成功写 both、γ=1"的结果驱动更新）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tau2.agent.llm_agent import LLMAgent
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, get_tasks, run_simulation

from chime.embed_qwen import qwen_embed
from chime.gate import run_gate
from chime.llm import USAGE, chat
from chime.memory import Attribution, Config, HierarchicalMemory
from chime.tau2_agent import ChimeAgent, render_trajectory

LLM = "deepseek/deepseek-chat"          # litellm 名称，供 executor 与用户模拟器
LLM_ARGS = {"temperature": 0.0}

# 评估侧模型：论文用 Qwen3.6-Flash，这里统一换成 DeepSeek（τ² 默认是 OpenAI 模型，本环境不可达）
import tau2.evaluator.evaluator_nl_assertions as _nl  # noqa: E402
from tau2.data_model.semantic_communication import CommunicateScoring, SemanticCommunicateSettings  # noqa: E402

_nl.DEFAULT_LLM_NL_ASSERTIONS = LLM
_nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(LLM_ARGS)
COMM_SCORING = CommunicateScoring(scorer="semantic", judge=SemanticCommunicateSettings(model=LLM, model_args=dict(LLM_ARGS)))


def llm_merger(old, scenario, experience):
    """合并：让 LLM 把两条经验改写成一条【补全】。"""
    out = chat([{"role": "user", "content":
                 "Merge two pieces of agent experience about the same situation into ONE concise, general, actionable "
                 "guideline (max 3 sentences). Remove duplicates; keep any distinct conditions.\n"
                 f"Situation: {old.scenario}\nA: {old.experience}\nB: {experience}\nReturn only the merged guideline."}])
    return old.scenario, out.strip() or old.experience


def run_one(domain, task, variant, memory):
    env = build_environment(domain)
    if variant == "no_plan":
        agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args=LLM_ARGS)
    else:
        agent = ChimeAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args=LLM_ARGS,
                           memory=memory if variant.startswith("chime") else None, use_plan=True)
    user = build_user("user_simulator", env, task, llm=LLM, llm_args=LLM_ARGS)
    orch = Orchestrator(domain=domain, agent=agent, user=user, environment=env, task=task,
                        max_steps=100, max_errors=10, seed=42)
    t0 = time.time()
    sim = run_simulation(orch, evaluation_type=EvaluationType.ALL, communicate_scoring=COMM_SCORING)
    reward = sim.reward_info.reward if sim.reward_info else 0.0
    return agent, sim, float(reward or 0.0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="airline")
    ap.add_argument("--variant", default="chime")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务（冒烟用）")
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    tasks = get_tasks(args.domain)
    random.Random(args.order_seed).shuffle(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    cut = int(round(len(tasks) * 0.7))
    train, test = tasks[:cut], tasks[cut:]

    outdir = Path(args.out) / f"{args.domain}_{args.variant}_s{args.order_seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    memory = HierarchicalMemory(qwen_embed, Config(), merger=llm_merger) if args.variant.startswith("chime") else None
    log = open(outdir / "episodes.jsonl", "w")

    def record(phase, task, agent, sim, reward, secs, extra=None):
        ri = sim.reward_info
        checks = {
            "db_match": getattr(getattr(ri, "db_check", None), "db_match", None) if ri else None,
            "action_checks": [{"action": c.action.name, "args": c.action.arguments, "matched": c.action_match}
                              for c in (getattr(ri, "action_checks", None) or [])] if ri else None,
            "gold_actions": [{"name": a.name, "args": a.arguments} for a in (task.evaluation_criteria.actions or [])],
        }
        row = {"phase": phase, "task": task.id, "reward": reward, "secs": round(secs, 1),
               "termination": str(sim.termination_reason), "n_messages": len(sim.messages),
               "plan": getattr(agent, "plan", None), "retrieved": getattr(agent, "retrieval_log", None),
               "eval_detail": checks, "trajectory": render_trajectory(sim.messages, max_tool_chars=2000)}
        row.update(extra or {})
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        return row

    # ---- 训练段：顺序，记忆演化 ----
    for i, task in enumerate(train):
        agent, sim, reward, secs = run_one(args.domain, task, args.variant, memory)
        extra = {}
        if memory is not None:
            success = reward >= 1.0
            traj = render_trajectory(sim.messages)
            if args.variant == "chime":
                g, raw = run_gate(agent.task_text, agent.plan, traj, success, reward, agent.retrieved)
                extra["gate"] = raw
            else:  # chime_nogate：结果驱动，不做阶段归因，用 LLM 只提炼经验
                g, raw = run_gate(agent.task_text, agent.plan, traj, success, reward, agent.retrieved)
                g = Attribution("both", g.new, 1.0, [])
                extra["gate"] = {"forced": "both", "raw": raw}
            extra["update"] = memory.update(g, success, agent.retrieved, task_id=task.id)
            extra["memory_size"] = len(memory)
            (outdir / "memory.json").write_text(memory.to_json())
        row = record("train", task, agent, sim, reward, secs, extra)
        print(f"[train {i+1}/{len(train)}] {task.id} reward={reward} mem={extra.get('memory_size')} "
              f"attr={extra.get('gate', {}).get('attribution') if isinstance(extra.get('gate'), dict) else ''} {secs:.0f}s", flush=True)

    # ---- 评估段：冻结记忆，并行 ----
    def eval_task(task):
        agent, sim, reward, secs = run_one(args.domain, task, args.variant, memory)
        return task, agent, sim, reward, secs

    test_rewards = []
    with ThreadPoolExecutor(args.eval_workers) as ex:
        for task, agent, sim, reward, secs in ex.map(eval_task, test):
            record("eval", task, agent, sim, reward, secs)
            test_rewards.append(reward)
            print(f"[eval] {task.id} reward={reward} {secs:.0f}s", flush=True)

    train_rewards = [json.loads(l)["reward"] for l in open(outdir / "episodes.jsonl") if json.loads(l)["phase"] == "train"]
    summary = {"domain": args.domain, "variant": args.variant, "order_seed": args.order_seed,
               "n_train": len(train), "n_eval": len(test),
               "train_acc": round(100 * sum(r >= 1 for r in train_rewards) / max(len(train_rewards), 1), 2),
               "eval_acc": round(100 * sum(r >= 1 for r in test_rewards) / max(len(test_rewards), 1), 2),
               "memory_size": len(memory) if memory else None, "usage_our_calls": USAGE}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
