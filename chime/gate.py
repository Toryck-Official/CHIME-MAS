"""Eq.6 信用归因门：同一冻结骨干做结构化自反思。

论文只说"review the task, plan, trajectory, outcome, and retrieved memories"，未公开 prompt。
以下 prompt 为我们按论文描述补全【补全】，字段与 Eq.6 一一对应。
"""
from __future__ import annotations

from .llm import chat, parse_json
from .memory import Attribution, MemoryItem

GATE_SYSTEM = """You are the credit attribution gate of a self-evolving agent that works in two stages:
a PLANNER writes a plan for the task, then an EXECUTOR carries it out with tools.
After an episode you decide which stage deserves the credit (success) or blame (failure),
and you distill reusable experience ONLY for that stage. Be strict: do not write memory
when the outcome is explained by external factors, missing evidence, or luck."""

GATE_USER = """# Task
{task}

# Plan produced by the planner
{plan}

# Trajectory (executor actions, tool results, user turns)
{trajectory}

# Outcome
{outcome}

# Memories retrieved for this episode
{memories}

Answer with ONE JSON object and nothing else:
{{
  "attribution": "planning" | "execution" | "both" | "none",
  "reason": "<one or two sentences citing concrete steps>",
  "planning_experience": {{"scenario": "<when this applies, phrased as a reusable task situation>", "experience": "<actionable guidance for planning>"}},
  "execution_experience": {{"scenario": "<when this applies>", "experience": "<actionable guidance for tool use / execution>"}},
  "confidence": <number between 0 and 1>,
  "misleading_memory_ids": ["<ids of retrieved memories that pushed the agent the wrong way>"]
}}
Rules:
- For a FAILED episode, attribute to the stage whose decision made the failure happen.
  "planning" = wrong decomposition, missing constraint, wrong dependency or policy reasoning in the plan.
  "execution" = the plan was adequate but tools were misused, arguments wrong, steps skipped or done out of order.
  "none" = caused by the user simulator, environment, ambiguous task, or not enough evidence.
- For a SUCCESSFUL episode, attribute to the stage that made the success possible.
- Leave the experience of a non-attributed stage as empty strings.
- Experiences must be general enough to transfer to other tasks of the same domain; never copy task-specific ids, names or numbers.
- Only list a memory id as misleading if you can point to the step where following it caused harm."""


def format_memories(items: list[MemoryItem]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"[{m.id}] ({m.stage}) when: {m.scenario} -> {m.experience}" for m in items)


def run_gate(task: str, plan: str, trajectory: str, success: bool, reward: float,
             retrieved: list[MemoryItem], model: str = "deepseek-chat") -> tuple[Attribution, dict]:
    outcome = f"{'SUCCESS' if success else 'FAILURE'} (reward={reward})"
    msg = [{"role": "system", "content": GATE_SYSTEM},
           {"role": "user", "content": GATE_USER.format(task=task, plan=plan, trajectory=trajectory,
                                                         outcome=outcome, memories=format_memories(retrieved))}]
    raw = chat(msg, model=model, json_mode=True)
    d = parse_json(raw)
    label = str(d.get("attribution", "none")).strip().lower()
    if label not in ("planning", "execution", "both", "none"):
        label = "none"
    def pair(k):
        x = d.get(k) or {}
        return (str(x.get("scenario", "")).strip(), str(x.get("experience", "")).strip())
    try:
        conf = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    valid_ids = {m.id for m in retrieved}
    mis = [i for i in d.get("misleading_memory_ids", []) or [] if i in valid_ids]
    g = Attribution(label=label, new={"plan": pair("planning_experience"), "exec": pair("execution_experience")},
                    confidence=max(0.0, min(1.0, conf)), misleading=mis)
    return g, d
