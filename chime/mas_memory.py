"""CHIME 的多智能体扩展：每个角色一个记忆库 + 三种归因/更新策略。

角色（对应 DyLAN 的结构）：
  proposer  第一层 agent，提出下一步动作
  checker   critic，对上游提议放行（Support）或给建议
  finalizer 第二层 agent 与决策 agent，给出最终执行的动作

更新策略：
  naive     每个角色各自套用 CHIME 归因门，只看自己能看到的信息，只决定"要不要记在自己身上"
  root_only 团队级归因找出根因角色；只在失败时、只给根因角色写教训（DoCtOR 式），成功不更新
  team      团队级归因，按责任与学习机会分配（本文方法）：
              失败：root 写教训；missed_check（检查者放行了本可发现的错误）写验证类经验；
                    compounded（自身又引入新错）写教训；coordination（察觉但没纠正）写协调经验；
                    faithful（忠实执行、无从发现）不写
              成功：只有 decisive（关键贡献）的角色加分、写经验；incidental（搭便车）不加分
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .gate import format_memories
from .llm import parse_json
from .memory import Attribution, Config, HierarchicalMemory, MemoryItem

ROLES = ("proposer", "checker", "finalizer")
ROLE_DESC = {
    "proposer": "proposes the next action at each step",
    "checker": "reviews proposed actions and either supports them or gives a correction",
    "finalizer": "combines proposals and critiques and outputs the action that is actually executed",
}

WRITE_ON_FAILURE = {"root", "missed_check", "compounded", "coordination"}
CREDIT_ON_SUCCESS = {"decisive"}


@dataclass
class TraceStep:
    idx: int
    entries: list[tuple[str, str, str]] = field(default_factory=list)  # (role, agent_name, text)
    action: str = ""
    observation: str = ""


def _clip(t: str, n: int) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def render_team_trace(steps: list[TraceStep], max_steps: int = 40, per_msg: int = 160) -> str:
    out = []
    for s in steps[:max_steps]:
        out.append(f"--- step {s.idx} ---")
        for role, name, text in s.entries:
            out.append(f"[{role}:{name}] {_clip(text, per_msg)}")
        out.append(f"[executed] {_clip(s.action, per_msg)}")
        out.append(f"[observation] {_clip(s.observation, per_msg)}")
    if len(steps) > max_steps:
        out.append(f"... ({len(steps) - max_steps} more steps omitted)")
    return "\n".join(out)


def render_role_view(steps: list[TraceStep], role: str, max_steps: int = 40, per_msg: int = 160) -> str:
    """naive 归因门的局部视角：只给该角色自己的输出，加上执行的动作和观察。"""
    out = []
    for s in steps[:max_steps]:
        mine = [t for r, _, t in s.entries if r == role]
        out.append(f"--- step {s.idx} ---")
        for t in mine:
            out.append(f"[you] {_clip(t, per_msg)}")
        out.append(f"[executed] {_clip(s.action, per_msg)}")
        out.append(f"[observation] {_clip(s.observation, per_msg)}")
    return "\n".join(out)


NAIVE_GATE = """You are the {role} in a multi-agent team ({role_desc}). After an episode you decide whether
YOUR OWN decisions deserve the credit (success) or blame (failure), and distill a reusable lesson for yourself.

# Task
{task}

# What you saw and did, with the executed actions and observations
{view}

# Outcome
{outcome}

# Your memories retrieved for this episode
{memories}

Answer with ONE JSON object:
{{"attribution": "self" | "none", "reason": "<one sentence>",
  "scenario": "<when the lesson applies>", "experience": "<actionable guidance for you>",
  "confidence": <0..1>, "misleading_memory_ids": ["<ids of your memories that misled you>"]}}
Leave scenario/experience empty when attribution is "none". The "scenario" must describe a reusable task situation (e.g. "when a shaker must be emptied before reuse"), never this episode: no step numbers, no "the proposer did X". Never copy task-specific object names into the lesson."""

TEAM_GATE = """You are the credit attribution gate of a multi-agent team that solves planning tasks step by step:
- proposer: {proposer}
- checker: {checker}
- finalizer: {finalizer}
Review the whole episode and decide, for EACH role, its responsibility and whether it has something to learn.

# Task
{task}

# Team trace (per step: proposals, checker verdicts, final decisions, executed action, observation)
{trace}

# Outcome
{outcome}

# Memories retrieved for this episode (role-tagged)
{memories}

Verdicts:
- For a FAILED episode use: "root" (its decision started the failure), "missed_check" (a checker that supported a
  wrong action it could have caught from what it saw), "compounded" (added its own independent error),
  "coordination" (noticed the problem but the team did not correct it), "faithful" (followed upstream input and could
  not have detected the problem from what it saw), "none".
- For a SUCCESSFUL episode use: "decisive" (its decisions were necessary for the success) or "incidental"
  (the team would very likely have succeeded without its contribution).
Answer with ONE JSON object:
{{"reason": "<two sentences citing steps>",
  "roles": {{"proposer": {{"verdict": "...", "scenario": "...", "experience": "..."}},
             "checker": {{"verdict": "...", "scenario": "...", "experience": "..."}},
             "finalizer": {{"verdict": "...", "scenario": "...", "experience": "..."}}}},
  "confidence": <0..1>, "misleading_memory_ids": ["..."]}}
Write scenario/experience only for roles that should learn (root, missed_check, compounded, coordination, decisive);
a checker's lesson must be about what to verify. The "scenario" must describe a reusable task situation (e.g. "when a shaker must be emptied before reuse"), never this episode: no step numbers, no "the proposer did X". Never copy task-specific object names into lessons."""


class ChimeMAS:
    """与框架无关的核心：记忆、轨迹记录、归因与更新。G-Memory 适配见 mas/memory/mas_memory/chime_mas.py。"""

    def __init__(self, mode: str, embed, llm_call, merger=None, config: Config | None = None):
        assert mode in ("naive", "root_only", "team")
        self.mode = mode
        self.llm_call = llm_call  # (messages: list[dict]) -> str
        kw = {"merger": merger} if merger else {}
        self.mem = HierarchicalMemory(embed, config or Config(), stages=ROLES, **kw)
        self.frozen = False
        self.reset("")
        self.log: list[dict] = []

    # ---- 每个任务 ----
    def reset(self, task: str):
        self.task = task
        self.steps: list[TraceStep] = [TraceStep(1)]
        self.retrieved: dict[str, list[MemoryItem]] = {r: [] for r in ROLES}

    def insights(self, role: str, query: str) -> list[str]:
        items = [m for m, _, _ in self.mem.retrieve(role, query)]
        seen = {m.id for m in self.retrieved[role]}
        self.retrieved[role] += [m for m in items if m.id not in seen]
        return [f"When {m.scenario}: {m.experience}" for m in items]

    def record(self, role: str, agent: str, text: str):
        if text:
            self.steps[-1].entries.append((role, agent, str(text)))

    def step_done(self, action: str, observation: str):
        self.steps[-1].action, self.steps[-1].observation = str(action), str(observation)
        self.steps.append(TraceStep(len(self.steps) + 1))

    # ---- 任务结束 ----
    def finish(self, success: bool, task_id: str = "") -> dict:
        if self.steps and not self.steps[-1].entries and not self.steps[-1].action:
            self.steps.pop()
        rec = {"task_id": task_id, "success": success, "mode": self.mode, "frozen": self.frozen,
               "retrieved": {r: [m.id for m in v] for r, v in self.retrieved.items()}}
        if self.frozen:
            self.log.append(rec)
            return rec
        outcome = "SUCCESS (all goal conditions satisfied)" if success else "FAILURE (goal not reached within the step limit)"
        if self.mode == "naive":
            rec["gates"] = {}
            for role in ROLES:
                prompt = NAIVE_GATE.format(role=role, role_desc=ROLE_DESC[role], task=self.task,
                                           view=render_role_view(self.steps, role), outcome=outcome,
                                           memories=format_memories(self.retrieved[role]))
                d = self._ask(prompt)
                conf = _num(d.get("confidence"))
                mis = [i for i in (d.get("misleading_memory_ids") or []) if i in {m.id for m in self.retrieved[role]}]
                is_self = str(d.get("attribution", "none")).lower() == "self"
                g = Attribution("none", {role: (str(d.get("scenario", "")), str(d.get("experience", "")))}, conf, mis)
                upd = self.mem.update(g, success, self.retrieved[role], task_id, targets=(role,) if is_self else ())
                rec["gates"][role] = {"raw": d, "update": upd}
        else:
            all_ret = [m for r in ROLES for m in self.retrieved[r]]
            prompt = TEAM_GATE.format(task=self.task, trace=render_team_trace(self.steps), outcome=outcome,
                                      memories=format_memories(all_ret), **ROLE_DESC)
            d = self._ask(prompt)
            roles = d.get("roles") or {}
            verdict = {r: str((roles.get(r) or {}).get("verdict", "none")).lower() for r in ROLES}
            new = {r: (str((roles.get(r) or {}).get("scenario", "")), str((roles.get(r) or {}).get("experience", "")))
                   for r in ROLES}
            if self.mode == "team":
                targets = tuple(r for r in ROLES if verdict[r] in (CREDIT_ON_SUCCESS if success else WRITE_ON_FAILURE))
            else:  # root_only：只在失败时写根因角色
                targets = () if success else tuple(r for r in ROLES if verdict[r] == "root")
            conf = _num(d.get("confidence"))
            mis = [i for i in (d.get("misleading_memory_ids") or []) if i in {m.id for m in all_ret}]
            g = Attribution("none", new, conf, mis)
            if self.mode == "root_only" and success:
                upd = {"skipped": "root_only does not update on success"}
            else:
                upd = self.mem.update(g, success, all_ret, task_id, targets=targets)
            rec.update(gate=d, verdict=verdict, targets=list(targets), update=upd)
        rec["memory_size"] = {r: len(self.mem.banks[r]) for r in ROLES}
        self.log.append(rec)
        return rec

    def _ask(self, prompt: str) -> dict:
        raw = self.llm_call([{"role": "user", "content": prompt}])
        try:
            return parse_json(raw)
        except Exception:
            return {"_unparsed": raw[:500]}


def _num(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
