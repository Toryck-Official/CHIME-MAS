"""S3：CHIME 的"先规划后执行"agent，接入 τ²-bench（需要已安装 tau2）。

一次对话只规划一次（论文未描述重规划【补全】）：
  收到用户首条消息 → 用它作任务 x_t 检索 plan 记忆 → planner 产出计划 p_t
  → 用 (x_t, p_t) 检索 exec 记忆 → 写入 executor 的系统提示 → 后续轮次照常工具调用。
变体：
  no_plan   = τ² 原生 LLMAgent（论文的 No-Plan 基线）
  plan_only = 有 planner，无记忆
  chime     = 完整方法
"""
from __future__ import annotations

from typing import Optional

from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent, LLMAgentState
from tau2.data_model.message import SystemMessage, UserMessage

from .gate import format_memories
from .llm import chat
from .memory import HierarchicalMemory, MemoryItem

PLANNER_SYSTEM = """You are the PLANNER of a customer-service agent. You do not talk to the user and do not call tools.
Given the domain policy and the user's opening request, write a short numbered plan for the executor:
the subgoals in order, the facts that must be looked up before acting, the policy checks that gate each
irreversible action, and what to do if a check fails. Keep it under 12 lines. Do not invent ids or facts."""

PLANNER_USER = """<policy>
{policy}
</policy>

<user_request>
{task}
</user_request>

<planning_experience>
{memories}
</planning_experience>

Write the plan now. Use the planning experience only where its situation matches this request."""

EXECUTOR_EXTRA = """
<plan>
{plan}
</plan>
<execution_experience>
{memories}
</execution_experience>
Follow the plan unless new information from tools or the user shows it is wrong; the policy always wins.
Use the execution experience only where its situation matches."""


def _mem_lines(items: list[MemoryItem]) -> str:
    return format_memories(items) if items else "(none)"


class ChimeAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm: str, llm_args: Optional[dict] = None,
                 memory: HierarchicalMemory | None = None, use_plan: bool = True, planner_model: str = "deepseek-chat"):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args or {})
        self.memory = memory
        self.use_plan = use_plan
        self.planner_model = planner_model
        # 供任务结束后的归因门使用
        self.task_text = ""
        self.plan = "(no plan)"
        self.retrieved: list[MemoryItem] = []
        self.retrieval_log: dict = {}

    def _prepare(self, first_user: str, state: LLMAgentState) -> None:
        self.task_text = first_user
        plan_mem = [m for m, _, _ in self.memory.retrieve("plan", first_user)] if self.memory else []
        self.plan = chat([{"role": "system", "content": PLANNER_SYSTEM},
                          {"role": "user", "content": PLANNER_USER.format(policy=self.domain_policy, task=first_user,
                                                                          memories=_mem_lines(plan_mem))}],
                         model=self.planner_model)
        exec_mem = [m for m, _, _ in self.memory.retrieve("exec", first_user + "\n" + self.plan)] if self.memory else []
        self.retrieved = plan_mem + exec_mem
        self.retrieval_log = {"plan": [m.id for m in plan_mem], "exec": [m.id for m in exec_mem]}
        extra = EXECUTOR_EXTRA.format(plan=self.plan, memories=_mem_lines(exec_mem))
        state.system_messages = [SystemMessage(role="system", content=state.system_messages[0].content + "\n" + extra)]

    def generate_next_message(self, message, state):
        if self.use_plan and not self.task_text and isinstance(message, UserMessage) and message.content:
            self._prepare(str(message.content), state)
        return super().generate_next_message(message, state)


def render_trajectory(messages, max_tool_chars: int = 600) -> str:
    """把 τ² 的对话转成归因门可读的文本。"""
    lines = []
    for m in messages:
        role = getattr(m.role, "value", m.role)
        if role == "assistant":
            for tc in (m.tool_calls or []):
                lines.append(f"[agent tool call] {tc.name}({tc.arguments})")
            if m.content:
                lines.append(f"[agent] {m.content}")
        elif role == "user":
            for tc in (getattr(m, "tool_calls", None) or []):
                lines.append(f"[user tool call] {tc.name}({tc.arguments})")
            if m.content:
                lines.append(f"[user] {m.content}")
        elif role == "tool":
            c = (m.content or "")
            lines.append(f"[tool result{' ERROR' if getattr(m, 'error', False) else ''}] "
                         f"{c[:max_tool_chars]}{'…' if len(c) > max_tool_chars else ''}")
    return "\n".join(lines)


__all__ = ["ChimeAgent", "render_trajectory", "AGENT_INSTRUCTION"]
