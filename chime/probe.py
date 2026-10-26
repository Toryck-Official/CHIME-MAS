"""三世界反事实探针（借用 agent-specific effect 的构造），用于在 DyLAN 2×2 + critic 团队中测量
"记忆被本角色用上"与"记忆影响了团队实际执行的动作"之间的差距。

在被探测的步骤 t（事实动作已经算出、尚未交给环境执行时）：
  世界 A（事实）  已记录的提议者输出 O、检查者意见、定稿者输出、执行动作 a
  世界 B（干预）  去掉某条记忆，只重新生成持有它的角色这一层的输出
                   提议者记忆 → 两个提议者的输出 O'；检查者记忆 → 检查者意见；定稿者记忆 → 定稿者输出
                   本地效应：B 中该层输出与 A 不同
  世界 C（传播）  下游按 B 的输出重新反应，上游保持 A 的记录；得到动作 a'
                   传播效应：a' 与对照动作不同
对照（同一步、不去掉任何记忆）：
  null_B  用完整上下文重新生成提议者输出，与 A 比较，估计提议者层的重采样差异
  null_C  用 A 的提议者输出重新模拟下游，得到 a0；传播效应同时报告 a'≠a 与 a'≠a0
记忆在步骤 t 没有被用的机会（例如提议者一致、检查者与定稿者未被调用）时，记为 opportunity=False。
"""
from __future__ import annotations

import json
import re

from .mas_memory import ChimeMAS, dumps


def _n(t) -> str:
    """比较用的规范化：小写、合并空白、去掉末尾句点（环境对 "x." 与 "x" 解析为同一动作）。"""
    return re.sub(r"\s+", " ", str(t or "").strip().lower()).rstrip(" .")


class ProbeMAS(ChimeMAS):
    """朴素多智能体 CHIME，记忆冻结；在被选中的步骤上运行三世界探针，结果追加写入 out_path。"""

    def __init__(self, *a, probe_steps=(0, 3, 6, 9), out_path="", budget_check=None, **kw):
        super().__init__("naive", *a, **kw)
        self.frozen = True
        self.probe_steps = set(probe_steps)
        self.out_path = out_path
        self.budget_check = budget_check or (lambda: False)
        self.items: dict[str, list] = {}
        self.task_index = None

    def insights(self, role, query):
        items = [m for m, _, _ in self.mem.retrieve(role, query)]
        self.items[role] = items
        return super().insights(role, query)

    # ---------- 模拟 ----------
    @staticmethod
    def _proposers(dy, prompt):
        out = {}
        for h in range(dy._height):
            n = dy._neurons.get(0, h)
            out[n.id] = n._agent.response(prompt, n.reasoning_config)
        return out

    @staticmethod
    def _consensus(answers: list[str]) -> str | None:
        return answers[0] if len(answers) == 2 and answers[0] == answers[1] else None

    def _downstream(self, dy, prop_out, prompt_f, fin_ins, chk_ins, crit_override=None, fin_override=None):
        decision_user_prompt = self._decision_prompt
        """给定提议者输出，模拟本步剩余流程，返回 (原始动作, 检查者意见, 定稿者输出)。"""
        c0 = self._consensus([prop_out[dy._neurons.get(0, h).id] for h in range(dy._height)])
        if c0 is not None:
            return c0, {}, {}
        crits, fins = {}, {}
        for h in range(dy._height):
            n = dy._neurons.get(1, h)
            msg = ""
            for edge in n.in_edges:
                up = edge.from_neuron
                ans = prop_out[up.id]
                msg += (f"## Agent {up.id}, role is {up.role}, importance score is {up.importance}, "
                        f"output is:\n\n {ans}\n")
                if dy._use_critic:
                    key = (n.id, up.id)
                    if crit_override is not None and key in crit_override:
                        crit = crit_override[key]
                    else:
                        saved = n.critic_insights
                        n.critic_insights = chk_ins
                        crit = n._critic_upstream_agent(prompt_f, ans)
                        n.critic_insights = saved
                    crits[key] = crit
                    msg += f"## Critic's Suggestions for Improvement: {crit}\n\n"
            if fin_override is not None and n.id in fin_override:
                fins[n.id] = fin_override[n.id]
            else:
                fp = ("\n# Outputs from other agents in the current round. Carefully consider their opinions and "
                      "extract valuable insights, but maintain independent thinking and avoid being misled.\n"
                      + msg + "\n" + "-" * 20 + "\n" + prompt_f)
                fins[n.id] = n._agent.response(fp, n.reasoning_config)
            c1 = self._consensus(list(fins.values()))
            if c1 is not None:
                return c1, crits, fins
        ups = "".join(f"Agent{i+1}: {a}\n" for i, a in enumerate(fins.values()))
        up = decision_user_prompt.format(agents_responses=ups)
        if fin_ins:
            up = "## Lessons from past tasks\n" + "\n".join(f"- {r}" for r in fin_ins) + "\n\n" + up
        dn = dy._decision_neuron
        return dn._agent.response(up, dn.reasoning_config), crits, fins

    # ---------- 探针 ----------
    def probe_step(self, dy, env, step_i, final_action, few_shots, successful_shots, role_ins,
                   format_task_prompt_with_insights, decision_user_prompt):
        if step_i not in self.probe_steps or self.budget_check():
            return
        self._decision_prompt = decision_user_prompt

        def prompt(ins):
            return format_task_prompt_with_insights(few_shots=few_shots, memory_few_shots=successful_shots,
                                                    insights=ins, task_description=dy.meta_memory.summarize())
        P, F, C = role_ins["proposer"], role_ins["finalizer"], role_ins["checker"]
        pf = prompt(F)
        # 世界 A：事实记录
        A_prop = {dy._neurons.get(0, h).id: dy._neurons.get(0, h).cached_answer for h in range(dy._height)}
        A_fin = {dy._neurons.get(1, h).id: dy._neurons.get(1, h).cached_answer for h in range(dy._height)
                 if dy._neurons.get(1, h).cached_answer is not None}
        A_crit = {(dy._neurons.get(1, h).id, up): c for h in range(dy._height)
                  for up, c in dy._neurons.get(1, h).last_critiques} if A_fin else {}
        debate = bool(A_fin)
        a = _n(final_action)
        act = lambda raw: _n(env.process_action(raw))
        recs = []
        base = {"task_index": self.task_index, "step": step_i + 1, "debate": debate, "action": a}
        # 对照
        nb = self._proposers(dy, prompt(P))
        null_b_local = any(_n(nb[k]) != _n(A_prop[k]) for k in A_prop)
        a0_raw, n_crit, n_fin = self._downstream(dy, A_prop, pf, F, C)
        a0 = act(a0_raw)
        recs.append({**base, "probe": "null", "null_action": a0, "null_B_local": null_b_local, "null_C_changed": a0 != a,
                     "null_crit_changed": any(_n(n_crit.get(k)) != _n(v) for k, v in A_crit.items()) if debate else None,
                     "null_fin_changed": any(_n(n_fin.get(k)) != _n(v) for k, v in A_fin.items()) if debate else None})
        # 提议者记忆
        for idx, m in enumerate(self.items.get("proposer", [])):
            ins = [x for j, x in enumerate(P) if j != idx]
            ob = self._proposers(dy, prompt(ins))
            local = any(_n(ob[k]) != _n(A_prop[k]) for k in A_prop)
            r = {**base, "probe": "proposer", "mem": m.id, "opportunity": True, "local": local}
            if local:
                ac, _, _ = self._downstream(dy, ob, pf, F, C)
                r.update(prop_vs_a=act(ac) != a, prop_vs_null=act(ac) != a0, cf_action=act(ac),
                         consensus_after=self._consensus(list(ob.values())) is not None)
            recs.append(r)
        # 检查者记忆：只在分歧步有机会
        for idx, m in enumerate(self.items.get("checker", [])):
            r = {**base, "probe": "checker", "mem": m.id, "opportunity": debate}
            if debate:
                chk = [x for j, x in enumerate(C) if j != idx]
                ac, crits, _ = self._downstream(dy, A_prop, pf, F, chk)
                r["local"] = any(_n(crits.get(k)) != _n(v) for k, v in A_crit.items())
                if r["local"]:
                    r.update(prop_vs_a=act(ac) != a, prop_vs_null=act(ac) != a0, cf_action=act(ac))
            recs.append(r)
        # 定稿者记忆：只在分歧步有机会；检查者意见沿用事实记录
        for idx, m in enumerate(self.items.get("finalizer", [])):
            r = {**base, "probe": "finalizer", "mem": m.id, "opportunity": debate}
            if debate:
                fin_ins = [x for j, x in enumerate(F) if j != idx]
                ac, _, fins = self._downstream(dy, A_prop, prompt(fin_ins), fin_ins, C, crit_override=A_crit)
                r["local"] = any(_n(fins.get(k)) != _n(v) for k, v in A_fin.items() if k in fins) or \
                    (set(fins) != set(A_fin))
                if r["local"]:
                    r.update(prop_vs_a=act(ac) != a, prop_vs_null=act(ac) != a0, cf_action=act(ac))
            recs.append(r)
        with open(self.out_path, "a") as f:
            for r in recs:
                f.write(dumps(r) + "\n")
