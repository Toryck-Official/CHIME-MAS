"""CHIME 记忆库：分层记忆（Eq.2/3）、分层检索（Eq.5）、信用感知更新（Eq.7/8/9 + 剪枝 + 合并/插入）。

与论文的对应关系写在每个函数的 docstring 里；论文没写清楚、由我们自行补全的地方标了【补全】。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

import numpy as np

STAGES = ("plan", "exec")


@dataclass
class Config:
    """Table 7 的超参。"""
    top_n: int = 20            # N：相似度粗筛条数
    top_k: int = 3             # K：重排后保留条数（每层）
    lam: float = 3.0           # λ：价值按复用次数收缩
    alpha: float = 1.0         # α：价值加成系数
    beta: float = 0.2          # β：价值加成截断
    theta_conf: float = 0.5    # 新经验的置信度门槛
    theta_merge: float = 0.88  # 场景描述相似度高于此值则合并
    n_min: int = 3             # 剪枝所需最少复用次数
    theta_prune: float = -0.1  # 价值低于此值则剪枝
    # 【补全】论文矛盾：Eq.8/9 对 δ=0 也做在线平均（价值被稀释、n 加一），正文却说"unchanged otherwise"。
    # True = 按公式字面执行；False = δ=0 时 v 和 n 都不动。作为消融项。
    update_on_zero_delta: bool = True


@dataclass
class MemoryItem:
    """Eq.3：m_i = (c_i, e_i, v_i, n_i)。"""
    stage: str
    scenario: str          # c_i：适用场景描述，检索键，也用于合并判断
    experience: str        # e_i：提炼出的经验
    value: float = 0.0     # v_i ∈ [-1, 1]
    n: int = 0             # n_i：复用次数
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    origin_task: str = ""  # 记录来源任务，便于追溯


@dataclass
class Attribution:
    """Eq.6：归因门输出 g_t = (a_t, {(c_new, e_new)}_ℓ, γ_t, M_mis)。"""
    label: str                                  # planning / execution / both / none
    new: dict[str, tuple[str, str]]             # stage -> (scenario, experience)
    confidence: float                           # γ_t ∈ [0, 1]
    misleading: list[str]                       # 被判为误导的已检索记忆 id


def stages_of(label: str) -> tuple[str, ...]:
    """Eq.7：L(a_t)。none 在论文中没有给出分支，按"不更新任何层"处理【补全】。"""
    return {"planning": ("plan",), "execution": ("exec",), "both": ("plan", "exec")}.get(label, ())


Embedder = Callable[[list[str]], np.ndarray]
Merger = Callable[[MemoryItem, str, str], tuple[str, str]]


def default_merger(old: MemoryItem, scenario: str, experience: str) -> tuple[str, str]:
    """合并方式论文未给出【补全】；默认保留旧场景、拼接经验。正式实验用 LLM 改写合并（见 agent 侧）。"""
    if experience.strip() and experience.strip() not in old.experience:
        return old.scenario, old.experience.rstrip() + "\n- " + experience.strip()
    return old.scenario, old.experience


class HierarchicalMemory:
    """Eq.2：M_t = (M_plan, M_exec)。"""

    def __init__(self, embed: Embedder, config: Config | None = None, merger: Merger = default_merger):
        self.embed = embed
        self.cfg = config or Config()
        self.merger = merger
        self.banks: dict[str, list[MemoryItem]] = {s: [] for s in STAGES}
        self._emb: dict[str, np.ndarray] = {}

    # ---------- 基础 ----------
    def _vec(self, item: MemoryItem) -> np.ndarray:
        if item.id not in self._emb:
            self._emb[item.id] = self.embed([item.scenario])[0]
        return self._emb[item.id]

    def _sims(self, stage: str, query: str) -> np.ndarray:
        bank = self.banks[stage]
        if not bank:
            return np.zeros(0)
        q = self.embed([query])[0]
        mat = np.stack([self._vec(m) for m in bank])
        return mat @ q  # 向量已归一化，点积即余弦

    def __len__(self) -> int:
        return sum(len(b) for b in self.banks.values())

    def get(self, mid: str) -> MemoryItem | None:
        for b in self.banks.values():
            for m in b:
                if m.id == mid:
                    return m
        return None

    # ---------- Eq.5 检索 ----------
    def retrieve(self, stage: str, query: str) -> list[tuple[MemoryItem, float, float]]:
        """返回 [(记忆, 相似度, 重排得分)]。
        1) 按 sim(q, c_i) 取 Top-N；
        2) 价值收缩 ṽ_i = v_i · n_i / (n_i + λ)；
        3) 得分 sim · [1 + clip(α ṽ_i, -β, β)]，取 Top-K。"""
        c = self.cfg
        sims = self._sims(stage, query)
        if sims.size == 0:
            return []
        cand = np.argsort(-sims)[: c.top_n]
        scored = []
        for i in cand:
            m = self.banks[stage][i]
            v_shrunk = m.value * m.n / (m.n + c.lam)
            score = sims[i] * (1 + float(np.clip(c.alpha * v_shrunk, -c.beta, c.beta)))
            scored.append((m, float(sims[i]), float(score)))
        scored.sort(key=lambda t: -t[2])
        return scored[: c.top_k]

    # ---------- Eq.7/8/9 + 剪枝 + 内容演化 ----------
    def update(self, g: Attribution, success: bool, retrieved: Iterable[MemoryItem], task_id: str = "") -> dict:
        """一次任务结束后的记忆更新，返回本次变更的日志。"""
        c = self.cfg
        L = stages_of(g.label)
        mis = set(g.misleading)
        log = {"value": [], "pruned": [], "merged": [], "inserted": [], "discarded": []}

        # Eq.8 + Eq.9：只对本轮被检索到的记忆做价值反馈
        seen = set()
        for m in retrieved:
            if m.id in seen or self.get(m.id) is None:
                continue
            seen.add(m.id)
            if m.id in mis:
                delta = -g.confidence
            elif success and m.stage in L:
                delta = g.confidence
            else:
                delta = 0.0
            if delta == 0.0 and not c.update_on_zero_delta:
                continue
            old = m.value
            m.value = float(np.clip((m.n * m.value + delta) / (m.n + 1), -1, 1))
            m.n += 1  # 每次被检索使用都计一次复用【补全：论文未说明 δ=0 时是否计数】
            log["value"].append({"id": m.id, "stage": m.stage, "delta": round(delta, 3), "v": [round(old, 3), round(m.value, 3)], "n": m.n})

        # 剪枝：n ≥ n_min 且 v < θ_prune
        for s in STAGES:
            keep = []
            for m in self.banks[s]:
                if m.n >= c.n_min and m.value < c.theta_prune:
                    log["pruned"].append(m.id)
                    self._emb.pop(m.id, None)
                else:
                    keep.append(m)
            self.banks[s] = keep

        # 内容演化：只写被归因的层
        for s in L:
            scen, exp = g.new.get(s, ("", ""))
            if g.confidence < c.theta_conf or not scen.strip() or not exp.strip():
                log["discarded"].append(s)
                continue
            sims = self._sims(s, scen)
            if sims.size and sims.max() > c.theta_merge:
                tgt = self.banks[s][int(sims.argmax())]
                tgt.scenario, tgt.experience = self.merger(tgt, scen, exp)
                self._emb.pop(tgt.id, None)
                log["merged"].append({"stage": s, "into": tgt.id, "sim": round(float(sims.max()), 3)})
            else:
                item = MemoryItem(stage=s, scenario=scen.strip(), experience=exp.strip(), origin_task=task_id)
                self.banks[s].append(item)
                log["inserted"].append({"stage": s, "id": item.id})
        return log

    # ---------- 持久化 ----------
    def to_json(self) -> str:
        return json.dumps({s: [asdict(m) for m in b] for s, b in self.banks.items()}, ensure_ascii=False, indent=1)

    def load_json(self, text: str) -> None:
        data = json.loads(text)
        self.banks = {s: [MemoryItem(**d) for d in data.get(s, [])] for s in STAGES}
        self._emb = {}
