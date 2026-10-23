"""S1/S2 单元测试：逐条核对论文公式。运行：python3 -m unittest discover -s tests"""
import unittest

import numpy as np

from chime.embed import hash_embed
from chime.memory import Attribution, Config, HierarchicalMemory, MemoryItem, stages_of


def fixed_embed(table):
    """测试用：文本 -> 指定方向的单位向量，便于精确控制相似度。"""
    def f(texts):
        out = []
        for t in texts:
            v = np.array(table[t], dtype=float)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)
    return f


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        # 查询 q 与 a、b 的余弦分别为 0.9 和 0.85
        t = {"q": [1, 0], "a": [0.9, np.sqrt(1 - 0.81)], "b": [0.85, np.sqrt(1 - 0.85 ** 2)]}
        self.mem = HierarchicalMemory(fixed_embed(t), Config(top_k=1))
        self.a = MemoryItem("plan", "a", "ea")
        self.b = MemoryItem("plan", "b", "eb")
        self.mem.banks["plan"] = [self.a, self.b]

    def test_similarity_dominates_when_no_value(self):
        top = self.mem.retrieve("plan", "q")[0]
        self.assertEqual(top[0].id, self.a.id)
        self.assertAlmostEqual(top[1], 0.9, places=6)
        self.assertAlmostEqual(top[2], 0.9, places=6)

    def test_value_shrinkage_and_clip(self):
        # b 价值高且复用多：ṽ = 1·30/33 ≈ 0.909，α·ṽ 被截到 β=0.2 → 0.85·1.2 = 1.02 > 0.9
        self.b.value, self.b.n = 1.0, 30
        top = self.mem.retrieve("plan", "q")[0]
        self.assertEqual(top[0].id, self.b.id)
        self.assertAlmostEqual(top[2], 0.85 * 1.2, places=6)

    def test_rarely_used_memory_shrinks_to_similarity(self):
        # n=1 时 ṽ = 1·1/4 = 0.25，截到 0.2；n=0 时 ṽ=0，退化为纯相似度
        self.b.value, self.b.n = 1.0, 0
        top = self.mem.retrieve("plan", "q")[0]
        self.assertEqual(top[0].id, self.a.id)

    def test_negative_value_demotes(self):
        self.a.value, self.a.n = -1.0, 30
        top = self.mem.retrieve("plan", "q")[0]
        self.assertEqual(top[0].id, self.b.id)  # 0.9·0.8=0.72 < 0.85


class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.mem = HierarchicalMemory(hash_embed)
        self.p = MemoryItem("plan", "booking flight change", "check fare rules first", value=0.5, n=1)
        self.x = MemoryItem("exec", "call update tool", "pass reservation id", value=0.0, n=0)
        self.mem.banks["plan"].append(self.p)
        self.mem.banks["exec"].append(self.x)

    def test_routing(self):
        self.assertEqual(stages_of("planning"), ("plan",))
        self.assertEqual(stages_of("execution"), ("exec",))
        self.assertEqual(stages_of("both"), ("plan", "exec"))
        self.assertEqual(stages_of("none"), ())

    def test_success_rewards_only_attributed_stage(self):
        g = Attribution("planning", {}, 0.8, [])
        self.mem.update(g, True, [self.p, self.x])
        self.assertAlmostEqual(self.p.value, (1 * 0.5 + 0.8) / 2)   # Eq.9 在线平均
        self.assertEqual(self.p.n, 2)
        self.assertAlmostEqual(self.x.value, 0.0)                    # 未被归因的层 δ=0
        self.assertEqual(self.x.n, 1)

    def test_misleading_penalized_even_on_success(self):
        g = Attribution("both", {}, 0.6, [self.x.id])
        self.mem.update(g, True, [self.p, self.x])
        self.assertAlmostEqual(self.x.value, -0.6)
        self.assertAlmostEqual(self.p.value, (0.5 + 0.6) / 2)

    def test_failure_unflagged_literal_eq9_dilutes(self):
        g = Attribution("execution", {}, 0.9, [])
        self.mem.update(g, False, [self.p, self.x])
        self.assertAlmostEqual(self.p.value, 0.5 / 2)  # δ=0 但 n 增加，均值被稀释
        self.assertEqual(self.p.n, 2)
        self.assertAlmostEqual(self.x.value, 0.0)

    def test_failure_unflagged_unchanged_when_switch_off(self):
        self.mem.cfg.update_on_zero_delta = False
        g = Attribution("execution", {}, 0.9, [])
        self.mem.update(g, False, [self.p, self.x])
        self.assertAlmostEqual(self.p.value, 0.5)
        self.assertEqual(self.p.n, 1)

    def test_prune(self):
        self.x.value, self.x.n = -0.05, 2
        g = Attribution("none", {}, 0.9, [self.x.id])
        log = self.mem.update(g, False, [self.x])   # v = (2·-0.05 - 0.9)/3 ≈ -0.333, n=3
        self.assertIn(self.x.id, log["pruned"])
        self.assertIsNone(self.mem.get(self.x.id))

    def test_insert_merge_discard(self):
        g = Attribution("planning", {"plan": ("cancel reservation with travel insurance", "verify insurance before refund")}, 0.7, [])
        log = self.mem.update(g, False, [])
        self.assertEqual(len(log["inserted"]), 1)
        # 同一场景再来一次 → 合并而不是新增
        g2 = Attribution("planning", {"plan": ("cancel reservation with travel insurance", "also check cabin class")}, 0.7, [])
        log2 = self.mem.update(g2, False, [])
        self.assertEqual(len(log2["merged"]), 1)
        self.assertEqual(len(self.mem.banks["plan"]), 2)
        # 低置信度丢弃
        g3 = Attribution("planning", {"plan": ("totally different case", "x")}, 0.3, [])
        log3 = self.mem.update(g3, False, [])
        self.assertEqual(log3["discarded"], ["plan"])

    def test_none_label_writes_nothing(self):
        g = Attribution("none", {"plan": ("s", "e"), "exec": ("s", "e")}, 0.9, [])
        log = self.mem.update(g, False, [])
        self.assertEqual(log["inserted"], [])

    def test_roundtrip(self):
        txt = self.mem.to_json()
        m2 = HierarchicalMemory(hash_embed)
        m2.load_json(txt)
        self.assertEqual(len(m2), 2)


if __name__ == "__main__":
    unittest.main()
