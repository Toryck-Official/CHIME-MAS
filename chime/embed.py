"""检索用的文本向量。

论文用 Qwen3-Embedding-0.6B。下载模型权重需要另行放行，先用一个无依赖的替代：
词 + 字符 n-gram 的哈希 TF-IDF 风格向量，L2 归一化。它对中英文都能工作，但相似度的数值
分布和 Qwen 向量不同，θ_merge=0.88 这类阈值需要重新标定（见 calibrate 记录）。
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

DIM = 4096
_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]")


def _feats(text: str) -> list[str]:
    t = text.lower()
    words = _TOKEN.findall(t)
    feats = list(words)
    feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    compact = re.sub(r"\s+", " ", t)
    feats += [compact[i:i + 4] for i in range(max(len(compact) - 3, 0))]
    return feats


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def hash_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for r, text in enumerate(texts):
        counts: dict[int, float] = {}
        for f in _feats(text):
            k = _h(f)
            idx, sign = k % DIM, 1.0 if (k >> 20) & 1 else -1.0
            counts[idx] = counts.get(idx, 0.0) + sign
        for idx, v in counts.items():
            out[r, idx] = math.copysign(1 + math.log(abs(v)), v) if v else 0.0
        norm = np.linalg.norm(out[r])
        if norm:
            out[r] /= norm
    return out
