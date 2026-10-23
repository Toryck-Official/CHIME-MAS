"""Qwen3-Embedding-0.6B（论文所用检索器），ONNX fp32 版本，CPU 推理，无需 PyTorch。

做法与官方一致：左填充、末 token 池化、L2 归一化。
查询与场景描述都不加 instruction 前缀（论文未说明【补全】），这样 sim(q, c) 与合并时的 sim(c, c') 同口径。
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

MODEL_DIR = os.environ.get("QWEN_EMB_DIR", "/home/claude/models/qwen3-emb")
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 2
    sess = ort.InferenceSession(os.path.join(MODEL_DIR, "onnx", "model.onnx"), opts, providers=["CPUExecutionProvider"])
    kv = [i for i in sess.get_inputs() if i.name.startswith("past_key_values")]
    return tok, sess, kv


_cache: dict[str, np.ndarray] = {}


def _embed_one(text: str) -> np.ndarray:
    tok, sess, kv = _load()
    ids = tok.encode(text).ids[-1024:]
    eot = tok.token_to_id("<|endoftext|>")
    if not ids or ids[-1] != eot:
        ids = ids + [eot]  # 末 token 池化取 <|endoftext|> 位置
    n = len(ids)
    feeds = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
    }
    for i in kv:
        feeds[i.name] = np.zeros((1, i.shape[1], 0, i.shape[3]), dtype=np.float32)
    h = sess.run(["last_hidden_state"], feeds)[0][0, -1]
    return (h / np.linalg.norm(h)).astype(np.float32)


def qwen_embed(texts: list[str]) -> np.ndarray:
    """逐条推理（单条无需填充，等价于左填充 + 末 token 池化），带缓存。"""
    out = []
    for t in texts:
        if t not in _cache:
            with _lock:
                if t not in _cache:
                    _cache[t] = _embed_one(t)
        out.append(_cache[t])
    return np.stack(out)
