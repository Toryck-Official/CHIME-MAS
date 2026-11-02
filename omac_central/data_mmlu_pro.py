"""从 MMLU-Pro（TIGER-Lab/MMLU-Pro，Hugging Face，test split）抽取中心化团队实验的题目。

MMLU 对 DeepSeek V4 Flash 已接近饱和（冒烟测试：团队 87%，三个 worker 90–93%，数学类 10/10），
worker 提示词几乎没有可提升的空间，因此换成同样按学科组织、10 个选项的更难版本 MMLU-Pro。
6 个类别按三个 worker 的专长分成三组：
  math（Mathematician）：math, physics
  med（Doctor）：health, biology
  law（Lawyer）：law, history
只用恰好 10 个选项的题。每类固定种子打乱后取 10 道训练题、60 道测试题（主实验先用每类前 30 道）。
用法：/home/claude/refs/gmemory/.venv/bin/python -I data_mmlu_pro.py /home/claude/untrusted/mmlu_pro/test.parquet
"""
import json
import os
import random
import sys

import pyarrow.parquet as pq

DOMAINS = {"math": ["math", "physics"], "med": ["health", "biology"], "law": ["law", "history"]}
N_TRAIN, N_TEST = 10, 60


def main(src):
    rows = pq.read_table(src).to_pylist()
    out = {"train": [], "test": []}
    for dom, cats in DOMAINS.items():
        for cat in cats:
            rs = sorted((r for r in rows if r["category"] == cat and len(r["options"]) == 10),
                        key=lambda r: r["question_id"])
            random.Random(f"mmlupro-{cat}").shuffle(rs)
            for split, part in (("train", rs[:N_TRAIN]), ("test", rs[N_TRAIN:N_TRAIN + N_TEST])):
                for i, r in enumerate(part):
                    out[split].append({"id": f"{cat}-{split}-{i}", "subject": cat, "domain": dom, "idx": i,
                                       "question_id": r["question_id"], "src": r["src"],
                                       "question": r["question"], "options": list(r["options"]),
                                       "answer": r["answer"]})
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mmlu_pro_central.json")
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in out.items()}, dst)


if __name__ == "__main__":
    main(sys.argv[1])
