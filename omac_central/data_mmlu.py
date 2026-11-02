"""从 MMLU（cais/mmlu，Hugging Face）抽取中心化团队实验的题目。

15 个学科都来自 OMAC 的 MMLU 学科列表，按团队中三个 worker 的专长分成三组：
  math（Mathematician）：abstract_algebra, college_mathematics, college_physics, high_school_statistics, machine_learning
  med（Doctor）：anatomy, clinical_knowledge, college_medicine, virology, high_school_biology
  law（Lawyer）：international_law, security_studies, us_foreign_policy, moral_scenarios, high_school_world_history
每个学科固定种子打乱后取 4 道训练题、12 道测试题：训练 60 题，测试 180 题。
（OMAC 自带的下采样 MMLU 只有训练 37 题、测试 31 题，测不出几个点的差异。）
用法：/home/claude/refs/gmemory/.venv/bin/python -I data_mmlu.py /home/claude/untrusted/mmlu
"""
import json
import os
import random
import sys

import pyarrow.parquet as pq

DOMAINS = {
    "math": ["abstract_algebra", "college_mathematics", "college_physics", "high_school_statistics", "machine_learning"],
    "med": ["anatomy", "clinical_knowledge", "college_medicine", "virology", "high_school_biology"],
    "law": ["international_law", "security_studies", "us_foreign_policy", "moral_scenarios", "high_school_world_history"],
}
N_TRAIN, N_TEST = 4, 12


def main(src):
    out = {"train": [], "test": []}
    for dom, subjects in DOMAINS.items():
        for s in subjects:
            rows = pq.read_table(os.path.join(src, f"{s}.parquet")).to_pylist()
            random.Random(f"mmlu-{s}").shuffle(rows)
            for split, part in (("train", rows[:N_TRAIN]), ("test", rows[N_TRAIN:N_TRAIN + N_TEST])):
                for i, r in enumerate(part):
                    out[split].append({"id": f"{s}-{split}-{i}", "subject": s, "domain": dom,
                                       "question": r["question"], "options": list(r["choices"]),
                                       "answer": "ABCD"[r["answer"]]})
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mmlu_central.json")
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in out.items()}, dst)


if __name__ == "__main__":
    main(sys.argv[1])
