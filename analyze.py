"""汇总 S6：3 种任务顺序 × 4 个变体。输出 results.json，并打印表格。

除了均值±标准差，还做按任务配对的比较：同一种顺序下，各变体的评估段是同一批任务，
因此可以逐题比较 CHIME 与对照组的胜/负/平，这比只看 15 题的成功率更稳。
"""
import collections
import json
import statistics as st
from pathlib import Path

VARIANTS = ["no_plan", "plan_only", "chime", "chime_nogate"]
SEEDS = [0, 1, 2]
ROOT = Path(__file__).resolve().parent / "runs"


def load(v, s):
    rows = [json.loads(l) for l in open(ROOT / f"airline_{v}_s{s}" / "episodes.jsonl")]
    return rows, json.load(open(ROOT / f"airline_{v}_s{s}" / "summary.json"))


def ms(xs):
    return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}"


out = {"per_variant": {}, "paired_eval_vs_chime": {}, "chime_gate": {}}
eval_by = {}
for v in VARIANTS:
    tr, ev, mem, pooled_ev = [], [], [], []
    for s in SEEDS:
        rows, summ = load(v, s)
        tr.append(summ["train_acc"]); ev.append(summ["eval_acc"])
        if summ["memory_size"] is not None:
            mem.append(summ["memory_size"])
        e = {r["task"]: r["reward"] >= 1 for r in rows if r["phase"] == "eval"}
        eval_by[(v, s)] = e
        pooled_ev += list(e.values())
    out["per_variant"][v] = {"train": ms(tr), "eval": ms(ev), "eval_pooled_45": round(100 * sum(pooled_ev) / len(pooled_ev), 1),
                             "memory": ms(mem) if mem else None, "train_by_seed": tr, "eval_by_seed": ev}

for v in VARIANTS:
    if v == "chime":
        continue
    w = l = t = 0
    for s in SEEDS:
        a, b = eval_by[("chime", s)], eval_by[(v, s)]
        for k in a:
            if a[k] and not b[k]:
                w += 1
            elif b[k] and not a[k]:
                l += 1
            else:
                t += 1
    out["paired_eval_vs_chime"][v] = {"chime_wins": w, "chime_losses": l, "ties": t}

# 归因门与记忆演化统计（CHIME 训练段）
labels = collections.Counter()
lab_by_outcome = collections.Counter()
conf, ins, mer, pru, dis, mis = [], 0, 0, 0, 0, 0
for s in SEEDS:
    rows, _ = load("chime", s)
    for r in rows:
        if r["phase"] != "train":
            continue
        g = r.get("gate") or {}
        lab = g.get("attribution", "none")
        labels[lab] += 1
        lab_by_outcome[(("success" if r["reward"] >= 1 else "failure"), lab)] += 1
        try:
            conf.append(float(g.get("confidence", 0)))
        except (TypeError, ValueError):
            pass
        u = r.get("update", {})
        ins += len(u.get("inserted", [])); mer += len(u.get("merged", [])); pru += len(u.get("pruned", []))
        dis += len(u.get("discarded", [])); mis += len(g.get("misleading_memory_ids") or [])
out["chime_gate"] = {"labels": dict(labels), "by_outcome": {f"{a}/{b}": c for (a, b), c in sorted(lab_by_outcome.items())},
                     "mean_confidence": round(st.mean(conf), 3), "inserted": ins, "merged": mer, "pruned": pru,
                     "discarded_low_conf": dis, "misleading_flags": mis}

(ROOT.parent / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
