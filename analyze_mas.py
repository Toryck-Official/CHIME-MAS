"""汇总多智能体 pilot：成功率 / 进度分（训练段、评估段）、各角色记忆条数、花费，以及信用分配的广度。

信用广度（回答 Q1：朴素做法是否"广撒网"）：
  成功时被记功的角色数：naive = 自认 self 的角色数；team = 判为 decisive 的角色数；root_only 成功不更新
  失败时写教训的角色数：naive = 自认 self 的角色数；team / root_only = targets 数
  team 的失败判决分布：root / missed_check / compounded / coordination / faithful / none
"""
import collections
import glob
import json
import os
import statistics as st

OUT = "/home/claude/chime/mas_runs"
MODES = ["empty", "naive", "root_only", "team"]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"


rows = {}
for m in MODES:
    for f in sorted(glob.glob(f"{OUT}/pddl_dylan_{m}_s*/episodes.jsonl")):
        seed = int(f.split("_s")[-1].split("/")[0])
        rows[(m, seed)] = [json.loads(l) for l in open(f)]

out = {"per_mode": {}, "credit": {}, "team_verdicts": {}}
for m in MODES:
    seeds = sorted(s for (mm, s) in rows if mm == m)
    if not seeds:
        continue
    tr_s, ev_s, tr_p, ev_p, cost, n = [], [], [], [], [], []
    for s in seeds:
        r = rows[(m, s)]
        tr = [x for x in r if x["phase"] == "train"]
        ev = [x for x in r if x["phase"] == "eval"]
        tr_s.append(100 * sum(x["success"] for x in tr) / len(tr) if tr else None)
        ev_s.append(100 * sum(x["success"] for x in ev) / len(ev) if ev else None)
        tr_p.append(100 * st.mean(x["progress"] for x in tr) if tr else None)
        ev_p.append(100 * st.mean(x["progress"] for x in ev) if ev else None)
        cost.append(r[-1]["usage"]["cost_cny_offpeak"])
        n.append((len(tr), len(ev)))
    mem = []
    for s in seeds:
        last = [x for x in rows[(m, s)] if "chime" in x and "memory_size" in x["chime"]]
        if last:
            mem.append(last[-1]["chime"]["memory_size"])
    out["per_mode"][m] = {"seeds": seeds, "n_train_eval": n,
                          "train_success": ms(tr_s), "eval_success": ms(ev_s),
                          "train_progress": ms(tr_p), "eval_progress": ms(ev_p),
                          "cost_cny": [round(c, 2) for c in cost], "memory_last": mem}

    if m == "empty":
        continue
    succ_credit, fail_write = [], []
    verdicts = collections.Counter()
    for s in seeds:
        for x in rows[(m, s)]:
            c = x.get("chime") or {}
            if x["phase"] != "train" or c.get("frozen"):
                continue
            if m == "naive":
                k = sum(1 for g in (c.get("gates") or {}).values()
                        if str((g.get("raw") or {}).get("attribution", "")).lower() == "self")
            else:
                k = len(c.get("targets") or [])
                for role, v in (c.get("verdict") or {}).items():
                    verdicts[("success" if x["success"] else "failure", role, v)] += 1
            (succ_credit if x["success"] else fail_write).append(k)
    out["credit"][m] = {
        "mean_roles_credited_per_success": round(st.mean(succ_credit), 2) if succ_credit else None,
        "n_success": len(succ_credit),
        "mean_roles_written_per_failure": round(st.mean(fail_write), 2) if fail_write else None,
        "n_failure": len(fail_write)}
    if verdicts:
        out["team_verdicts"][m] = {f"{a}/{b}/{c}": v for (a, b, c), v in sorted(verdicts.items())}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
