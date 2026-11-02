"""冒烟测试：基线配置在测试集前若干题上跑一遍，检查解析、基线难度、每题成本。
用法：ALLOW_PEAK=1 python3 smoke.py [每类题数，默认 5]
（结果进缓存，主实验会直接复用）
"""
import collections
import json
import sqlite3
import sys
import time

import team

per = int(sys.argv[1]) if len(sys.argv) > 1 else 5
data = team.load_data()
items = [it for it in data["test"] if it["idx"] < per]
t0 = time.time()
recs = team.evaluate(items, team.base_config(), threads=16)
dt = time.time() - t0

out = {"n": len(recs), "team_acc": round(team.accuracy(recs), 3), "seconds": round(dt),
       "worker_acc": {w: round(sum(r["worker_correct"][w] for r in recs) / len(recs), 3) for w in team.WORKERS},
       "worker_acc_own_domain": {}, "route_sets": collections.Counter("+".join(r["selected"]) for r in recs),
       "route_fallback": sum(r["route_fallback"] for r in recs),
       "final_unparsed": sum(r["final"] is None for r in recs),
       "worker_unparsed": {w: sum(r["worker_ans"][w] is None for r in recs) for w in team.WORKERS},
       "by_domain": {}, "cost_cny_this_run": round(team.COST["cny"], 4), "api_calls": team.COST["calls"],
       "cached_calls": team.COST["cached"]}
for dom, w in (("math", "Mathematician"), ("med", "Doctor"), ("law", "Lawyer")):
    rs = [r for r in recs if r["domain"] == dom]
    out["by_domain"][dom] = {"team": round(team.accuracy(rs), 3),
                             **{x: round(sum(r["worker_correct"][x] for r in rs) / len(rs), 3) for x in team.WORKERS}}
    out["worker_acc_own_domain"][w] = out["by_domain"][dom][w]
con = sqlite3.connect(team._DB)
us = [json.loads(u) for (u,) in con.execute("SELECT usage FROM c")]
out["finish_length"] = sum(u.get("finish_reason") == "length" for u in us)
out["mean_tokens"] = {"prompt": round(sum(u.get("prompt_tokens", 0) for u in us) / len(us)),
                      "completion": round(sum(u.get("completion_tokens", 0) for u in us) / len(us))}
out["cost_per_item_offpeak_cny"] = round(sum(u["cny"] / (2 if u.get("peak") else 1) for u in us) / max(1, len(recs)), 4)
print(json.dumps(out, ensure_ascii=False, indent=1))
r = recs[0]
print("\n--- sample dispatch to Mathematician ---\n" + r["messages"]["Mathematician"][:1500])
print("\n--- sample route output ---\n" + r["route_out"][-600:])
json.dump(recs, open(f"runs/smoke/mmlu_pro_{len(recs)}.json", "w"), ensure_ascii=False, indent=1)
