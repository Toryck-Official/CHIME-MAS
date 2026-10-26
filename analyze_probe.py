"""汇总三世界探针（run_probe.py）：本地效应、传播效应、重采样噪声与成本。
用法：python3 -I analyze_probe.py   输出 mas_runs/probe/probe_summary.json 并打印
"""
import collections
import json
import os
import re

P = "/home/claude/chime/mas_runs/probe"
SHARDS = [P, "/home/claude/chime/mas_runs/probe_b"]  # 两个并行分片


def n(t):
    return re.sub(r"\s+", " ", str(t or "").strip().lower()).rstrip(" .")


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    eps, rows, seen, shard_cost = [], [], set(), []
    for d in SHARDS:
        if not os.path.exists(os.path.join(d, "episodes.jsonl")):
            continue
        es = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl"))]
        done = {e["task_index"] for e in es}
        eps += es
        shard_cost.append(es[-1]["usage"]["cost_cny_offpeak"] if es else 0)
        for l in (open(os.path.join(d, "probes.jsonl")) if os.path.exists(os.path.join(d, "probes.jsonl")) else []):
            r = json.loads(l)
            key = (r["task_index"], r["step"], r["probe"], r.get("mem"))
            if r["task_index"] in done and key not in seen:  # 只保留已完成题目的探针，去重
                seen.add(key)
                rows.append(r)
    nulls = {(r["task_index"], r["step"]): r for r in rows if r["probe"] == "null"}
    out = {"episodes": len(eps), "success": sum(e["success"] for e in eps),
           "probed_steps": len(nulls), "debate_steps": sum(r["debate"] for r in nulls.values())}
    out["noise"] = {
        "proposer_regen_changed_pct": pct(sum(r["null_B_local"] for r in nulls.values()), len(nulls)),
        "downstream_resim_action_changed_pct": pct(sum(n(r.get("null_action")) != n(r["action"]) for r in nulls.values()),
                                                   len(nulls)),
        "critique_regen_changed_pct_debate": pct(sum(bool(r["null_crit_changed"]) for r in nulls.values() if r["debate"]),
                                                 out["debate_steps"]),
        "finalizer_regen_changed_pct_debate": pct(sum(bool(r["null_fin_changed"]) for r in nulls.values() if r["debate"]),
                                                  out["debate_steps"]),
    }
    roles = {}
    for role in ("proposer", "checker", "finalizer"):
        rs = [r for r in rows if r["probe"] == role]
        opp = [r for r in rs if r["opportunity"]]
        loc = [r for r in opp if r.get("local")]
        prop_a = [r for r in loc if n(r.get("cf_action")) != n(r["action"])]
        prop_null = [r for r in loc if n(r.get("cf_action")) != n(nulls[(r["task_index"], r["step"])].get("null_action"))]
        # 按"题 × 记忆"汇总：本题中是否在某一步有本地效应、是否在某一步传到动作
        mem = collections.defaultdict(lambda: {"local": False, "prop": False})
        for r in opp:
            k = (r["task_index"], r["mem"])
            mem[k]
            if r.get("local"):
                mem[k]["local"] = True
                if n(r.get("cf_action")) != n(r["action"]):
                    mem[k]["prop"] = True
        ml = [v for v in mem.values() if v["local"]]
        roles[role] = {
            "probes": len(rs), "with_opportunity": len(opp),
            "local_effect": len(loc), "local_effect_pct": pct(len(loc), len(opp)),
            "local_and_reached_action": len(prop_a),
            "local_but_not_reached_pct": pct(len(loc) - len(prop_a), len(loc)),
            "reached_action_vs_null_resim": len(prop_null),
            "task_memory_pairs": len(mem), "pairs_with_local_effect": len(ml),
            "pairs_local_never_reached_action": sum(not v["prop"] for v in ml),
            "local_by_step_type": {"consensus": sum(r.get("local", False) for r in opp if not r["debate"]),
                                   "debate": sum(r.get("local", False) for r in opp if r["debate"])},
        }
    out["roles"] = roles
    out["cost_cny"] = round(sum(shard_cost), 4)
    out["cost_per_probed_step_cny"] = round(out["cost_cny"] / max(1, len(nulls)), 4)
    json.dump(out, open(os.path.join(P, "probe_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
