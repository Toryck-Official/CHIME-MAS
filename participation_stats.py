"""离线统计（不调用 API）：多智能体朴素 CHIME 轨迹中，各角色的实际参与情况，以及按参与度归因与两种 LLM 归因的差异。

参与的定义来自 DyLAN 的结构，不依赖 LLM 判断：
  一致步：两个 proposer 输出相同 → 直接执行，只有 proposer 参与
  分歧步：proposer 不一致 → checker 审查、finalizer 给出动作，三方都参与
  finalizer 拍板：分歧步中最终执行的动作与两个提议都不同
按参与度归因（规则）：一个角色参与了至少一个"关键步"才算有份
  失败题的关键步 = 浪费步（执行的是 think 或无效动作）；成功题的关键步 = 推进步（非 think、非无效）
用法：python3 -I participation_stats.py mas_runs/cmp/pddl_dylan_naive_s0
"""
import collections
import json
import os
import re
import sys

ROLES = ("proposer", "checker", "finalizer")


def norm(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())


def wasted(step):
    return norm(step["action"]).startswith("think") or "not valid" in step["observation"].lower()


def analyze(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    rows = [r for r in rows if r["phase"] == "train"]
    cmp_path = os.path.join(run_dir, "attr_compare.jsonl")
    central = {json.loads(l)["k"]: json.loads(l) for l in open(cmp_path)} if os.path.exists(cmp_path) else {}

    agg = collections.Counter()
    by_game = collections.defaultdict(collections.Counter)
    per_ep = []
    for r in rows:
        steps = r["chime"]["steps"]
        part = {ro: 0 for ro in ROLES}
        key_part = {ro: 0 for ro in ROLES}
        n_key = 0
        for s in steps:
            roles_here = {e[0] for e in s["entries"]}
            debate = "checker" in roles_here
            props = [norm(e[2]) for e in s["entries"] if e[0] == "proposer"]
            override = debate and norm(s["action"]) not in props
            is_key = wasted(s) if not r["success"] else not wasted(s)
            n_key += is_key
            agg["steps"] += 1
            by_game[r["game"]]["steps"] += 1
            agg["debate_steps"] += debate
            by_game[r["game"]]["debate_steps"] += debate
            agg["override"] += override
            agg["wasted_steps"] += wasted(s)
            for ro in ROLES:
                if ro in roles_here:
                    part[ro] += 1
                    key_part[ro] += is_key
            for e in s["entries"]:
                if e[0] == "checker":
                    sup = norm(e[2]).startswith("support")
                    agg["checker_support"] += sup
                    agg["checker_suggest"] += not sup
                    if sup and wasted(s):
                        agg["checker_support_on_wasted_step"] += 1
            if wasted(s):
                agg["wasted_in_debate"] += debate
        rule = sorted(ro for ro in ROLES if key_part[ro] > 0)
        local = sorted(ro for ro, g in r["chime"]["gates"].items()
                       if str((g.get("raw") or {}).get("attribution", "")).lower() == "self")
        cen = central.get(r["k"], {}).get("central")
        per_ep.append({"k": r["k"], "game": r["game"], "success": r["success"], "n_steps": len(steps),
                       "participation_share": {ro: round(part[ro] / len(steps), 3) for ro in ROLES},
                       "key_steps": n_key, "key_participation": key_part,
                       "rule": rule, "local": local, "central": cen})

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    n = len(per_ep)
    summary = {
        "episodes": n,
        "steps": agg["steps"],
        "debate_step_pct": pct(agg["debate_steps"], agg["steps"]),
        "finalizer_override_pct_of_debate": pct(agg["override"], agg["debate_steps"]),
        "checker_support_pct": pct(agg["checker_support"], agg["checker_support"] + agg["checker_suggest"]),
        "checker_supports_on_wasted_step_pct": pct(agg["checker_support_on_wasted_step"], agg["checker_support"]),
        "wasted_step_pct": pct(agg["wasted_steps"], agg["steps"]),
        "wasted_steps_in_debate_pct": pct(agg["wasted_in_debate"], agg["wasted_steps"]),
        "debate_step_pct_by_game": {g: pct(c["debate_steps"], c["steps"]) for g, c in sorted(by_game.items())},
        "mean_participation_share": {ro: round(sum(e["participation_share"][ro] for e in per_ep) / n, 3) for ro in ROLES},
        "episodes_role_has_key_part": {ro: sum(ro in e["rule"] for e in per_ep) for ro in ROLES},
        "local_claims": {ro: sum(ro in e["local"] for e in per_ep) for ro in ROLES},
        "central_attributed": {ro: sum(ro in (e["central"] or []) for e in per_ep) for ro in ROLES},
        "agreement_pct": {
            "rule_vs_local": pct(sum(e["rule"] == e["local"] for e in per_ep), n),
            "rule_vs_central": pct(sum(e["rule"] == e["central"] for e in per_ep if e["central"] is not None), n),
            "local_vs_central": pct(sum(e["local"] == e["central"] for e in per_ep if e["central"] is not None), n),
        },
        "checker_low_participation_episodes": sum(e["participation_share"]["checker"] < 0.2 for e in per_ep),
    }
    # 检查者记忆的来源：写入时它在该题的参与比例
    mem = json.load(open(os.path.join(run_dir, "memory.json")))
    share_by_task = {str(r["task_index"]): e["participation_share"] for r, e in zip(rows, per_ep)}
    origins = [share_by_task.get(m["origin_task"], {}).get("checker") for m in mem.get("checker", [])]
    origins = [o for o in origins if o is not None]
    summary["checker_memories"] = {"n": len(mem.get("checker", [])),
                                   "written_from_episodes_with_participation_below_20pct": sum(o < 0.2 for o in origins),
                                   "median_participation_at_write": sorted(origins)[len(origins) // 2] if origins else None}
    with open(os.path.join(run_dir, "participation.json"), "w") as f:
        json.dump({"summary": summary, "episodes": per_ep}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    analyze(sys.argv[1])
