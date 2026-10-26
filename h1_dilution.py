"""H1（离线，不调用 API）：CHIME 价值更新中的两类信用错配。

失败 × 有用（稀释）：CHIME 的 Eq.9 在 δ=0 时仍做在线平均（本实现 update_on_zero_delta=True，按公式执行）。
  一条当前价值为正的记忆，在一次失败中被检索、但没有被标为误导时，δ=0，价值被拉向 0。
  统计：失败题中被检索记忆的更新里，δ=0 的比例；落在正价值记忆上的次数与损失的价值；
  这些记忆之后是否又在成功题中拿到 +γ（CHIME 自己认为它有用）；
  结束时的实际价值与"只对非零 δ 求平均"（不稀释）的反事实价值之差。
  危害实验的原版运行中，无责失败已知（最优步数 > 步数上限），单独统计发生在无责失败上的稀释。
成功 × 没用（搭便车）：E2 植入的无用记忆 PLANT_INERT，每次成功被检索都拿到 +γ。
用法：python3 -I h1_dilution.py   输出 mas_runs/h1_dilution.json 并打印
"""
import collections
import json
import os

R = "/home/claude/chime/mas_runs"
RUNS = {
    "single": ["cmp/pddl_single_single_chime_s0", "harm/pddl_single_single_chime_orig_hint_s0",
               "harm/pddl_single_single_chime_orig_hint_s1", "e2/gripper_single_chime_s0", "e2/barman_single_chime_s0"],
    "mas_naive": ["cmp/pddl_dylan_naive_s0", "harm/pddl_dylan_naive_orig_hint_s0", "e2/gripper_naive_s0",
                  "e2/barman_naive_s0"],
    "mas_central": ["e2/gripper_central_s0", "e2/barman_central_s0"],
}


def updates(e):
    c = e.get("chime") or {}
    if "gates" in c:
        return [u for g in c["gates"].values() for u in ((g.get("update") or {}).get("value") or [])]
    return (c.get("update") or {}).get("value") or []


def analyze(run):
    eps = [json.loads(l) for l in open(os.path.join(R, run, "episodes.jsonl"))]
    eps = [e for e in eps if e.get("phase", "train") == "train"]
    hist = collections.defaultdict(list)  # id -> [(k, success, delta, v_old, v_new, no_fault)]
    for e in eps:
        for u in updates(e):
            hist[u["id"]].append((e["k"], e["success"], u["delta"], u["v"][0], u["v"][1], bool(e.get("no_fault")),
                                  u["n"]))
    fail_upd = [(i, h) for i, hs in hist.items() for h in hs if not h[1]]
    zero = [(i, h) for i, h in fail_upd if h[2] == 0]
    neg = [(i, h) for i, h in fail_upd if h[2] < 0]
    pos_zero = [(i, h) for i, h in zero if h[3] > 0]
    later_credit = sum(any(x[0] > h[0] and x[1] and x[2] > 0 for x in hist[i]) for i, h in pos_zero)
    lost = sum(h[3] - h[4] for _, h in pos_zero)
    nf_zero = [(i, h) for i, h in pos_zero if h[5]]
    # 结束时：实际价值 vs 只对非零 δ 求平均的反事实价值（只看最后仍有更新记录的记忆）
    # 对检索的影响：Eq.5 中价值只通过 1 + clip(α·v·n/(n+λ), -β, β) 进入重排得分（α=1, β=0.2, λ=3）
    gaps, big, mult = [], 0, []
    clip = lambda x: max(-0.2, min(0.2, x))
    for i, hs in hist.items():
        nz = [h[2] for h in hs if h[2] != 0]
        if not nz:
            continue
        cf = sum(nz) / len(nz)
        actual, n = hs[-1][4], hs[-1][6]
        if cf > 0:
            gaps.append(cf - actual)
            big += actual < 0.5 * cf
            mult.append(clip(cf * n / (n + 3)) - clip(actual * n / (n + 3)))
    return {"train_eps": len(eps), "fail_eps": sum(not e["success"] for e in eps),
            "updates_in_failures": len(fail_upd), "zero_delta_in_failures": len(zero), "flagged_in_failures": len(neg),
            "zero_delta_pct": round(100 * len(zero) / len(fail_upd), 1) if fail_upd else None,
            "dilutions_of_positive_memories": len(pos_zero),
            "of_which_later_credited_in_success": later_credit,
            "value_lost_total": round(lost, 2),
            "mean_loss_per_dilution": round(lost / len(pos_zero), 3) if pos_zero else None,
            "dilutions_in_no_fault_failures": len(nf_zero),
            "positive_memories_final": len(gaps),
            "mean_gap_cf_minus_actual": round(sum(gaps) / len(gaps), 3) if gaps else None,
            "below_half_of_cf_value": big,
            "mean_rerank_multiplier_diff": round(sum(mult) / len(mult), 3) if mult else None,
            "rerank_multiplier_diff_gt_0.05": sum(m > 0.05 for m in mult)}


def inert():
    out = {}
    for run in ("gripper_single_chime_s0", "gripper_central_s0", "barman_single_chime_s0", "barman_central_s0",
                "barman_naive_s0"):
        eps = [json.loads(l) for l in open(os.path.join(R, "e2", run, "episodes.jsonl"))]
        traj = [(e["k"], e["planted"]["INERT"]["v"] if e["planted"]["INERT"] else None) for e in eps]
        plus = sum(u["delta"] > 0 for e in eps for u in updates(e) if u["id"] == "PLANT_INERT")
        last = eps[-1]["planted"]["INERT"]
        n = last["n"] if last else 0
        out[run] = {"inert_plus_gamma_updates": plus, "final_v": traj[-1][1],
                    "max_v": max(v for _, v in traj if v is not None),
                    "final_rerank_multiplier": round(1 + max(-0.2, min(0.2, last["v"] * n / (n + 3))), 3) if last else None}
    return out


def main():
    res = {grp: {r: analyze(r) for r in runs} for grp, runs in RUNS.items()}
    res["success_x_useless_inert"] = inert()
    json.dump(res, open(os.path.join(R, "h1_dilution.json"), "w"), ensure_ascii=False, indent=1)
    for grp, d in res.items():
        print("==", grp)
        for r, s in d.items():
            print(" ", r, s)


if __name__ == "__main__":
    main()
