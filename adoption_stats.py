"""3b 必要性检验（离线，不调用 API）：如果只有"输出被下游采纳"的角色才参与价值更新，会去掉多少次 +γ。

采纳的定义（从 DyLAN 团队轨迹直接读出，不依赖 LLM）：
  proposer   执行的动作与它某次提议相同
  finalizer  分歧步中执行的动作与它的输出相同，且不同于两个提议（真正改写了结果）
  checker    分歧步中它给出非 Support 的建议，且执行的动作不同于两个提议（建议后发生了改写）
信用：朴素多智能体 CHIME 在成功题中给每个自认 self 的角色检索到的记忆 +γ。
用法：python3 -I adoption_stats.py <run_dir> [<run_dir> ...]
"""
import collections
import json
import os
import re
import sys

ROLES = ("proposer", "checker", "finalizer")


def norm(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())


def adoption(steps):
    cnt = collections.Counter()
    for s in steps:
        act = norm(s["action"])
        props = [norm(e[2]) for e in s["entries"] if e[0] == "proposer"]
        debate = any(e[0] == "checker" for e in s["entries"])
        if act in props:
            cnt["proposer"] += 1
        if debate and act not in props:
            if any(norm(e[2]) == act for e in s["entries"] if e[0] == "finalizer"):
                cnt["finalizer"] += 1
            if any(not norm(e[2]).startswith("support") for e in s["entries"] if e[0] == "checker"):
                cnt["checker"] += 1
    return cnt


def analyze(run_dir):
    eps = [json.loads(l) for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    eps = [e for e in eps if e.get("phase", "train") == "train" and (e.get("chime") or {}).get("gates")]
    credits, removed = collections.Counter(), collections.Counter()
    blame, blame_no_part = collections.Counter(), collections.Counter()
    share = collections.defaultdict(list)
    for e in eps:
        c = e["chime"]
        ad = adoption(c["steps"])
        n = len(c["steps"])
        for r in ROLES:
            share[r].append(ad[r] / n if n else 0)
            claimed = str((c["gates"][r].get("raw") or {}).get("attribution", "")).lower() == "self"
            if not claimed:
                continue
            if e["success"]:
                credits[r] += 1
                removed[r] += ad[r] == 0
            else:
                blame[r] += 1
                blame_no_part[r] += ad[r] == 0
    return {"episodes": len(eps), "success_eps": sum(e["success"] for e in eps),
            "mean_adoption_share": {r: round(sum(v) / len(v), 3) for r, v in share.items()},
            "success_credits_by_role": dict(credits),
            "credits_removed_by_adoption_gate": dict(removed),
            "removed_pct": round(100 * sum(removed.values()) / max(1, sum(credits.values())), 1),
            "failure_lessons_by_role": dict(blame),
            "failure_lessons_from_roles_never_adopted": dict(blame_no_part)}


if __name__ == "__main__":
    out = {d: analyze(d) for d in sys.argv[1:]}
    json.dump(out, open("/home/claude/chime/mas_runs/adoption_stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
