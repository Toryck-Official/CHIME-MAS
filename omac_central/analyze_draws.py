"""worker 层多次独立采样的汇总（run_round1.py 的 wcell 单元），以及 Lawyer 线的团队格。

每次采样：同一 master 下重新生成分派给该 worker 的消息，原提示词与优化后提示词对同一条消息各答一次。
输出 runs/round1/draws.json，并打印 Markdown 表格。
用法：python3 analyze_draws.py [runs/round1]
"""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round1")
OUT = {}


def draws(role):
    pat = "wtest_{}_r*.jsonl" if role == "Mathematician" else f"wtest_{role}_" + "{}_r*.jsonl"
    rows = []
    for master in ("old", "disp"):
        for p in sorted(glob.glob(os.path.join(D, pat.format(master)))):
            recs = [json.loads(l) for l in open(p)]
            n = len(recs)
            co, cn = sum(r["correct_old"] for r in recs), sum(r["correct_new"] for r in recs)
            rows.append({"master": master, "draw": re.search(r"_(r\d+)\.jsonl$", p).group(1), "n": n,
                         "acc_old": 100 * co / n, "acc_new": 100 * cn / n, "effect": 100 * (cn - co) / n,
                         "chars_old": st.mean(len(r["resp_old"]) for r in recs),
                         "chars_new": st.mean(len(r["resp_new"]) for r in recs),
                         "unparsed_old": sum(r["ans_old"] is None for r in recs),
                         "unparsed_new": sum(r["ans_new"] is None for r in recs),
                         "msg_chars": st.mean(len(r["message"]) for r in recs)})
    return rows


def summarize(rows):
    s = {}
    for m in ("old", "disp"):
        rs = [r for r in rows if r["master"] == m]
        if rs:
            s[m] = {"draws": len(rs), "effect_mean": st.mean(r["effect"] for r in rs),
                    "effect_range": [min(r["effect"] for r in rs), max(r["effect"] for r in rs)],
                    "acc_old_mean": st.mean(r["acc_old"] for r in rs), "acc_new_mean": st.mean(r["acc_new"] for r in rs),
                    "extra_chars_mean": st.mean(r["chars_new"] - r["chars_old"] for r in rs),
                    "extra_chars_range": [min(r["chars_new"] - r["chars_old"] for r in rs),
                                          max(r["chars_new"] - r["chars_old"] for r in rs)]}
    if "old" in s and "disp" in s:
        s["effect_change_disp_minus_old"] = s["disp"]["effect_mean"] - s["old"]["effect_mean"]
        s["extra_chars_change_disp_minus_old"] = s["disp"]["extra_chars_mean"] - s["old"]["extra_chars_mean"]
    return s


for role in ("Mathematician", "Lawyer"):
    rows = draws(role)
    if not rows:
        continue
    OUT[role] = {"draws": rows, "summary": summarize(rows)}
    print(f"\n### {role}\n")
    print("| 分派 | 采样 | 原提示词 | 优化后 | 效果 | 回答字符（原→优化后） | 多出字符 | 未解析（原/优化后） |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {'原' if r['master'] == 'old' else '新'} | {r['draw']} | {r['acc_old']:.1f} | {r['acc_new']:.1f} | "
              f"{r['effect']:+.1f} | {r['chars_old']:.0f} → {r['chars_new']:.0f} | {r['chars_new'] - r['chars_old']:+.0f} | "
              f"{r['unparsed_old']} / {r['unparsed_new']} |")
    print(json.dumps(OUT[role]["summary"], ensure_ascii=False, indent=1))

# Lawyer 线的团队格与验证集挑选
cells = {}
for c in ("C1", "C3", "C9", "C2L", "C4L", "C10L"):
    p = os.path.join(D, f"test_{c}.jsonl")
    if os.path.exists(p):
        cells[c] = {json.loads(l)["id"]: json.loads(l) for l in open(p)}
if "C2L" in cells:
    def acc(c, who=None):
        rs = cells[c].values()
        return 100 * sum((r["worker_correct"][who] if who else r["correct"]) for r in rs) / len(cells[c])
    OUT["lawyer_cells"] = {c: {"team": acc(c), "Lawyer": acc(c, "Lawyer")} for c in cells}
    print("\n### Lawyer 线团队格\n")
    print("| 格 | 团队 | Lawyer 自己 |")
    print("|---|---|---|")
    for c, v in OUT["lawyer_cells"].items():
        print(f"| {c} | {v['team']:.1f} | {v['Lawyer']:.1f} |")
vp = os.path.join(D, "opt", "val_worker_Lawyer.json")
if os.path.exists(vp):
    v = json.load(open(vp))
    OUT["lawyer_validation"] = {"scores": v["scores"], "own_domain_scores": v["own_domain_scores"], "chosen": v["chosen"],
                                "chosen_nonbase": v["chosen_nonbase"]}
json.dump(OUT, open(os.path.join(D, "draws.json"), "w"), ensure_ascii=False, indent=1)
