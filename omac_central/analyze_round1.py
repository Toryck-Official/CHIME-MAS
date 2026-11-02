"""第一轮分析：RQ1（master 部件被优化后，已优化 worker 的增益是否保留）与 Phase 0（worker 收到的内容变了多少）。

输出 runs/round1/analysis.json，并打印 Markdown 表格。
用法：python3 analyze_round1.py [runs/round1]
主要量：
  - 各格团队正确率、各 worker 自己的正确率（影子调用：三个 worker 每题都答）
  - W_new 的增益 g_M = acc(M, W_new) − acc(M, W_old)，以及交互 g_M − g_old（按题配对 bootstrap，10000 次）
    团队层用团队对错，worker 层用 Mathematician 自己的对错
  - 噪声参照：C9/C10 分别是 C1/C2 的不走缓存重跑，g_rerun = acc(C10) − acc(C9)
  - Phase 0：同一题、同一 worker，原 dispatch 与优化后 dispatch 写出的消息相似度、长度比、题干/选项是否原样保留、
    是否加了答案格式要求或提示；以原 dispatch 的重跑（C9）作噪声参照
  - 行为：Mathematician 答案改变率（换 dispatch vs 重跑）、W_new 逐题赢/输集合在新 master 下的保留率、
    选人参与率、汇总采纳率
  - 敏感性：宽松解析（认 'Final answer: X' 等格式）下重算正确率与增益，区分格式失败与答错
"""
import difflib
import json
import os
import re
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round1")
W = ["Mathematician", "Doctor", "Lawyer"]
MW = "Mathematician"
B = 10000
rng = np.random.default_rng(0)
MASTERS = {"old": ("C1", "C2"), "disp": ("C3", "C4"), "route": ("C5", "C6"), "agg": ("C7", "C8"),
           "rerun": ("C9", "C10")}


def load(cell):
    p = os.path.join(D, f"test_{cell}.jsonl")
    if not os.path.exists(p):
        return None
    rs = [json.loads(l) for l in open(p)]
    return {r["id"]: r for r in rs}


cells = {c: load(c) for c in [f"C{i}" for i in range(1, 11)]}
cells = {c: v for c, v in cells.items() if v}
ids = sorted(cells["C1"])
n = len(ids)
data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mmlu_pro_central.json")))
items = {it["id"]: it for it in data["test"]}


def parse_strict(reply):
    """与 team.parse_single_choice 相同（OMAC 解析规则扩展到 A–J）。"""
    m = re.findall(r"\(([A-J])\)", reply or "")
    if m:
        return m[-1]
    m = re.findall(r"([A-J])\)", reply or "")
    return m[-1] if m else None


def parse_lenient(reply):
    """敏感性分析用：严格规则解析不出时，再认 'Final answer: X' / 'answer is X' / 结尾单独的字母。
    用来区分"格式没照做"和"真的答错"——优化后的 dispatch 可能要求别的答案格式。"""
    s = parse_strict(reply)
    if s:
        return s
    m = re.findall(r"(?i:final answer|answer|choice|option)\s*(?:is|:|：|=)?\s*[*\s]*\(?([A-J])\b", reply or "")
    if m:
        return m[-1]
    m = re.findall(r"\b([A-J])\b[\s*.]*$", (reply or "").strip())
    return m[-1] if m else None


def vec(cell, kind="team"):
    """kind：team / team_lenient / 某个 worker 名 / 某个 worker 名 + '_lenient'。"""
    rs = cells[cell]
    if kind == "team":
        return np.array([rs[i]["correct"] for i in ids], dtype=float)
    if kind == "team_lenient":
        return np.array([parse_lenient(rs[i]["agg_out"]) == rs[i]["gold"] for i in ids], dtype=float)
    if kind.endswith("_lenient"):
        w = kind[:-len("_lenient")]
        return np.array([parse_lenient(rs[i]["worker_resp"][w]) == rs[i]["gold"] for i in ids], dtype=float)
    return np.array([rs[i]["worker_correct"][kind] for i in ids], dtype=float)


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


IDX = rng.integers(0, n, size=(B, n))


def boot_mean(x):
    return {"est": round(float(x.mean()), 4), "ci95": ci(x[IDX].mean(axis=1))}


out = {"n_test": n, "cells_present": sorted(cells, key=lambda c: int(c[1:]))}

# ---------- 1. 正确率 ----------
acc = {}
for c in out["cells_present"]:
    acc[c] = {"team": boot_mean(vec(c))}
    for w in W:
        acc[c][w] = round(float(vec(c, w).mean()), 4)
    for dom, w in (("math", "Mathematician"), ("med", "Doctor"), ("law", "Lawyer")):
        sel = np.array([cells[c][i]["domain"] == dom for i in ids])
        acc[c][f"{w}_own_domain"] = round(float(vec(c, w)[sel].mean()), 4)
    acc[c]["final_unparsed"] = sum(cells[c][i]["final"] is None for i in ids)
    acc[c]["worker_unparsed"] = {w: sum(cells[c][i]["worker_ans"][w] is None for i in ids) for w in W}
    acc[c]["worker_unparsed_lenient"] = {w: sum(parse_lenient(cells[c][i]["worker_resp"][w]) is None for i in ids)
                                         for w in W}
    acc[c]["team_lenient"] = round(float(vec(c, "team_lenient").mean()), 4)
    acc[c][f"{MW}_lenient"] = round(float(vec(c, MW + "_lenient").mean()), 4)
out["accuracy"] = acc

# ---------- 2. W_new 的增益与交互 ----------
gains = {}
for kind in ("team", MW, "team_lenient", MW + "_lenient"):
    g = {}
    for m, (co, cn) in MASTERS.items():
        if co in cells and cn in cells:
            g[m] = vec(cn, kind) - vec(co, kind)
    res = {m: boot_mean(v) for m, v in g.items()}
    for m in ("disp", "route", "agg", "rerun"):
        if m in g and "old" in g:
            d = g[m] - g["old"]
            res[f"interaction_{m}"] = boot_mean(d)
    gains[kind] = res
out["gain_of_W_new"] = gains


# ---------- 3. Phase 0：worker 收到的消息 ----------
def norm(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


FMT_PAT = re.compile(r"final answer\s*[:：]|answer\s*:\s*\[|format|\(x\)|box", re.I)
HINT_PAT = re.compile(r"\bhint\b|the (correct )?answer (is|should be|appears to be)|likely (answer|option)|"
                      r"i (believe|think) (the answer|it) is", re.I)
STEP_PAT = re.compile(r"step[- ]by[- ]step|verify|double[- ]check|check (your|each)", re.I)


def msg_stats(a_cell, b_cell, w):
    sims, lens, stem_a, stem_b, opt_a, opt_b, fmt, hint, step = [], [], 0, 0, 0, 0, 0, 0, 0
    for i in ids:
        a, b = cells[a_cell][i]["messages"][w], cells[b_cell][i]["messages"][w]
        sims.append(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())
        lens.append(len(b) / max(1, len(a)))
        it = items[i]
        q = norm(it["question"])
        stem_a += q in norm(a)
        stem_b += q in norm(b)
        opt_a += all(norm(o) in norm(a) for o in it["options"])
        opt_b += all(norm(o) in norm(b) for o in it["options"])
        src = it["question"] + " " + " ".join(it["options"])  # 只数题目原文之外新加的措辞
        fmt += len(FMT_PAT.findall(b)) > len(FMT_PAT.findall(src))
        hint += len(HINT_PAT.findall(b)) > len(HINT_PAT.findall(src))
        step += len(STEP_PAT.findall(b)) > len(STEP_PAT.findall(src))
    s = np.array(sims)
    return {"similarity_median": round(float(np.median(s)), 3), "similarity_mean": round(float(s.mean()), 3),
            "share_similarity_below_0.8": round(float((s < 0.8).mean()), 3),
            "length_ratio_median": round(float(np.median(lens)), 2),
            "stem_verbatim": [stem_a, stem_b], "all_options_verbatim": [opt_a, opt_b],
            "new_msgs_with_format_instruction": fmt, "new_msgs_with_hint_or_answer": hint,
            "new_msgs_with_step_or_verify": step}


phase0 = {}
if "C3" in cells:
    for w in W:
        phase0[w] = {"disp0_vs_disp1": msg_stats("C1", "C3", w)}
        if "C9" in cells:
            phase0[w]["disp0_vs_rerun"] = msg_stats("C1", "C9", w)
out["phase0_messages"] = phase0


# ---------- 4. 行为：答案改变、逐题赢/输保留 ----------
def flips(c1, c2, w):
    return sum(cells[c1][i]["worker_ans"][w] != cells[c2][i]["worker_ans"][w] for i in ids)


beh = {}
pairs = {"W_old: disp change (C1→C3)": ("C1", "C3"), "W_old: rerun (C1→C9)": ("C1", "C9"),
         "W_new: disp change (C2→C4)": ("C2", "C4"), "W_new: rerun (C2→C10)": ("C2", "C10")}
for name, (a, b) in pairs.items():
    if a in cells and b in cells:
        beh[name] = {w: flips(a, b, w) for w in W}
out["worker_answer_changes"] = beh


def winloss(co, cn, kind):
    o, nw = vec(co, kind), vec(cn, kind)
    return set(np.array(ids)[(nw == 1) & (o == 0)]), set(np.array(ids)[(nw == 0) & (o == 1)])


ret = {}
for kind in ("team", MW):
    if "C2" not in cells:
        break
    F0, L0 = winloss("C1", "C2", kind)
    r = {"old": {"wins": len(F0), "losses": len(L0)}}
    for m, (co, cn) in MASTERS.items():
        if m == "old" or co not in cells or cn not in cells:
            continue
        F, L = winloss(co, cn, kind)
        r[m] = {"wins": len(F), "losses": len(L), "wins_kept": len(F0 & F), "losses_kept": len(L0 & L)}
    ret[kind] = r
out["item_level_retention"] = ret


# ---------- 5. 选人参与率、汇总采纳率 ----------
def route_stats(c):
    rs = cells[c]
    part = {w: round(sum(w in rs[i]["selected"] for i in ids) / n, 3) for w in W}
    part["mean_experts"] = round(sum(len(rs[i]["selected"]) for i in ids) / n, 2)
    part["fallback"] = sum(rs[i]["route_fallback"] for i in ids)
    return part


def agg_stats(c):
    rs = cells[c]
    adopt, override, unanimous_override = {}, 0, 0
    for w in W:
        sel = [i for i in ids if w in rs[i]["selected"] and rs[i]["worker_ans"][w]]
        adopt[w] = round(sum(rs[i]["final"] == rs[i]["worker_ans"][w] for i in sel) / max(1, len(sel)), 3)
    for i in ids:
        answers = {rs[i]["worker_ans"][w] for w in rs[i]["selected"]}
        if rs[i]["final"] not in answers:
            override += 1
        if len(answers) == 1 and None not in answers and rs[i]["final"] not in answers:
            unanimous_override += 1
    return {"adoption_given_routed": adopt, "final_not_any_routed_answer": override,
            "override_of_unanimous_routed": unanimous_override}


out["routing"] = {c: route_stats(c) for c in out["cells_present"]}
out["aggregation"] = {c: agg_stats(c) for c in out["cells_present"]}

# ---------- 6. 优化过程 ----------
opt = {}
for comp in ("worker", "dispatch", "route", "aggregate"):
    p = os.path.join(D, "opt", f"choice_{comp}.json")
    if os.path.exists(p):
        c = json.load(open(p))
        opt[comp] = {k: c.get(k) for k in ("chosen", "used", "fallback_nonbase", "chosen_by_team_score", "scores",
                                            "team_scores", "local_scores")}
        opt[comp]["prompt_chars"] = len(c["prompt"])
out["optimization"] = opt
cl = os.path.join(D, "cost_log.jsonl")
if os.path.exists(cl):
    out["cost_runs"] = [json.loads(l) for l in open(cl)]

json.dump(out, open(os.path.join(D, "analysis.json"), "w"), ensure_ascii=False, indent=1)


# ---------- 打印 ----------
def f(x):
    return f"{100 * x['est']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"


print(f"n_test = {n}\n")
print("| cell | team acc (95% CI) | Math | Doctor | Lawyer | Math own-domain |")
print("|---|---|---|---|---|---|")
for c in out["cells_present"]:
    a = acc[c]
    t = a["team"]
    print(f"| {c} | {100 * t['est']:.1f} [{100 * t['ci95'][0]:.1f}, {100 * t['ci95'][1]:.1f}] | {100 * a['Mathematician']:.1f} | "
          f"{100 * a['Doctor']:.1f} | {100 * a['Lawyer']:.1f} | {100 * a['Mathematician_own_domain']:.1f} |")
print("\nGain of W_new (points, 95% CI):")
for kind, r in gains.items():
    print(f"  [{kind}] " + "; ".join(f"{m}: {f(v)}" for m, v in r.items()))
print("\nPhase 0:", json.dumps(phase0, ensure_ascii=False, indent=1))
print("\nWorker answer changes:", json.dumps(beh, ensure_ascii=False))
print("\nItem-level retention:", json.dumps(ret, ensure_ascii=False))
print("\nRouting:", json.dumps(out["routing"], ensure_ascii=False))
print("\nAggregation:", json.dumps(out["aggregation"], ensure_ascii=False))
