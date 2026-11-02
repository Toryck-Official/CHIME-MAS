"""RQ2a（离线，不调 API）：用协作日志里的依赖信息，能否事先判断"下一步改哪个部件"更划算？

做法：在开始优化 master 时的状态（M_old + W_new）下，把团队答错的题按"改哪一个部件就有可能改对"分类。
因为三个 worker 每题都答（影子调用），这些分类可以直接从日志里读出来：
  - 选人可修：被选中的 worker 全错，但有没被选中的 worker 答对了（换个人就可能对）
  - 汇总可修：至少一个被选中的 worker 答对了，但最终答案错了（汇总没采纳对的）
  - 作答可修：三个 worker 全错（只有改变 worker 收到的内容或 worker 本身才可能改对；对应 dispatch 和 worker 提示词）
每类的题数是只改该部件时团队正确率的提升上限（"可修上限"）。再拿第一轮各部件优化后的实际提升对照：
训练集上 = 最优候选的分数 − 原提示词分数；测试集上 = C4/C6/C8 − C2。
用法：python3 rq2a_offline.py [runs/round1]
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round1")
W = ["Mathematician", "Doctor", "Lawyer"]


def classify(recs):
    c = {"n": len(recs), "team_wrong": 0, "route_fixable": 0, "aggregate_fixable": 0, "answer_fixable": 0,
         "aggregate_overrode_unanimous_correct": 0}
    for r in recs:
        if r["correct"]:
            continue
        c["team_wrong"] += 1
        routed_right = [w for w in r["selected"] if r["worker_correct"][w]]
        other_right = [w for w in W if w not in r["selected"] and r["worker_correct"][w]]
        if routed_right:
            c["aggregate_fixable"] += 1
            if len(routed_right) == len(r["selected"]):
                c["aggregate_overrode_unanimous_correct"] += 1
        elif other_right:
            c["route_fixable"] += 1
        else:
            c["answer_fixable"] += 1
    return c


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


out = {}
s = json.load(open(os.path.join(D, "summary.json")))
w_used = s["worker"]["used"]
train_state = load_jsonl(os.path.join(D, "opt", f"train_worker_{w_used}.jsonl"))  # M_old + W_new，训练集
test_state = load_jsonl(os.path.join(D, "test_C2.jsonl"))
out["headroom_train"] = classify(train_state)
out["headroom_test"] = classify(test_state)
# worker 优化开始时（M_old + W_old）的状态，作为对照
out["headroom_train_before_worker_opt"] = classify(load_jsonl(os.path.join(D, "opt", "train_worker_0.jsonl")))


def acc(recs):
    return sum(r["correct"] for r in recs) / len(recs)


realized = {}
c2 = acc(test_state)
for comp, cell in (("dispatch", "C4"), ("route", "C6"), ("aggregate", "C8")):
    sc = {int(k): v for k, v in s[comp]["scores"].items()}
    t = load_jsonl(os.path.join(D, f"test_{cell}.jsonl"))
    realized[comp] = {"train_gain_items": round((sc[s[comp]["used"]] - sc[0]) * out["headroom_train"]["n"]),
                      "train_best_gain_items": round((max(sc.values()) - sc[0]) * out["headroom_train"]["n"]),
                      "test_gain_items": round((acc(t) - c2) * len(t)) if t else None}
out["realized"] = realized
map_ = {"dispatch": "answer_fixable", "route": "route_fixable", "aggregate": "aggregate_fixable"}
out["ranking"] = {
    "by_headroom_train": sorted(map_, key=lambda c: -out["headroom_train"][map_[c]]),
    "by_realized_train": sorted(realized, key=lambda c: -realized[c]["train_best_gain_items"]),
    "by_realized_test": sorted(realized, key=lambda c: -(realized[c]["test_gain_items"] or 0)),
}
json.dump(out, open(os.path.join(D, "rq2a.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
