"""E1 汇总：各归因门条件对照真相的准确率、混淆矩阵、误导记忆召回、置信度校准；多智能体截短样本的过度归责。
用法：python3 -I analyze_e1.py   输出 mas_runs/e1/e1_summary.json 并打印
"""
import collections
import json
import os

E1 = os.environ.get("E1_SAMPLES", "/home/claude/chime/mas_runs/e1")
LABELS = ("planning", "execution", "both", "none")


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    allrows = [json.loads(l) for l in open(os.path.join(E1, "gate_results.jsonl"))]
    unparsed = collections.Counter(r["cond"] for r in allrows if "_unparsed" in json.dumps(r.get("raw", {}))[:40])
    rows = [r for r in allrows if "_unparsed" not in json.dumps(r.get("raw", {}))[:40]]
    single = [r for r in rows if not r["kind"].startswith("mas")]
    mas = [r for r in rows if r["kind"].startswith("mas")]
    out = {"single": {}, "mas": {}, "unparsed_excluded": dict(unparsed)}
    for cond in sorted({r["cond"] for r in single}):
        rs = [r for r in single if r["cond"] == cond and r["strict"]]
        conf = {t: collections.Counter(r["label"] for r in rs if r["truth"] == t) for t in ("planning", "execution", "none")}
        by_kind = {}
        for k in sorted({r["kind"] for r in rs}):
            ks = [r for r in rs if r["kind"] == k]
            by_kind[k] = {"n": len(ks), "exact": pct(sum(r["label"] == r["truth"] for r in ks), len(ks)),
                          "labels": dict(collections.Counter(r["label"] for r in ks))}
        mem = [r for r in rs if r["kind"] == "mem"]
        bins = collections.defaultdict(list)
        for r in rs:
            b = "<0.8" if r["conf"] < 0.8 else "0.8-0.89" if r["conf"] < 0.9 else ">=0.9"
            bins[b].append(r["label"] == r["truth"])
        out["single"][cond] = {
            "n": len(rs), "exact_acc": pct(sum(r["label"] == r["truth"] for r in rs), len(rs)),
            "by_kind": by_kind,
            "confusion_truth_to_label": {t: dict(c) for t, c in conf.items()},
            "label_dist": dict(collections.Counter(r["label"] for r in rs)),
            "mis_recall_on_mem": pct(sum("PLANT_BAD" in r["mis"] for r in mem), len(mem)) if mem else None,
            "calibration": {b: {"n": len(v), "acc": pct(sum(v), len(v))} for b, v in sorted(bins.items())},
            "n_nonstrict_excluded": sum(1 for r in single if r["cond"] == cond and not r["strict"]),
        }
    for cond in sorted({r["cond"] for r in mas}):
        rs = [r for r in mas if r["cond"] == cond]
        out["mas"][cond] = {
            "n": len(rs),
            "correct_no_role_blamed_pct": pct(sum(not r["blamed"] for r in rs), len(rs)),
            "mean_roles_blamed": round(sum(len(r["blamed"]) for r in rs) / len(rs), 2) if rs else None,
            "lessons_written": sum(r["lessons"] for r in rs),
            "role_blamed_counts": dict(collections.Counter(x for r in rs for x in r["blamed"])),
        }
    json.dump(out, open(os.path.join(E1, "e1_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
