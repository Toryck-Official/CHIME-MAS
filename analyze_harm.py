"""危害实验汇总：orig 与 oracle 的评估段成功率、来自无责失败的教训占比、记忆条数，以及无责失败上归因门给出的标签。
用法：python3 -I analyze_harm.py   输出 mas_runs/harm/harm_summary.json 并打印
"""
import collections
import json
import os

H = "/home/claude/chime/mas_runs/harm"


def inserted(c):
    ups = [c.get("update")] if "update" in c else [g.get("update") for g in (c.get("gates") or {}).values()]
    return sum(len((u or {}).get("inserted", [])) + len((u or {}).get("merged", [])) for u in ups)


def labels(c):
    if "gates" in c:
        return [r for r, g in c["gates"].items() if str((g.get("raw") or {}).get("attribution", "")).lower() == "self"]
    return (c.get("gate") or {}).get("attribution")


def main():
    out = {}
    for tag in sorted(os.listdir(H)):
        p = os.path.join(H, tag, "episodes.jsonl")
        if not os.path.exists(p):
            continue
        eps = [json.loads(l) for l in open(p)]
        tr = [e for e in eps if e["phase"] == "train"]
        ev = [e for e in eps if e["phase"] == "eval"]
        nf = [e for e in tr if e["no_fault"]]
        w_nf = sum(inserted(e["chime"]) for e in nf)
        w_all = sum(inserted(e["chime"]) for e in tr)
        summ = json.load(open(os.path.join(H, tag, "summary.json"))) if os.path.exists(os.path.join(H, tag, "summary.json")) else {}
        out[tag] = {
            "n_train": len(tr), "n_eval": len(ev),
            "train_success": round(100 * sum(e["success"] for e in tr) / len(tr), 1) if tr else None,
            "eval_success": round(100 * sum(e["success"] for e in ev) / len(ev), 1) if ev else None,
            "eval_progress": round(100 * sum(e["progress"] for e in ev) / len(ev), 1) if ev else None,
            "no_fault_failures": len(nf),
            "writes_total": w_all, "writes_from_no_fault": w_nf,
            "writes_from_no_fault_pct": round(100 * w_nf / w_all, 1) if w_all else None,
            "gate_labels_on_no_fault": dict(collections.Counter(str(labels(e["chime"])) for e in nf
                                                                if not e["chime"].get("oracle_skipped"))),
            "memory_size": summ.get("memory_size"), "cost": eps[-1]["usage"]["cost_cny_offpeak"] if eps else 0,
            "eval_by_task": {e["task_index"]: e["success"] for e in ev},
        }
    # 配对：同一顺序下 orig 与 oracle 在评估段同一批题上的胜负
    for tag, s in out.items():
        if "_orig" in tag:
            o = out.get(tag.replace("_orig", "_oracle"))
            if o:
                a, b = s["eval_by_task"], o["eval_by_task"]
                common = set(a) & set(b)
                s["paired_vs_oracle"] = {"oracle_wins": sum(b[t] and not a[t] for t in common),
                                         "orig_wins": sum(a[t] and not b[t] for t in common), "n": len(common)}
    json.dump(out, open(os.path.join(H, "harm_summary.json"), "w"), ensure_ascii=False, indent=1)
    for tag, s in out.items():
        print(tag, {k: v for k, v in s.items() if k != "eval_by_task"})


if __name__ == "__main__":
    main()
