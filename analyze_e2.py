"""E2 汇总：植入的误导记忆 / 无用记忆在三种归因方式下的命运，以及补偿性记忆的数量。
补偿性记忆：新写入的记忆中主张"两个爪子都用 / 一趟搬两个"的条目（与 PLANT_BAD 相反）。
用法：python3 -I analyze_e2.py   输出 mas_runs/e2/e2_summary.json 并打印
"""
import json
import os
import re

E2 = "/home/claude/chime/mas_runs/e2"
COMPENSATE = re.compile(r"both grippers|two balls|two objects|pairs?\b|two at a time|each gripper|fill both|"
                        r"two per trip|both hands|maximi[sz]e (the )?(carry|capacity)|full capacity", re.I)


def flags(c):
    if "gates" in c:
        return {r: [i for i in (g.get("raw") or {}).get("misleading_memory_ids") or [] if str(i).startswith("PLANT")]
                for r, g in c["gates"].items()}
    return {"gate": [i for i in (c.get("gate") or {}).get("misleading_memory_ids") or [] if str(i).startswith("PLANT")]}


def retrieved_ids(c):
    r = c.get("retrieved")
    return set(r) if isinstance(r, list) else {i for v in (r or {}).values() for i in v}


def main():
    out = {}
    for tag in sorted(os.listdir(E2)):
        p = os.path.join(E2, tag, "episodes.jsonl")
        if not os.path.exists(p):
            continue
        eps = [json.loads(l) for l in open(p)]
        timeline, comp = [], []
        pruned_at = {}
        for e in eps:
            st = e["planted"]
            for kind in ("BAD", "INERT"):
                if st[kind] is None and kind not in pruned_at:
                    pruned_at[kind] = e["k"]
            fl = flags(e["chime"])
            timeline.append({"k": e["k"], "success": e["success"], "progress": round(e["progress"], 2),
                             "bad_retrieved": "PLANT_BAD" in retrieved_ids(e["chime"]),
                             "bad_flagged_by": [r for r, v in fl.items() if "PLANT_BAD" in v],
                             "inert_flagged_by": [r for r, v in fl.items() if "PLANT_INERT" in v],
                             "bad": st["BAD"] and [st["BAD"]["v"], st["BAD"]["n"]],
                             "inert": st["INERT"] and [st["INERT"]["v"], st["INERT"]["n"]],
                             "targets": e["chime"].get("targets") or e["chime"].get("gate", {}).get("attribution")
                             or [r for r, g in (e["chime"].get("gates") or {}).items()
                                 if str((g.get("raw") or {}).get("attribution", "")).lower() == "self"]})
            for m in e["inserted_text"]:
                if COMPENSATE.search(m["scenario"] + " " + m["experience"]):
                    comp.append({"k": e["k"], "stage": m["stage"], "experience": m["experience"][:240]})
        last = eps[-1]["planted"] if eps else {}
        out[tag] = {
            "n": len(eps), "success_pct": round(100 * sum(e["success"] for e in eps) / len(eps), 1) if eps else None,
            "bad_retrieved_eps": sum(t["bad_retrieved"] for t in timeline),
            "bad_flagged_eps": sum(bool(t["bad_flagged_by"]) for t in timeline),
            "inert_flagged_eps": sum(bool(t["inert_flagged_by"]) for t in timeline),
            "pruned_at_episode": pruned_at,
            "final_bad": last.get("BAD"), "final_inert": last.get("INERT"),
            "compensating_memories": len(comp), "compensating_by_stage": {
                s: sum(c["stage"] == s for c in comp) for s in sorted({c["stage"] for c in comp})},
            "cost": eps[-1]["usage"]["cost_cny_offpeak"] if eps else 0,
            "timeline": timeline, "compensating_examples": comp[:6],
        }
    json.dump(out, open(os.path.join(E2, "e2_summary.json"), "w"), ensure_ascii=False, indent=1)
    for tag, s in out.items():
        print(tag, {k: v for k, v in s.items() if k not in ("timeline", "compensating_examples")})


if __name__ == "__main__":
    main()
