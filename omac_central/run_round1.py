"""第一轮：中心化团队上的 OMAC 式依次优化，以及 8 个测试组合 + 2 个噪声重跑。

步骤（与 OMAC 默认的 'Mathematician_rank' 顺序一致：先优化一个 worker，再优化 master 的部件）：
  1. 原 master（M_old）下优化 Mathematician 的角色提示词 → W_new
  2. 固定 W_new，分别从原提示词出发优化 master 的 dispatch / route / aggregate → M_disp / M_route / M_agg
     （三个部件各自独立优化，互不叠加）
  3. 测试集（每类 30 题，共 180 题）上跑：
       C1 M_old+W_old   C2 M_old+W_new
       C3 M_disp+W_old  C4 M_disp+W_new
       C5 M_route+W_old C6 M_route+W_new
       C7 M_agg+W_old   C8 M_agg+W_new
       C9 = C1 不走缓存重跑   C10 = C2 不走缓存重跑（两者共用同一 salt，所以 C10 只重跑 Mathematician 和汇总）
选中规则：按 OMAC 取训练分数最高者；若最高的是原提示词（即 OMAC 不会改动这个部件），改用分数最高的非原提示词，
并在 summary 里标出（否则这一格就没有改动可测）。
用法：python3 run_round1.py            （北京时间工作日高峰时段会自动暂停新请求）
      ALLOW_PEAK=1 python3 run_round1.py --dry   （3 道训练题、3 道测试题，验证流程）
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import optimizer
import team

DRY = "--dry" in sys.argv
OUT = os.path.join(team.ROOT, "runs", "round1_dry" if DRY else "round1")
SEED = 99 if DRY else 0
os.makedirs(OUT, exist_ok=True)
data = team.load_data(n_test=1 if DRY else 30)
DRY_SUBJ = ("math", "health", "law")
train = [it for it in data["train"] if it["idx"] < 1 and it["subject"] in DRY_SUBJ] if DRY else data["train"]
test = [it for it in data["test"] if it["subject"] in DRY_SUBJ] if DRY else data["test"]
COMPONENTS = ["dispatch", "route", "aggregate"]
CELLS = {"dispatch": ("C3", "C4"), "route": ("C5", "C6"), "aggregate": ("C7", "C8")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | api_calls={team.COST['calls']} cached={team.COST['cached']} "
          f"cny={team.COST['cny']:.3f}", flush=True)


def run_cell(name, cfg, salt=""):
    path = os.path.join(OUT, f"test_{name}.jsonl")
    recs = team.evaluate(test, cfg, salt=salt, threads=20)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"cell": name, "salt": salt, "cfg": cfg}, open(os.path.join(OUT, f"cfg_{name}.json"), "w"),
              ensure_ascii=False, indent=1)
    log(f"{name}: team acc {team.accuracy(recs):.4f} on {len(recs)}")
    return recs


def pick(res):
    pid = res["chosen"] if res["chosen"] != 0 else res["chosen_nonbase"]
    return pid, res["candidates"][pid], res["chosen"] == 0


def main():
    t0 = time.time()
    base = team.base_config()
    opt_dir = os.path.join(OUT, "opt")
    log(f"start: train {len(train)}, test {len(test)}")

    # 1. 基线测试 + 噪声重跑 + worker 优化（互不依赖，并行）
    with ThreadPoolExecutor(3) as ex:
        f1 = ex.submit(run_cell, "C1", base)
        f9 = ex.submit(run_cell, "C9", base, "r1")
        fw = ex.submit(optimizer.optimize, "worker", base, train, opt_dir, "Mathematician", SEED)
        wres = fw.result()
        f1.result(), f9.result()
    w_id, w_new, w_fallback = pick(wres)
    cfg_w = optimizer.with_candidate(base, "worker", w_new)
    log(f"worker done: chosen {wres['chosen']}, used {w_id}, fallback={w_fallback}")

    # 2. 固定 W_new，分别优化 master 三个部件；同时跑 C2、C10
    with ThreadPoolExecutor(5) as ex:
        fs = {c: ex.submit(optimizer.optimize, c, cfg_w, train, opt_dir, "Mathematician", SEED) for c in COMPONENTS}
        f2 = ex.submit(run_cell, "C2", cfg_w)
        f10 = ex.submit(run_cell, "C10", cfg_w, "r1")
        mres = {c: f.result() for c, f in fs.items()}
        f2.result(), f10.result()
    log("master components done")

    # 3. 测试 C3–C8
    summary = {"worker": {"chosen": wres["chosen"], "used": w_id, "fallback_nonbase": w_fallback,
                          "scores": wres["scores"], "team_scores": wres["team_scores"],
                          "local_scores": wres["local_scores"], "chosen_by_team_score": wres["chosen_by_team_score"],
                          "prompt": w_new}}
    jobs = {}
    for c in COMPONENTS:
        pid, prompt, fb = pick(mres[c])
        summary[c] = {"chosen": mres[c]["chosen"], "used": pid, "fallback_nonbase": fb, "scores": mres[c]["scores"],
                      "local_scores": mres[c]["local_scores"], "prompt": prompt}
        old, new = CELLS[c]
        jobs[old] = optimizer.with_candidate(base, c, prompt)
        jobs[new] = optimizer.with_candidate(cfg_w, c, prompt)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda kv: run_cell(*kv), jobs.items()))
    summary["cost"] = dict(team.COST)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
    log("all done")


if __name__ == "__main__":
    main()
