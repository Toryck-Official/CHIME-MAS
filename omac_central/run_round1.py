"""第一轮：中心化团队上的 OMAC 式依次优化，以及 8 个测试组合 + 2 个噪声重跑。按"单元"分批跑，跑完一个分析一个。

步骤（与 OMAC 默认的 'Mathematician_rank' 顺序一致：先优化一个 worker，再优化 master 的部件）：
  opt:worker     原 master（M_old）下优化 Mathematician 的角色提示词 → W_new
  opt:dispatch / opt:route / opt:aggregate
                 固定 W_new，各自从原提示词出发优化 master 的一个部件（互不叠加）
  测试集（每类 30 题，共 180 题）：
       cell:C1 M_old+W_old    cell:C2 M_old+W_new
       cell:C3 M_disp+W_old   cell:C4 M_disp+W_new
       cell:C5 M_route+W_old  cell:C6 M_route+W_new
       cell:C7 M_agg+W_old    cell:C8 M_agg+W_new
       cell:C9 = C1 不走缓存重跑   cell:C10 = C2 不走缓存重跑（两者共用 salt r1，C10 只重跑 Mathematician 和汇总）
  wcell:old:r2 / wcell:disp:r1 …   只重跑 Mathematician 一条链（W_old、W_new 同一条消息各答一次），给 worker 层增益多采样
依赖：C2/C10 与 opt:dispatch/route/aggregate 需要 opt:worker 先完成；C3/C4 需要 opt:dispatch；C5/C6 需要 opt:route；
      C7/C8 需要 opt:aggregate。
选中规则：按 OMAC 取训练分数最高者；若最高的是原提示词（即 OMAC 不会改动这个部件），改用分数最高的非原提示词，
并在 opt/choice_*.json 里标出（否则这一格就没有改动可测）。
用法：python3 run_round1.py UNIT [UNIT ...]     同一条命令里的多个单元并行跑
      北京时间工作日高峰时段默认暂停新请求，ALLOW_PEAK=1 时照常跑；余额不足时暂停等充值。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import optimizer
import team

OUT = os.path.join(team.ROOT, "runs", "round1")
OPT = os.path.join(OUT, "opt")
SEED = 0
os.makedirs(OPT, exist_ok=True)
data = team.load_data(n_test=30)
train, test = data["train"], data["test"]
# cell → (被优化的 master 部件或 None, 是否用 W_new, salt)
CELLS = {"C1": (None, False, ""), "C2": (None, True, ""), "C3": ("dispatch", False, ""), "C4": ("dispatch", True, ""),
         "C5": ("route", False, ""), "C6": ("route", True, ""), "C7": ("aggregate", False, ""),
         "C8": ("aggregate", True, ""), "C9": (None, False, "r1"), "C10": (None, True, "r1")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | api_calls={team.COST['calls']} cached={team.COST['cached']} "
          f"cny={team.COST['cny']:.3f}", flush=True)


def choice(component):
    """读优化结果，按选中规则返回 (pid, prompt)，并写 choice_{component}.json。"""
    res = json.load(open(os.path.join(OPT, f"opt_{component}.json")))
    pid = res["chosen"] if res["chosen"] != 0 else res["chosen_nonbase"]
    json.dump({"component": component, "chosen": res["chosen"], "used": pid, "fallback_nonbase": res["chosen"] == 0,
               "chosen_by_team_score": res.get("chosen_by_team_score"), "scores": res["scores"],
               "team_scores": res["team_scores"], "local_scores": res["local_scores"], "prompt": res["candidates"][pid]},
              open(os.path.join(OPT, f"choice_{component}.json"), "w"), ensure_ascii=False, indent=1)
    return pid, res["candidates"][pid]


def cfg_for(component, use_w_new):
    cfg = team.base_config()
    if use_w_new:
        cfg = optimizer.with_candidate(cfg, "worker", choice("worker")[1])
    if component:
        cfg = optimizer.with_candidate(cfg, component, choice(component)[1])
    return cfg


def run_wcell(master, salt):
    """只重跑 Mathematician 这一条链（分派给它的消息 + W_old、W_new 各答一次），给 worker 层的增益多采几次样。
    master：old 或 disp；salt 换一个就是一次新的独立重跑（分派消息和作答都重新生成）。"""
    comp = None if master == "old" else "dispatch"
    cfg_old, cfg_new = cfg_for(comp, False), cfg_for(comp, True)
    W = "Mathematician"

    def one(item):
        m = team.dispatch(item, cfg_old, W, salt)  # 分派不依赖 worker 提示词
        r_old, r_new = team.work(item, cfg_old, W, m, salt), team.work(item, cfg_new, W, m, salt)
        a_old, a_new = team.parse_single_choice(r_old), team.parse_single_choice(r_new)
        return {"id": item["id"], "domain": item["domain"], "gold": item["answer"], "message": m,
                "resp_old": r_old, "resp_new": r_new, "ans_old": a_old, "ans_new": a_new,
                "correct_old": a_old == item["answer"], "correct_new": a_new == item["answer"]}
    with ThreadPoolExecutor(60) as ex:
        recs = list(ex.map(one, test))
    with open(os.path.join(OUT, f"wtest_{master}_{salt or 'r0'}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    g = (sum(r["correct_new"] for r in recs) - sum(r["correct_old"] for r in recs)) / len(recs)
    log(f"wcell:{master}:{salt}: W_old {sum(r['correct_old'] for r in recs)}/{len(recs)}, "
        f"W_new {sum(r['correct_new'] for r in recs)}/{len(recs)}, gain {100 * g:+.1f}")


def run_unit(unit):
    kind, name = unit.split(":", 1)
    t0 = time.time()
    if kind == "wcell":
        master, salt = name.split(":")
        run_wcell(master, salt)
        return
    if kind == "opt":
        base = team.base_config() if name == "worker" else cfg_for(None, True)
        optimizer.optimize(name, base, train, OPT, "Mathematician", SEED)
        pid, _ = choice(name)
        log(f"opt:{name} done in {(time.time() - t0) / 60:.1f} min, used candidate {pid}")
    else:
        comp, use_w, salt = CELLS[name]
        cfg = cfg_for(comp, use_w)
        recs = team.evaluate(test, cfg, salt=salt, threads=60)
        with open(os.path.join(OUT, f"test_{name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump({"cell": name, "salt": salt, "cfg": cfg}, open(os.path.join(OUT, f"cfg_{name}.json"), "w"),
                  ensure_ascii=False, indent=1)
        log(f"cell:{name} team acc {team.accuracy(recs):.4f} on {len(recs)} in {(time.time() - t0) / 60:.1f} min")


def main(units):
    log(f"start {units}: train {len(train)}, test {len(test)}, peak={team.is_peak()}, "
        f"ALLOW_PEAK={os.environ.get('ALLOW_PEAK', '0')}")
    with ThreadPoolExecutor(len(units)) as ex:
        list(ex.map(run_unit, units))
    with open(os.path.join(OUT, "cost_log.jsonl"), "a") as f:
        f.write(json.dumps({"units": units, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **team.COST}) + "\n")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1:])
