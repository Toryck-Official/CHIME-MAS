"""OMAC 优化算法（Semantic Initializer + Contrastive Comparator）的移植，用于中心化团队的四个部件：
worker（Mathematician 的角色提示词）、master 的 dispatch / route / aggregate 提示词。

与 OMAC 代码（code/MMLU/run_evol.py、prompt_iteration/prompt.py）一致的部分：
  - initialize_num=3：示例提示词（id 0）+ 在同一段对话里依次让 LLM 输出 2 个新提示词
    （"Directly output the {i}th prompt without any prefix or explanation."）
  - FD_max_iter=3，FD_min_improvement=0.01，FD_tolerance_iter=2，FD_sample_threshold=1：
    第 t 轮正父本 = 分数最高者，负父本 = 分数最低的 (1+t) 个里随机一个（不超过已有数量的一半）
  - 父本以 '\n\nThe positive parent prompt is: "…"\n\nThe negative parent prompt is: "…"' 附在 FD 模板后，
    问 "Directly output the content of the child prompt without any prefix or explanation."
  - 最终取训练分数最高者；同分取 id 小的（Python 稳定排序，id 0 即原提示词）
  - 优化器温度 1.0，max_tokens 2048
  - worker 的 ROLE_INIT / ROLE_FD 模板原文照搬；route 改写自 PRE_RANK_*，aggregate 改写自 RANK_*，
    dispatch 是新部件，按同样格式新写
与 OMAC 的差别：
  - OMAC 给初始化生成的提示词加 "{uuid}: " 前缀（仅为了作字典键，会被 agent 看到），这里不加
  - 负父本的随机数用固定种子
  - worker 的分数用它自己在训练集全部 60 题上的正确率（影子调用）。中心化团队里 master 会自己解题
    （冒烟测试中选人阶段就已算出答案），团队分数对 worker 提示词几乎不敏感，用 OMAC 的团队分数
    选 worker 基本是在选噪声。团队分数照常记录，也记录按 OMAC 团队分数会选中哪一个。
  - master 三个部件的分数是团队在训练集上的正确率（与 OMAC 一致）
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import team

INIT_NUM, FD_MAX_ITER, FD_MIN_IMPROVEMENT, FD_TOLERANCE_ITER, FD_SAMPLE_THRESHOLD = 3, 3, 0.01, 2, 1
OPT_TEMPERATURE, OPT_MAX_TOKENS = 1.0, 2048

# ---------------- worker：OMAC 原文 ----------------
ROLE_INIT = """Generate {} distinct prompts to instruct an LLM to resolve some general reasoning problems acting as the given role: '{}'.
Each prompt should guide the model to accurately and efficiently resolve problems while adhering to the specified role.
Each prompt must begin strictly with the following content: '{}'. Then, you should consider adding more detailed, logical, and thorough instructions, which can help the LLM resolve problems better while acting as the given role.
Do not output anything currently. Instead, I will provide a sequence number, and you should return only the corresponding prompt one by one.
Do not create any specific instances of the problems in the prompt, because they are not provided now.
Ensure that the generated prompts follow the given example format but differ in content and structure from the example itself. The example is as follows: '{}'.
"""

ROLE_FD = """Generate and output a child prompt for an LLM to resolve some general reasoning problems acting like the given role '{}'.
At the end, a pair of parent prompts is provided: one positive and one negative. The positive parent prompt has been shown to be more effective and efficient in guiding the LLM to resolve problems following the given role.
Your task is to carefully compare the two parent prompts, identifying the key reasons why the positive parent prompt performs better. Based on these insights, generate and output a child prompt that further improves upon the positive parent prompt to enhance problem-solving.
Do not create any instances of the problem in the prompt, because they are not provided now.
The child prompt must begin strictly with the following content: '{}'. Then, you can consider adding more detailed, logical, and thorough instructions based on the insights you have gained from the comparison.
Output only the content of the child prompt excluding the reasoning process."""

# ---------------- route：改写自 OMAC 的 PRE_RANK_INIT / PRE_RANK_FD ----------------
ROUTE_INIT = """Create {} distinct prompts for an LLM to choose some experts best suited for resolving some general reasoning problems related to math, hard science, humanities, social sciences, etc.
Each prompt should decide and specify the number of the chosen experts, or let the LLM decide it. The minimal number is 1 and maximum number is 3.
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should help to accurately and efficiently identify the experts best suited for problem solving.
Note that all information about the task and candidate experts has been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for expert selection.
You may consider adding more detailed and thorough instructions to help the LLM select the experts better.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

ROUTE_FD = """Create and output a child prompt for an LLM to choose some experts best suited for resolving some general reasoning problems related to math, hard science, humanities, social sciences, etc.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to select more useful and effective experts for problem resolution.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose experts more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

# ---------------- dispatch：新部件，按 OMAC 模板的格式写 ----------------
DISPATCH_INIT = """Create {} distinct prompts for an LLM that coordinates a team of expert agents. For some general reasoning problems related to math, hard science, humanities, social sciences, etc., the LLM writes the message sent to one expert. The expert only sees this message and must resolve the problem from it.
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should help the LLM write a message that leads the expert to resolve the problem accurately and efficiently.
Note that the problem and the name and function description of the expert have been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for writing the message.
You may consider adding more detailed and thorough instructions to help the LLM write better messages.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

DISPATCH_FD = """Create and output a child prompt for an LLM that coordinates a team of expert agents and writes the message sent to one expert for resolving some general reasoning problems related to math, hard science, humanities, social sciences, etc. The expert only sees this message.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to write messages that lead the experts to resolve the problems.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to write messages more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

# ---------------- aggregate：改写自 OMAC 的 RANK_INIT / RANK_FD ----------------
AGG_INIT = """Create {} different prompts for an LLM to decide the final answer to some general reasoning problems related to math, hard science, humanities, social sciences, etc., by examining the solutions from some expert agents.
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
The prompt should help to accurately and efficiently decide the final answer that resolves the given problems best.
Note that all the solutions and the problem have been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for deciding the final answer.
You may consider adding more detailed and thorough instructions to help the LLM decide the final answer better.
The instruction on the answer format will be appended after the prompt automatically, so the prompt does not need to specify it.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

AGG_FD = """Create and output a child prompt for an LLM to decide the final answer to some general reasoning problems (related to math, hard science, humanities, social sciences, etc.) by examining the solutions from some expert agents.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to decide the correct final answer.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to decide the final answer more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""


def templates(component: str, example: str, role: str = "Mathematician"):
    if component == "worker":
        return ROLE_INIT.format(INIT_NUM - 1, role, example, example), ROLE_FD.format(role, example)
    init, fd = {"route": (ROUTE_INIT, ROUTE_FD), "dispatch": (DISPATCH_INIT, DISPATCH_FD),
                "aggregate": (AGG_INIT, AGG_FD)}[component]
    return init.format(INIT_NUM - 1, example), fd


def with_candidate(cfg: dict, component: str, prompt: str, role: str = "Mathematician") -> dict:
    new = {**cfg, "workers": dict(cfg["workers"])}
    if component == "worker":
        new["workers"][role] = prompt
    else:
        new[component] = prompt
    return new


def optimize(component: str, base_cfg: dict, train: list[dict], out_dir: str, role: str = "Mathematician",
             seed: int = 0, threads: int = 60) -> dict:
    """返回 {"candidates", "scores", "team_scores", "local_scores", "chosen", "chosen_nonbase", ...}，并写日志。"""
    os.makedirs(out_dir, exist_ok=True)
    # 文件名与 salt 用 tag：Mathematician 沿用 "worker"（与第一轮已有结果一致），其他 worker 用 "worker_<角色>"
    tag = component if (component != "worker" or role == "Mathematician") else f"worker_{role}"
    rng = random.Random(f"{tag}-{seed}")
    salt = f"opt-{tag}-{seed}"
    example = base_cfg["workers"][role] if component == "worker" else base_cfg[component]
    init_sys, fd_sys = templates(component, example, role)
    cands, scores, team_scores, local_scores, log = [], {}, {}, {}, []

    def evaluate(pid: int):
        cfg = with_candidate(base_cfg, component, cands[pid], role)
        recs = team.evaluate(train, cfg, threads=threads)
        with open(os.path.join(out_dir, f"train_{tag}_{pid}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        team_scores[pid] = team.accuracy(recs)
        local_scores[pid] = sum(r["worker_correct"][role] for r in recs) / len(recs)
        scores[pid] = local_scores[pid] if component == "worker" else team_scores[pid]
        print(f"[{tag}] cand {pid}: score={scores[pid]:.4f} team={team_scores[pid]:.4f} "
              f"local={local_scores[pid]:.4f}", flush=True)

    # Semantic Initializer
    cands.append(example)
    dialog = [{"role": "system", "content": init_sys}]
    for i in range(INIT_NUM - 1):
        dialog.append({"role": "user", "content": f"Directly output the {i + 1}th prompt without any prefix or explanation."})
        out = team.call(dialog, temperature=OPT_TEMPERATURE, max_tokens=OPT_MAX_TOKENS, salt=salt).rstrip("\n")
        dialog.append({"role": "assistant", "content": out})
        cands.append(out)
    log.append({"stage": "init", "dialog": dialog})
    with ThreadPoolExecutor(len(cands)) as ex:  # 初始候选互不依赖，并行评估（不影响结果）
        list(ex.map(evaluate, range(len(cands))))

    # Contrastive Comparator（FD）
    tolerance, max_score = 0, max(scores.values())
    for t in range(FD_MAX_ITER):
        thr = FD_SAMPLE_THRESHOLD + t
        if len(cands) < 2 * thr:
            thr = len(cands) // 2
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        pos = ranked[0][0]
        neg = ranked[len(cands) - 1 - rng.randrange(thr)][0]
        sys_p = fd_sys + '\n\nThe positive parent prompt is: "{}"'.format(cands[pos]) + \
            '\n\nThe negative parent prompt is: "{}"'.format(cands[neg])
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": "Directly output the content of the child prompt without any prefix or explanation."}]
        child = team.call(msgs, temperature=OPT_TEMPERATURE, max_tokens=OPT_MAX_TOKENS,
                          salt=f"{salt}-fd{t}").rstrip("\n")  # 每轮独立采样（避免同一对父本命中缓存）
        cands.append(child)
        log.append({"stage": f"fd{t}", "positive": pos, "negative": neg, "messages": msgs, "child": child})
        evaluate(len(cands) - 1)
        if max(scores.values()) - max_score < FD_MIN_IMPROVEMENT:
            tolerance += 1
        max_score = max(scores.values())
        if tolerance >= FD_TOLERANCE_ITER:
            break

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_team = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)
    res = {"component": component, "tag": tag, "role": role if component == "worker" else None,
           "candidates": cands, "scores": scores, "team_scores": team_scores, "local_scores": local_scores,
           "score_kind": "local (worker's own train accuracy)" if component == "worker" else "team train accuracy",
           "chosen": ranked[0][0], "chosen_nonbase": next(pid for pid, _ in ranked if pid != 0),
           "chosen_by_team_score": ranked_team[0][0], "log": log}
    json.dump(res, open(os.path.join(out_dir, f"opt_{tag}.json"), "w"), ensure_ascii=False, indent=1)
    return res
