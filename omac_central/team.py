"""中心化团队（1 个 master + 3 个 worker），在 OMAC 的 MMLU 设定上加一个 master。

每道题的流程：
  1. 选人（route）：master 读题和三个 worker 的角色说明，选 1–3 个 worker
  2. 分派（dispatch）：master 为每个 worker 单独写一段消息；worker 只看得到这段消息
     三个 worker 都会被分派、都会作答（影子调用），但只有被选中的人的回答进入汇总。
     这样改选人时不会顺带改掉子指令，各 worker 的输出可以在不同条件之间复用，
     也能在每道题上测每个 worker 自己的正确率。
  3. 作答：worker 的系统提示词 = 角色提示词（OMAC 的 ROLE_MAP，可被优化）+ OMAC 的作答说明
  4. 汇总（aggregate）：master 读被选中 worker 的回答，给出团队答案
与 OMAC 的差别：
  - master 看到的 worker 说明固定为 OMAC 原始角色描述，不随 worker 提示词的优化而变
    （OMAC 的 pre-rank 会看到优化后的提示词）。这样 worker 只能通过自己的回答影响 master，
    隔离出"master 改动 → worker 收到的内容"这一条通道。
  - 团队内所有调用温度为 0（OMAC 的 agent 温度为 1.0），降低噪声；优化器保持 OMAC 的 1.0。
所有调用按消息内容缓存在 sqlite 中：同一部件、同一输入的调用在各条件之间自动复用；
salt 不同则强制重新生成（用于测量重跑噪声）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
for line in open("/home/claude/chime/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)

MODEL = "deepseek-chat"
PRICE = (0.02, 1.0, 4.0)  # 元 / 百万 token（闲时）：缓存命中输入、未命中输入、输出；高峰翻倍

# ---------------- 高峰时段 ----------------


def is_peak(now: datetime | None = None) -> bool:
    bj = (now or datetime.now(timezone.utc)).astimezone(timezone(timedelta(hours=8)))
    hm = bj.hour * 100 + bj.minute
    return bj.isoweekday() <= 5 and (900 <= hm < 1200 or 1400 <= hm < 1800)


def wait_offpeak():
    """高峰时段（且未设 ALLOW_PEAK=1）时阻塞，直到闲时。长任务跨入高峰时会自动暂停。"""
    warned = False
    while is_peak() and os.environ.get("ALLOW_PEAK") != "1":
        if not warned:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] peak hours: pausing new API calls", flush=True)
            warned = True
        time.sleep(60)


# ---------------- 带缓存的调用 ----------------
_DB = os.path.join(ROOT, "cache.sqlite")
_LOCK = threading.Lock()
_CONN = sqlite3.connect(_DB, check_same_thread=False, timeout=60)
_CONN.execute("PRAGMA journal_mode=WAL")
_CONN.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, content TEXT, usage TEXT, ts REAL)")
_CONN.commit()
COST = {"calls": 0, "cached": 0, "cny": 0.0}
_INFLIGHT = threading.BoundedSemaphore(int(os.environ.get("MAX_INFLIGHT", "64")))  # 同时在途的 API 请求上限


def call(messages: list[dict], temperature: float = 0.0, max_tokens: int = 1500, salt: str = "") -> str:
    key = hashlib.sha256(json.dumps([MODEL, temperature, max_tokens, messages, salt], ensure_ascii=False)
                         .encode()).hexdigest()
    with _LOCK:
        row = _CONN.execute("SELECT content FROM c WHERE k=?", (key,)).fetchone()
        if row is not None:
            COST["cached"] += 1
            return row[0]
    wait_offpeak()
    url = os.environ["DEEPSEEK_BASE_URL"].rstrip("/") + "/chat/completions"
    body = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}
    last = None
    for i in range(8):
        try:
            with _INFLIGHT:
                r = requests.post(url, json=body, headers=headers, timeout=300)
            if r.status_code == 200:
                d = r.json()
                content = d["choices"][0]["message"]["content"] or ""
                u = d.get("usage", {})
                u["finish_reason"] = d["choices"][0].get("finish_reason")
                u["peak"] = is_peak()
                hit = u.get("prompt_cache_hit_tokens", 0)
                miss = u.get("prompt_tokens", 0) - hit
                cny = (2.0 if u["peak"] else 1.0) * (hit * PRICE[0] + miss * PRICE[1]
                                                     + u.get("completion_tokens", 0) * PRICE[2]) / 1e6
                u["cny"] = cny
                with _LOCK:
                    _CONN.execute("INSERT OR REPLACE INTO c VALUES (?,?,?,?)", (key, content, json.dumps(u), time.time()))
                    _CONN.commit()
                    COST["calls"] += 1
                    COST["cny"] += cny
                return content
            last = f"{r.status_code} {r.text[:200]}"
            if r.status_code in (400, 401, 402, 403):
                raise RuntimeError(f"DeepSeek {last}")
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(min(3 * 2 ** i, 60))
    raise RuntimeError(f"DeepSeek request failed: {last}")


# ---------------- 解析（OMAC utils.parse_single_choice / parse_ranks 的等价实现） ----------------


def parse_single_choice(reply: str):
    """OMAC 的 parse_single_choice 只认 A–D（大小写均可）；MMLU-Pro 有 A–J 十个选项。
    小写 (i)/(a) 常出现在推理里的分点编号，所以这里只认大写；仍取最后一个匹配。"""
    m = re.findall(r"\(([A-J])\)", reply or "")
    if m:
        return m[-1]
    m = re.findall(r"([A-J])\)", reply or "")
    return m[-1] if m else None


def parse_ranks(reply: str, n: int) -> list[int]:
    for grp in reversed(re.findall(r"\[([\d,\s]+)\]", reply or "")):
        idx = []
        for x in grp.split(","):
            x = x.strip()
            if x.isdigit() and 1 <= int(x) <= n and int(x) - 1 not in idx:
                idx.append(int(x) - 1)
        if idx:
            return sorted(idx)
    return []


# ---------------- 团队 ----------------
WORKERS = ["Mathematician", "Doctor", "Lawyer"]
# OMAC prompt_lib.ROLE_MAP 原文
ROLE_MAP = {
    "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
    "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient's age, lifestyle and medical history when providing your recommendations.",
    "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
}
# OMAC 的 worker 系统提示词 = ROLE_MAP[role] + "\n" + SYSTEM_PROMPT_MMLU（"Here's a debate. Explain your reasons at
# each round thoroughly.\nAll questions are single choice."）；这里没有多轮辩论，去掉辩论措辞。
WORKER_SYSTEM_SUFFIX = "\nExplain your reasons thoroughly.\nAll questions are single choice."
LETTERS = "ABCDEFGHIJ"


def answer_suffix(item) -> str:
    """OMAC：'Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D).'
    按题目的选项数扩展到 (A)–(J)。"""
    ls = [f"({c})" for c in LETTERS[:len(item["options"])]]
    return ("\n\nPut your answer in the form (X) at the end of your response. (X) represents choice "
            + ", ".join(ls[:-1]) + ", or " + ls[-1] + ".")


def qtext(item) -> str:
    """OMAC 的 MMLU_QUESTION：'Can you answer the following question as accurately as possible? {}: A) {}, B) {}, ...'"""
    opts = ", ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return f"Can you answer the following question as accurately as possible? {item['question']}: {opts} "


MASTER_SYSTEM = ("You are the coordinator of a team of expert agents. You decide which experts to consult, "
                 "write the message sent to each expert, and combine the experts' answers into the team's final answer.")

# 选人：改写自 OMAC 的 PRE_RANK_DEFAULT / PRE_RANK_FORMAT（候选从 4–7 个 agent 改为 1–3 个 expert）
ROUTE_DEFAULT = ("Take functionality, efficiency, and necessity into consideration, choose the experts best suited for "
                 "resolving the given problem. You may choose 1 to 3 experts. Think it step by step. Put your answer in "
                 "the form like [1] or [1,3] or [1,2,3] at the end of your response.")
ROUTE_FORMAT = ("Please ensure to specify the choice of experts strictly in the format of a single list as described above "
                "(like [1] or [2,3] or [1,2,3]). The elements of this list should represent the sequence numbers of all "
                "selected experts. Avoid expressing your choice in the format like '[expert 1] and [expert 2]'. Also, "
                "ensure that anything within '[]' consists solely of sequence numbers of experts, separated by commas, "
                "with no additional text like 'expert'.")
# 分派：OMAC 没有这个部件，按同样风格新写
DISPATCH_DEFAULT = ("Write the message that you will send to this expert. The message must contain the complete question "
                    "and all answer options exactly as given. Ask the expert to solve the question and explain its "
                    "reasoning.")
# 汇总：改写自 OMAC 的 RANK_DEFAULT（从"选出最好的 2 个解"改为"给出团队答案"）
AGG_DEFAULT = ("Examine the experts' solutions step by step. Notice that their answers might all be wrong. Decide the "
               "final answer of the team and explain your reasoning.")


def base_config() -> dict:
    return {"route": ROUTE_DEFAULT, "dispatch": DISPATCH_DEFAULT, "aggregate": AGG_DEFAULT,
            "workers": dict(ROLE_MAP)}


def _experts_block() -> str:
    return "".join(f"\nExpert {i+1}: Its name is: {w}\nIts function description is:\n{ROLE_MAP[w]}"
                   for i, w in enumerate(WORKERS))


def route(item, cfg, salt=""):
    user = ("Here is the task and question:\n" + qtext(item) + "\n\nThese are the experts and their functional "
            "description: " + _experts_block() + "\n\n" + cfg["route"] + "\n" + ROUTE_FORMAT)
    out = call([{"role": "system", "content": MASTER_SYSTEM}, {"role": "user", "content": user}],
               max_tokens=1000, salt=salt)
    return out, parse_ranks(out, len(WORKERS))


def dispatch(item, cfg, worker, salt=""):
    user = ("Here is the question:\n" + qtext(item) + f"\n\nYou will send a message to the following expert.\n"
            f"Its name is: {worker}\nIts function description is:\n{ROLE_MAP[worker]}\n\n" + cfg["dispatch"] +
            "\n\nOutput only the message to send to the expert.")
    return call([{"role": "system", "content": MASTER_SYSTEM}, {"role": "user", "content": user}],
                max_tokens=1000, salt=salt)


def work(item, cfg, worker, message, salt=""):
    return call([{"role": "system", "content": cfg["workers"][worker] + WORKER_SYSTEM_SUFFIX},
                 {"role": "user", "content": message + answer_suffix(item)}], max_tokens=2000, salt=salt)


def aggregate(item, cfg, chosen: list[tuple[str, str]], salt=""):
    sols = "".join(f"\n\nExpert {w}'s solution: ```{r}```" for w, r in chosen)
    user = ("Here is the question:\n" + qtext(item) + "\n\nThese are the solutions from the experts you consulted:" +
            sols + "\n\n" + cfg["aggregate"] + answer_suffix(item))
    return call([{"role": "system", "content": MASTER_SYSTEM}, {"role": "user", "content": user}],
                max_tokens=1500, salt=salt)


def _salt(salt, part):
    """salt 可以是字符串（所有部件同一个）或 dict（按部件指定，键：route/dispatch/aggregate/各 worker 名）。"""
    return salt.get(part, "") if isinstance(salt, dict) else salt


def run_item(item, cfg, salt="") -> dict:
    with ThreadPoolExecutor(4) as ex:
        f_route = ex.submit(route, item, cfg, _salt(salt, "route"))

        def chain(w):
            m = dispatch(item, cfg, w, _salt(salt, "dispatch"))
            return m, work(item, cfg, w, m, _salt(salt, w))
        f_chain = {w: ex.submit(chain, w) for w in WORKERS}
        r_out, sel = f_route.result()
        msgs = {w: f_chain[w].result()[0] for w in WORKERS}
        resp = {w: f_chain[w].result()[1] for w in WORKERS}
    fallback = not sel
    sel = sel or list(range(len(WORKERS)))
    chosen = [WORKERS[i] for i in sel]
    a_out = aggregate(item, cfg, [(w, resp[w]) for w in chosen], _salt(salt, "aggregate"))
    final = parse_single_choice(a_out)
    wans = {w: parse_single_choice(resp[w]) for w in WORKERS}
    return {"id": item["id"], "domain": item["domain"], "subject": item["subject"], "gold": item["answer"],
            "selected": chosen, "route_fallback": fallback, "messages": msgs, "worker_resp": resp,
            "worker_ans": wans, "worker_correct": {w: wans[w] == item["answer"] for w in WORKERS},
            "route_out": r_out, "agg_out": a_out, "final": final, "correct": final == item["answer"]}


def evaluate(items, cfg, salt="", threads=16) -> list[dict]:
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(lambda it: run_item(it, cfg, salt), items))


def accuracy(recs) -> float:
    return sum(r["correct"] for r in recs) / len(recs) if recs else 0.0


def load_data(name="mmlu_pro_central", n_test=30):
    """n_test：每类取前 n_test 道测试题（数据文件里每类有 60 道）。"""
    d = json.load(open(os.path.join(ROOT, "data", f"{name}.json")))
    if name == "mmlu_pro_central":
        d["test"] = [it for it in d["test"] if it["idx"] < n_test]
    return d
