"""最小的 DeepSeek 客户端（OpenAI 兼容接口），带重试与 token 计数。密钥从 .env 读取。"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    env = _ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def chat(messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.0,
         max_tokens: int = 4096, json_mode: bool = False, retries: int = 6) -> str:
    """论文 Table 7：temperature 0，失败重试。"""
    url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}
    for i in range(retries):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=300)
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage", {})
                USAGE["calls"] += 1
                USAGE["prompt_tokens"] += u.get("prompt_tokens", 0)
                USAGE["completion_tokens"] += u.get("completion_tokens", 0)
                return d["choices"][0]["message"]["content"] or ""
            if r.status_code in (400, 401, 402, 403):
                raise RuntimeError(f"DeepSeek {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            pass
        time.sleep(min(5 * 2 ** i, 60))
    raise RuntimeError("DeepSeek request failed after retries")


def parse_json(text: str) -> dict:
    """容错解析：去掉代码块围栏，取第一个完整 JSON 对象。"""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise
