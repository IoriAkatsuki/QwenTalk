"""LLM tool calling for /ws/chat — Qwen3.6 + llama.cpp ReAct loop。

chat_with_tools_stream(prompt, system, max_tokens, disable_thinking, history)
  → yields (kind, payload, full_so_far)
    kind ∈ {tool_call, tool_result, sentence, done, error}
流程: detect (tools=auto) → 若 tool_calls 执行 → 再 detect；MAX_ITERATIONS 防死循环。
history: list | None — WS connection 级会话历史；in-place append。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import requests

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from tools import get_system_info, get_temperature  # noqa: E402
from tool_schemas import TOOL_SCHEMAS as _ALL_SCHEMAS  # noqa: E402 — 单一 schema 源
from voice_pipeline import LLM_URL, SENTENCE_BREAK, _iter_sse_deltas  # noqa: E402

ROUND_TIMEOUT_S = 150.0  # 每轮 detect 超时；含 system prompt 504 字 + tools schema + tool_result 累积
WTTR_TIMEOUT_S = 8.0
BING_TIMEOUT_S = 8.0
BING_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
MAX_ITERATIONS = 4  # ReAct loop 硬上限：模型 tool_call 最多 4 次防死循环
HISTORY_MAX_CHARS = 2500  # L0 history 截窗阈值（≈ 1000-2500 tokens，给 Qwen3.6 4096 留 buffer）
HISTORY_KEEP_TAIL = 6  # 截窗时保留最近 K 条 + 当前 user


def web_search(query: str, max_results: int = 3) -> dict:
    """Bing 中国版 HTML 抓取（板卡国内可达；DDG/SearXNG 都不可用的 fallback）。"""
    import re
    import urllib.parse
    import urllib.request
    try:
        url = f"https://cn.bing.com/search?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": BING_UA})
        html = urllib.request.urlopen(req, timeout=BING_TIMEOUT_S).read()
        html = html.decode("utf-8", errors="ignore")
        blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
        results = []
        for b in blocks[:max_results]:
            h2 = re.search(r"<h2[^>]*>(.*?)</h2>", b, re.S)
            p = re.search(r"<p[^>]*>(.*?)</p>", b, re.S)
            raw_title = h2.group(1) if h2 else ""
            raw_snippet = p.group(1) if p else ""
            title = re.sub(r"<[^>]+>", "", raw_title)
            title = re.sub(r"&ensp;|&nbsp;|&amp;", " ", title)
            title = re.sub(r"\s+", " ", title).strip()
            snippet = re.sub(r"<[^>]+>", "", raw_snippet)
            snippet = re.sub(r"&ensp;|&#0183;|&nbsp;|&amp;", " ", snippet)
            snippet = re.sub(r"\s+", " ", snippet)[:200]
            if snippet or title:
                results.append({"title": title[:80], "snippet": snippet})
        if not results:
            return {"error": "Bing 未返回结果"}
        return {"query": query, "results": results, "source": "cn.bing.com"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Bing 失败: {type(e).__name__}: {e}"}


def get_weather(city: str = "Beijing") -> dict:
    """wttr.in 天气查询（无 KEY，板卡国内网络可达）。"""
    try:
        r = requests.get(
            f"https://wttr.in/{city}?format=j1", timeout=WTTR_TIMEOUT_S,
            headers={"User-Agent": "curl/7.0"},
        )
        r.raise_for_status()
        d = r.json()
        c = d["current_condition"][0]
        return {
            "city": city,
            "temp_c": c["temp_C"],
            "feels_like_c": c["FeelsLikeC"],
            "desc": c["weatherDesc"][0]["value"],
            "humidity_pct": c["humidity"],
            "wind_kmh": c.get("windspeedKmph", "n/a"),
            "source": "wttr.in",
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"wttr.in 失败: {type(e).__name__}: {e}"}


TOOLS = {"web_search": web_search, "get_weather": get_weather,
         "get_temperature": get_temperature, "get_system_info": get_system_info}
_SCHEMA_BY_NAME = {s["function"]["name"]: s for s in _ALL_SCHEMAS}
TOOL_SCHEMAS = [_SCHEMA_BY_NAME[name] for name in TOOLS if name in _SCHEMA_BY_NAME]
_missing = [n for n in TOOLS if n not in _SCHEMA_BY_NAME]
assert not _missing, f"tool_schemas.py 缺 schema: {_missing}"


def _maybe_thinking_kwarg(disable_thinking: bool | None) -> dict:
    if disable_thinking is True:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


DETECT_TOKEN_BUDGET = 32  # detect 阶段只需 1-32 token 就够判断 tool_call vs content


def _detect_once(
    messages: list, system: str | None, max_tokens: int,
    disable_thinking: bool | None,
) -> dict:
    """单轮 non-streaming detect；带 tools=auto 让模型自选 call 或答。

    关键优化：max_tokens=DETECT_TOKEN_BUDGET，不让 detect 浪费 token 生成完整答案。
    - 如果 tool_call：第 1 个 token 就是 special token，立即返回 finish_reason=tool_calls
    - 如果 content：前几十字仅作为"模型决定回答"的信号，最终回答由 _stream_final_answer 重生成
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {
        "messages": msgs,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "max_tokens": DETECT_TOKEN_BUDGET,
        "temperature": 0.7,
        "stream": False,
        **_maybe_thinking_kwarg(disable_thinking),
    }
    # max_tokens 在外层（chat_with_tools_stream 传入）保留给 _stream_final_answer 使用
    _ = max_tokens
    r = requests.post(LLM_URL, json=payload, timeout=ROUND_TIMEOUT_S)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.json()


def _stream_final_answer(
    messages: list, system: str | None, max_tokens: int,
    disable_thinking: bool | None, fallback_content: str,
):
    """检测到 tool_calls=空（模型要回答）时，用 streaming 重新 call 取真流式。

    KV cache 命中：prompt 部分 ~0 cost，只重做 generation。
    不带 tools 字段防止模型又 tool_call（之前实测带 tool_choice=none 反而拖慢）。
    若 streaming 失败/无输出，回退到 detect 时拿到的 fallback_content。
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        **_maybe_thinking_kwarg(disable_thinking),
    }
    full, buf, received = "", "", False
    try:
        for delta in _iter_sse_deltas(payload):
            received = True
            buf += delta
            full += delta
            while True:
                m = SENTENCE_BREAK.search(buf)
                if not m:
                    break
                end = m.end()
                sent = buf[:end].strip()
                buf = buf[end:]
                if sent:
                    yield ("sentence", sent, full)
        if buf.strip():
            yield ("sentence", buf.strip(), full)
    except Exception as e:  # noqa: BLE001 — streaming 失败回退 non-stream content
        full = fallback_content
        received = bool(full.strip())
        for sent in _split_sentences(full):
            yield ("sentence", sent, full)
        if not received:
            yield ("error", f"streaming + fallback 都失败: {e}", None)
            return
    if not received or not full.strip():
        for sent in _split_sentences(fallback_content):
            yield ("sentence", sent, fallback_content)
        full = fallback_content
    yield ("done", None, full)


def _exec_tool(name: str, args_json: str) -> dict:
    """执行 tool 函数，捕异常成 error dict。"""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(args_json) if args_json else {}
        return fn(**args)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _split_sentences(text: str):
    """按句切；用于把 non-stream content 模拟流式 yield。"""
    if not text:
        return
    buf = text
    while True:
        m = SENTENCE_BREAK.search(buf)
        if not m:
            break
        end = m.end()
        sent = buf[:end].strip()
        buf = buf[end:]
        if sent:
            yield sent
    if buf.strip():
        yield buf.strip()


def _yield_tool_calls(tool_calls: list, messages: list):
    """执行所有 tool_calls，append 到 messages，yield tool_call/tool_result 事件。"""
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")
        yield ("tool_call", {"name": name, "arguments": args}, "")
        result = _exec_tool(name, args)
        yield ("tool_result", {"name": name, "result": result}, "")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        })


def _trim_history(messages: list, max_chars: int = HISTORY_MAX_CHARS) -> None:
    """L0 history 截窗：in-place 丢老消息，保留末尾 K 条 + 末尾 user。system 由外层注入不受影响。"""
    size = lambda: sum(len(str(m.get("content") or "")) for m in messages)  # noqa: E731
    if size() <= max_chars:
        return
    if len(messages) > HISTORY_KEEP_TAIL:
        del messages[: len(messages) - HISTORY_KEEP_TAIL]
    while size() > max_chars and len(messages) > 1:
        messages.pop(0)


def chat_with_tools_stream(
    prompt: str, system: str | None = None,
    max_tokens: int = 500, disable_thinking: bool | None = None,
    history: list | None = None,
) -> Generator:
    """ReAct loop: detect → tool_call → exec → re-detect → ... → final answer。

    history: 外部维护的会话历史 list（不含 system；仅 user/assistant/tool）。
      None → single-shot；传 list → in-place append user/assistant/tool_result 供 caller 复用。
    """
    messages: list = history if history is not None else []
    messages.append({"role": "user", "content": prompt})
    _trim_history(messages)

    for iteration in range(MAX_ITERATIONS):
        try:
            response = _detect_once(messages, system, max_tokens, disable_thinking)
        except requests.RequestException as e:
            yield ("error", f"iter {iteration + 1} failed: {e}", None)
            return

        choice = response["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # 模型决定回答 → 用 streaming 二次 call 拿真流式体验。
            # KV cache 命中：prompt 部分 0 cost，只有 generation 是新成本。
            final_text = ""
            for evt in _stream_final_answer(
                messages, system, max_tokens, disable_thinking,
                fallback_content=msg.get("content") or "",
            ):
                if evt[0] == "done":
                    final_text = evt[2] or final_text
                yield evt
            # L0 history: 把 assistant 最终回答写回 messages，供下一轮上下文
            if final_text.strip():
                messages.append({"role": "assistant", "content": final_text})
            return

        # 有 tool_call → append assistant + 执行 tools → 进入下一轮
        messages.append({
            "role": "assistant", "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })
        yield from _yield_tool_calls(tool_calls, messages)

    # MAX_ITERATIONS 耗尽仍未自然回答 — fail loud
    last_tool = next((m.get("name", "工具") for m in reversed(messages) if m.get("role") == "tool"), "工具")
    msg = (f"我尝试调用了 {MAX_ITERATIONS} 次工具（最后一次 {last_tool}）"
           f"但仍没能找到合适的答案，请换个问法或稍后再试。")
    yield ("sentence", msg, msg)
    yield ("done", None, msg)
