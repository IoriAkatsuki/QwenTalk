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


def _accumulate_tool_call_delta(tool_calls_acc: list, tc_delta: dict) -> None:
    """SSE 增量更新 tool_calls 数组（按 delta.tool_calls[i].index 索引）。"""
    idx = tc_delta.get("index", 0)
    while len(tool_calls_acc) <= idx:
        tool_calls_acc.append({"function": {"arguments": ""}})
    tc = tool_calls_acc[idx]
    if "id" in tc_delta:
        tc["id"] = tc_delta["id"]
    if "type" in tc_delta:
        tc["type"] = tc_delta["type"]
    fn_delta = tc_delta.get("function") or {}
    if "name" in fn_delta:
        tc["function"]["name"] = fn_delta["name"]
    if "arguments" in fn_delta:
        tc["function"]["arguments"] += fn_delta["arguments"]


def _stream_one_round(
    messages: list, system: str | None, max_tokens: int,
    disable_thinking: bool | None,
):
    """G: streaming + tools 单轮 — 边读 SSE delta 边判 tool_call vs content。

    省掉 detect non-stream 整轮（之前 11s warm 首字 → 现在 ~2-5s 首字）。
    Yields:
      ('sentence', sent, content_full) — 流式句子
      ('__final__', {finish_reason, tool_calls, content}, content) — 一轮终态
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {
        "messages": msgs, "tools": TOOL_SCHEMAS, "tool_choice": "auto",
        "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
        **_maybe_thinking_kwarg(disable_thinking),
    }
    tool_calls_acc: list[dict] = []
    content_full, content_buf = "", ""
    finish_reason: str | None = None

    with requests.post(LLM_URL, json=payload, stream=True, timeout=ROUND_TIMEOUT_S) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                j = json.loads(data)
                choice = j["choices"][0]
                delta = choice.get("delta", {})
                for tc_d in (delta.get("tool_calls") or []):
                    _accumulate_tool_call_delta(tool_calls_acc, tc_d)
                if delta.get("content"):
                    content_full += delta["content"]
                    content_buf += delta["content"]
                    while True:
                        m = SENTENCE_BREAK.search(content_buf)
                        if not m:
                            break
                        end = m.end()
                        sent = content_buf[:end].strip()
                        content_buf = content_buf[end:]
                        if sent:
                            yield ("sentence", sent, content_full)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    if content_buf.strip():
        yield ("sentence", content_buf.strip(), content_full)
    yield ("__final__", {
        "finish_reason": finish_reason,
        "tool_calls": tool_calls_acc,
        "content": content_full,
    }, content_full)


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
        final_info: dict | None = None
        try:
            for evt in _stream_one_round(messages, system, max_tokens, disable_thinking):
                if evt[0] == "__final__":
                    final_info = evt[1]
                    break
                yield evt  # passthrough sentence
        except requests.RequestException as e:
            yield ("error", f"iter {iteration + 1} failed: {e}", None)
            return

        tool_calls = (final_info or {}).get("tool_calls") or []
        content = (final_info or {}).get("content", "") or ""
        finish = (final_info or {}).get("finish_reason")

        if finish == "tool_calls" and tool_calls:
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": tool_calls,
            })
            yield from _yield_tool_calls(tool_calls, messages)
            continue

        # content 模式：sentence 已经流式 yield 过；写回 history + done
        if content.strip():
            messages.append({"role": "assistant", "content": content})
        yield ("done", None, content)
        return

    # MAX_ITERATIONS 耗尽仍未自然回答 — fail loud
    last_tool = next((m.get("name", "工具") for m in reversed(messages) if m.get("role") == "tool"), "工具")
    msg = (f"我尝试调用了 {MAX_ITERATIONS} 次工具（最后一次 {last_tool}）"
           f"但仍没能找到合适的答案，请换个问法或稍后再试。")
    yield ("sentence", msg, msg)
    yield ("done", None, msg)
