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

from tool_schemas import TOOL_SCHEMAS as _ALL_SCHEMAS  # noqa: E402 — 单一 schema 源
from voice_pipeline import LLM_URL, SENTENCE_BREAK, _iter_sse_deltas  # noqa: E402
from . import tool_impls  # noqa: E402 — 9 个工具实现拆离，避免本文件撑爆 300

ROUND_TIMEOUT_S = 150.0  # 每轮 detect 超时；含 system prompt 504 字 + tools schema + tool_result 累积
MAX_ITERATIONS = 4  # ReAct loop 硬上限：模型 tool_call 最多 4 次防死循环
HISTORY_MAX_CHARS = 2500  # L0 history 截窗阈值（≈ 1000-2500 tokens，给 Qwen3.6 4096 留 buffer）
HISTORY_KEEP_TAIL = 6  # 截窗时保留最近 K 条 + 当前 user


TOOLS = {
    # 网络 / 外部
    "web_search": tool_impls.web_search,
    "get_weather": tool_impls.get_weather,
    # 系统状态
    "get_temperature": tool_impls.get_temperature,
    "get_system_info": tool_impls.get_system_info,
    # D435 + NPU 感知
    "get_gesture": tool_impls.get_gesture,
    "get_distance": tool_impls.get_distance,
    "get_scene": tool_impls.get_scene,
    "identify_user": tool_impls.identify_user,
    # TTS / 主动输出
    "speak": tool_impls.speak,
    # L2 长期记忆
    "memory_store": tool_impls.memory_store,
    "memory_recall": tool_impls.memory_recall,
}
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
