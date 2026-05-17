"""WS /ws/chat 单次请求处理 — 抽离自 server.py 以保持 server.py ≤ 300 行。

调用：await stream_chat_to_ws(ws, req)
请求：{text, system?, max_tokens?, thinking?: on|off|auto}
回包 type:
  sentence | tool_call | tool_result | done | error

事件流是 chat_tools.chat_with_tools_stream 的 4 类事件 +
sentence 的二次封装。
"""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import WebSocket

from .memory import MemoryStore

ROUND_TIMEOUT_S = 180.0
# 板卡默认路径；本机/CI 覆盖用 env INTEL_CHAN_MEMORY_DB。
_DEFAULT_DB = os.environ.get("INTEL_CHAN_MEMORY_DB",
                             "/home/intel/QwenTalk/memory.db")
_memory_store = MemoryStore(_DEFAULT_DB)
# 当前 server 进程的 session_id（server.py lifespan 注入）。
_current_session_id: int | None = None

# perception 注入器：server.py lifespan 注册，让 system prompt 拿到当前感知状态。
_perception_provider = None  # type: ignore[assignment]


def set_perception_provider(provider) -> None:
    """server.py 启动时注入 PerceptionFusion 实例。无注入则 system prompt 退化为纯时段感知。"""
    global _perception_provider
    _perception_provider = provider


def set_session_id(sid: int | None) -> None:
    """server.py lifespan 调：设置/清除当前会话 id（写 turns 时绑定）。"""
    global _current_session_id
    _current_session_id = sid


def get_memory_store() -> MemoryStore:
    """供 server.py lifespan / 测试访问全局 store。"""
    return _memory_store


def _default_system() -> str:
    """用 intel_chan_persona 动态构造 system prompt（接入 perception 当前状态）。"""
    import sys
    from pathlib import Path
    proj = Path(__file__).resolve().parent.parent
    if str(proj) not in sys.path:
        sys.path.insert(0, str(proj))
    from intel_chan_persona import IntelChanContext, build_system_prompt

    ctx = IntelChanContext()
    if _perception_provider is not None:
        try:
            snap = _perception_provider.snapshot()
            ctx.distance_m = float(getattr(snap, "distance_m", -1.0))
            emo = getattr(snap, "emotion", None)
            if emo is not None:
                ctx.emotion_label = getattr(emo, "label", None)
        except Exception:  # noqa: BLE001 — 拿不到就退化
            pass
    # L2: 注入 persona facts 到 system prompt
    try:
        facts = _memory_store.get_persona_facts(top_n=5)
        ctx.recent_memories = [f"{f['key']}: {f['value']}" for f in facts]
    except Exception:  # noqa: BLE001 — DB 异常不阻塞对话
        pass
    return build_system_prompt(ctx)


def _parse_chat_req(req: dict) -> tuple[str, str, int, object]:
    """从客户端请求提取 (text, system, max_tokens, disable_thinking)。"""
    from voice_pipeline import detect_thinking_support  # lazy: 板卡 only

    text = str(req.get("text", "")).strip()
    system = req.get("system") or _default_system()
    max_tokens = int(req.get("max_tokens", 200))
    disable_thinking = detect_thinking_support(req.get("thinking", "auto"))
    return text, system, max_tokens, disable_thinking


def _make_chat_producer(loop, q, text, system, max_tokens, disable_thinking, history):
    """构造跑在 executor 的 producer — 转发 chat_with_tools_stream 全部事件。

    chat_tools yields (kind, payload, full_so_far) 直接入队，由消费端按 kind 分发。
    history: WS 级会话历史；in-place append by chat_with_tools_stream。
    捕异常成 error 帧避免消费端挂死。
    """
    from .chat_tools import chat_with_tools_stream

    def producer() -> None:
        try:
            for evt in chat_with_tools_stream(
                text, system=system, max_tokens=max_tokens,
                disable_thinking=disable_thinking, history=history,
            ):
                loop.call_soon_threadsafe(q.put_nowait, evt)
        except Exception as e:  # noqa: BLE001 — 吞所有，否则消费端挂死
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e), None))

    return producer


async def _dispatch(ws: WebSocket, kind: str, payload, full_so_far) -> tuple[str, bool]:
    """单事件分发为 WS 消息。返回 (new_full, is_terminal)。"""
    if kind == "sentence":
        await ws.send_text(json.dumps(
            {"type": "sentence", "text": payload}, ensure_ascii=False,
        ))
        return (full_so_far or "", False)
    if kind == "tool_call":
        await ws.send_text(json.dumps(
            {"type": "tool_call", "name": payload.get("name"),
             "arguments": payload.get("arguments")}, ensure_ascii=False,
        ))
        return ("", False)
    if kind == "tool_result":
        await ws.send_text(json.dumps(
            {"type": "tool_result", "name": payload.get("name"),
             "result": payload.get("result")}, ensure_ascii=False,
        ))
        return ("", False)
    if kind == "done":
        await ws.send_text(json.dumps(
            {"type": "done", "full": full_so_far or ""}, ensure_ascii=False,
        ))
        return ("", True)
    if kind == "error":
        await ws.send_text(json.dumps(
            {"type": "error", "message": str(payload)},
        ))
        return ("", True)
    return ("", False)


def _hydrate_history_from_l1(history: list) -> None:
    """首次 receive 时：从 SQLite recall_recent_turns 预填 history（实现 L1 reconnect 恢复）。"""
    if history:
        return  # 同一 connection 已有上下文，不重填
    try:
        recent = _memory_store.recall_recent_turns(max_turns=6)
    except Exception as e:  # noqa: BLE001
        print(f"[L1] recall failed: {e}")
        return
    for t in recent:
        history.append({"role": t["role"], "content": t["content"]})
    if recent:
        print(f"[L1] hydrated {len(recent)} turns from SQLite")


def _persist_last_turn(history: list) -> None:
    """terminal=True 时把本轮 user + assistant 写 SQLite。

    error 路径下 history 可能只有 user（没 assistant 回复），仍持久化 user
    以保留对话语境，下次 reconnect 能看到上轮提问。
    """
    if not history:
        return
    sid = _current_session_id
    # 倒着找最近一对 user/assistant（跳过中间的 tool_call/tool）
    last_assistant = None
    last_user = None
    for msg in reversed(history):
        role = msg.get("role")
        if last_assistant is None and role == "assistant" and msg.get("content"):
            last_assistant = msg
        elif last_user is None and role == "user":
            last_user = msg
            break
    try:
        if last_user is not None:
            _memory_store.append_turn(sid, "user", last_user.get("content", ""))
        if last_assistant is not None:
            _memory_store.append_turn(sid, "assistant", last_assistant.get("content", ""))
    except Exception as e:  # noqa: BLE001 — DB 故障不阻塞会话
        print(f"[L1] persist failed: {e}")


async def stream_chat_to_ws(ws: WebSocket, req: dict, history: list) -> None:
    """单次聊天请求：先 tool-detect → 必要时执行 tool → 流式回包给 WS。

    history: WS connection 持有的会话历史 list（L0 短期记忆）；
      chat_with_tools_stream 会 in-place append user/assistant/tool，供下一轮上下文。
      首次 receive 时从 SQLite hydrate L1 跨 reconnect 历史。
    """
    text, system, max_tokens, disable_thinking = _parse_chat_req(req)
    if not text:
        await ws.send_text(json.dumps({"type": "error", "message": "empty text"}))
        return
    # L1: connection 首次请求 — 从 SQLite 恢复上一段会话
    _hydrate_history_from_l1(history)
    # L0 debug：每次请求打印 history 长度 + 入栈前 user msg
    print(f"[L0] before turn: history_len={len(history)} new_user={text[:40]!r}")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    producer = _make_chat_producer(loop, q, text, system, max_tokens, disable_thinking, history)
    future = asyncio.ensure_future(loop.run_in_executor(None, producer))

    full = ""
    while True:
        try:
            evt = await asyncio.wait_for(q.get(), timeout=ROUND_TIMEOUT_S)
        except asyncio.TimeoutError:
            await ws.send_text(json.dumps({"type": "error", "message": "timeout"}))
            future.cancel()
            return
        kind, payload, full_so_far = evt
        new_full, terminal = await _dispatch(ws, kind, payload, full_so_far)
        if new_full:
            full = new_full
        if terminal:
            print(f"[L0] after turn: history_len={len(history)} last_role={history[-1].get('role') if history else None}")
            _persist_last_turn(history)
            return
