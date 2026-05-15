"""FastAPI WebUI for D435 + NPU 手势识别 + Intel 酱 实时驱动。

路由:
    GET  /                主页
    GET  /intel_chan      Intel 酱 Live2D 前端
    GET  /stream.mjpg     MJPEG 视频流
    GET  /events          SSE 状态推送（兼容旧客户端）
    GET  /api/state       一次性 JSON 状态
    WS   /ws/perception   PerceptionState 5Hz 推送（emotion/gaze/hand/face）
    WS   /ws/chat         流式聊天 (用户文字 → LLM stream → 按句切分)

启动:
    python -m QwenTalk.webui.server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .pipeline import GesturePipeline
from .perception_fusion import PerceptionFusion

STATIC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STATIC_DIR.parent
INDEX_HTML = STATIC_DIR / "index.html"
INTEL_CHAN_HTML = PROJECT_DIR / "intel_chan.html"
LIVE2D_DEMO_HTML = PROJECT_DIR / "live2d_demo.html"
SSE_INTERVAL_S = 0.2  # 5 Hz 状态推送

# 一次性把项目根加入 sys.path（避免每个请求重复 insert 污染累积，详见 codex review HIGH-3）
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

app = FastAPI(title="Intel 酱 — Embedded AI Companion")
pipeline = GesturePipeline()
perception = PerceptionFusion(pipeline, rate_hz=5)
_ws_perception_clients: set[WebSocket] = set()
# CRITICAL-1 fix: 在 startup 保存主 loop 引用；
# perception 线程用它做 run_coroutine_threadsafe，避免 get_event_loop() 在非 asyncio 线程抛 RuntimeError。
_main_loop: asyncio.AbstractEventLoop | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    pipeline.start()
    perception.subscribe(_broadcast_perception_sync)
    perception.start()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    perception.stop()
    pipeline.stop()


def _broadcast_perception_sync(state) -> None:
    """同步调用桥接 → 入 asyncio loop（来自 perception 线程）。"""
    if _main_loop is None or not _main_loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(_broadcast_perception(state.to_dict()), _main_loop)


async def _broadcast_perception(payload: dict) -> None:
    dead: list[WebSocket] = []
    msg = json.dumps(payload, ensure_ascii=False)
    for ws in list(_ws_perception_clients):
        try:
            await ws.send_text(msg)
        except Exception:  # noqa: BLE001 — client gone
            dead.append(ws)
    for ws in dead:
        _ws_perception_clients.discard(ws)


def _serve_html(path: Path) -> HTMLResponse:
    if not path.exists():
        return HTMLResponse(f"<h1>{path.name} missing</h1>", status_code=404)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return _serve_html(INDEX_HTML)


@app.get("/intel_chan", response_class=HTMLResponse)
async def intel_chan_page() -> HTMLResponse:
    """Intel 酱 主前端 (Live2D + WS perception)。"""
    return _serve_html(INTEL_CHAN_HTML if INTEL_CHAN_HTML.exists() else LIVE2D_DEMO_HTML)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    snap = pipeline.snapshot()
    snap["perception"] = perception.snapshot().to_dict()
    return JSONResponse(snap)


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    boundary = "frame"
    return StreamingResponse(
        _mjpeg_iter(boundary),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


async def _mjpeg_iter(boundary: str):
    sep = f"--{boundary}\r\n".encode()
    while True:
        jpg = await asyncio.to_thread(pipeline.latest_jpeg, 2.0)
        if jpg is None:
            continue
        header = (
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpg)}\r\n\r\n"
        ).encode()
        yield sep + header + jpg + b"\r\n"


@app.get("/events")
async def events() -> StreamingResponse:
    return StreamingResponse(_sse_iter(), media_type="text/event-stream")


async def _sse_iter():
    last_idx = -1
    while True:
        snap = pipeline.snapshot()
        if snap["frame_idx"] != last_idx:
            last_idx = snap["frame_idx"]
            yield f"data: {json.dumps(snap)}\n\n"
        await asyncio.sleep(SSE_INTERVAL_S)


@app.websocket("/ws/perception")
async def ws_perception(ws: WebSocket) -> None:
    """订阅 PerceptionState 5Hz 推送 — 前端 Live2D 表情/眼动驱动源。"""
    await ws.accept()
    _ws_perception_clients.add(ws)
    try:
        # 立即发一次当前状态，避免客户端等下一帧
        await ws.send_text(json.dumps(perception.snapshot().to_dict(), ensure_ascii=False))
        while True:
            await ws.receive_text()  # 保活，忽略内容
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        _ws_perception_clients.discard(ws)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """流式聊天代理: 客户端发用户消息 → 转发 llama-server → 按句切分回客户端。

    Protocol:
        client → {"text": "你好", "thinking": "off"}
        server → {"type": "sentence", "text": "你好啊！"} 一句一帧
        server → {"type": "done", "full": "你好啊！很高兴见到你。"} 结束
        server → {"type": "error", "message": "..."}
    """
    await ws.accept()
    try:
        while True:
            req = json.loads(await ws.receive_text())
            await _stream_chat_to_ws(ws, req)
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:  # noqa: BLE001
            pass


async def _stream_chat_to_ws(ws: WebSocket, req: dict) -> None:
    """单次聊天请求：流式拉 call_llm_stream → 每句立即推 WS。"""
    text, system, max_tokens, disable_thinking = _parse_chat_req(req)
    if not text:
        await ws.send_text(json.dumps({"type": "error", "message": "empty text"}))
        return

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    producer = _make_chat_producer(loop, q, text, system, max_tokens, disable_thinking)
    # HIGH-2 fix: ensure_future + executor，任何未捕异常会 propagate 到 future，
    # 由下面的 timeout 兜底防止 ws 永远挂起。
    future = asyncio.ensure_future(loop.run_in_executor(None, producer))

    full = ""
    while True:
        try:
            kind, payload, full_so_far = await asyncio.wait_for(q.get(), timeout=180.0)
        except asyncio.TimeoutError:
            await ws.send_text(json.dumps({"type": "error", "message": "timeout"}))
            future.cancel()
            return
        if kind == "sentence":
            full = full_so_far or full
            await ws.send_text(json.dumps(
                {"type": "sentence", "text": payload}, ensure_ascii=False
            ))
        elif kind == "done":
            await ws.send_text(json.dumps({"type": "done", "full": full}, ensure_ascii=False))
            return
        elif kind == "error":
            await ws.send_text(json.dumps({"type": "error", "message": payload}))
            return


def _make_chat_producer(loop, q, text, system, max_tokens, disable_thinking):
    """构造跑在 executor 的 producer，捕异常成 error 帧入队不挂死消费端。"""
    from voice_pipeline import call_llm_stream  # noqa: E402

    def producer() -> None:
        try:
            for sentence, full_so_far in call_llm_stream(
                text, system=system, max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            ):
                loop.call_soon_threadsafe(q.put_nowait, ("sentence", sentence, full_so_far))
            loop.call_soon_threadsafe(q.put_nowait, ("done", None, None))
        except Exception as e:  # noqa: BLE001 — 必须吞，否则消费端挂死
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e), None))

    return producer


def _parse_chat_req(req: dict) -> tuple[str, str, int, object]:
    """从客户端请求提取 (text, system, max_tokens, disable_thinking)。"""
    from voice_pipeline import detect_thinking_support  # noqa: E402

    text = str(req.get("text", "")).strip()
    system = req.get("system") or "你是 Intel 酱，一个温柔的嵌入式 AI 助手，用一两句话回答。"
    max_tokens = int(req.get("max_tokens", 200))
    disable_thinking = detect_thinking_support(req.get("thinking", "auto"))
    return text, system, max_tokens, disable_thinking


def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture WebUI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    args = parser.parse_args()

    pipeline.device = args.device
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
