"""FastAPI WebUI for D435 + NPU 手势识别 + Intel 酱 实时驱动。

路由: / · /intel_chan · /stream.mjpg · /events · /api/state · /api/health ·
WS /ws/perception · WS /ws/chat。启动: python -m QwenTalk.webui.server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .chat_handler import set_perception_provider, stream_chat_to_ws
from .event_bus import EventBus, TriggerRules
from .pipeline import GesturePipeline
from .perception_fusion import PerceptionFusion

STATIC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STATIC_DIR.parent
INDEX_HTML = STATIC_DIR / "index.html"
INTEL_CHAN_HTML = PROJECT_DIR / "intel_chan.html"
LIVE2D_DEMO_HTML = PROJECT_DIR / "live2d_demo.html"
SSE_INTERVAL_S = 0.2  # 5 Hz 状态推送

# 一次性把项目根加入 sys.path（codex review HIGH-3：避免每请求重复 insert 累积）
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

pipeline = GesturePipeline()
perception = PerceptionFusion(pipeline, rate_hz=5)
event_bus = EventBus()
trigger_rules = TriggerRules(event_bus)
_ws_perception_clients: set[WebSocket] = set()
# CRITICAL-1: startup 保存主 loop 给 perception 线程做 run_coroutine_threadsafe
_main_loop: asyncio.AbstractEventLoop | None = None

# event_bus → perception.push_event 转发：让 user.arrived 等事件经 WS 推到前端。
_BUS_FORWARD = ("user.arrived", "user.left", "user.approaching",
                "gaze.away", "gaze.back", "user.silent")


def _forward_bus_to_perception(event) -> None:
    """EventBus handler — 把规则触发的事件灌进 perception 的事件通道。"""
    perception.push_event(event.type)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: startup → yield → shutdown（替代 deprecated @on_event）。"""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    # D435/NPU 是可选硬件：缺失时进入 degraded（chat/LLM 不依赖视觉），
    # /api/health 仍报 init_error 让前端可见；webui 不静默瘫痪。
    try:
        pipeline.start(init_timeout=10.0)
    except Exception as e:  # noqa: BLE001 — degraded mode 兜底
        print(f"[WARN] pipeline degraded (D435/NPU?): {e}")
    set_perception_provider(perception)
    perception.subscribe(_broadcast_perception_sync)
    perception.subscribe(trigger_rules.on_perception)
    for evt in _BUS_FORWARD:
        event_bus.subscribe(evt, _forward_bus_to_perception)
    perception.start()
    try:
        yield
    finally:
        perception.stop()
        pipeline.stop()


app = FastAPI(title="Intel 酱 — Embedded AI Companion", lifespan=_lifespan)


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
    return StreamingResponse(_mjpeg_iter(boundary),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}")


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
    """流式聊天代理：tool-aware（chat_handler.stream_chat_to_ws）。

    Protocol: {text, system?, max_tokens?, thinking?}
      → {type: sentence|tool_call|tool_result|done|error, ...}
    L0 会话历史：connection 级 history list 跨多轮 receive 复用，
    断连即清空（前端 reload 后服务端无状态）。
    """
    await ws.accept()
    history: list = []
    try:
        while True:
            req = json.loads(await ws.receive_text())
            await stream_chat_to_ws(ws, req, history)
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:  # noqa: BLE001
            pass


@app.get("/api/health")
async def api_health() -> JSONResponse:
    """硬件状态聚合：pipeline / perception / llama-server 三部件健康度。"""
    pipe, perc = _get_pipeline_health(), _get_perception_health()
    llm = await _check_llama_reachable()
    return JSONResponse({"pipeline": pipe, "perception": perc, "llama_server": llm,
                         "status": _aggregate_status(pipe, perc, llm)})


def _get_pipeline_health() -> dict:
    """优先 pipeline.health_status()（子任务 1 接口），缺失时 fallback 自检线程。"""
    if hasattr(pipeline, "health_status"):
        return pipeline.health_status()
    alive = pipeline._thread is not None and pipeline._thread.is_alive()
    return {"alive": alive, "ready": True, "init_error": None}


def _get_perception_health() -> dict:
    t = perception._thread
    return {"thread_alive": t is not None and t.is_alive(),
            "subscribers": len(perception._subs)}


async def _check_llama_reachable(
    url: str = "http://127.0.0.1:8080/v1/models", timeout: float = 2.0,
) -> dict:
    """异步探测 llama-server — urllib 跑 executor 不 block event loop。"""
    import urllib.request
    def _probe() -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001 — 网络/超时都视作 unreachable
            return False
    try:
        ok = await asyncio.wait_for(asyncio.to_thread(_probe), timeout=timeout + 1.0)
        return {"reachable": bool(ok), "url": url}
    except Exception:  # noqa: BLE001
        return {"reachable": False, "url": url}


def _aggregate_status(pipe: dict, perc: dict, llm: dict) -> str:
    """核心线程死 → down；ready/llm 缺失 → degraded；全活 → ok。"""
    if not pipe.get("alive") or not perc.get("thread_alive"):
        return "down"
    if not pipe.get("ready") or not llm.get("reachable"):
        return "degraded"
    return "ok"


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
