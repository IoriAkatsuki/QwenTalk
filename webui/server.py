"""FastAPI WebUI for D435 + NPU 手势识别远程查看。

路由:
    GET /               主页
    GET /stream.mjpg    MJPEG 视频流（multipart/x-mixed-replace）
    GET /events         Server-Sent Events 状态推送
    GET /api/state      一次性 JSON 状态（轮询用）

启动:
    python -m QwenTalk.webui.server  或
    bash QwenTalk/webui_start.sh
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .pipeline import GesturePipeline

STATIC_DIR = Path(__file__).resolve().parent
INDEX_HTML = STATIC_DIR / "index.html"
SSE_INTERVAL_S = 0.2  # 5 Hz 状态推送

app = FastAPI(title="QwenTalk Gesture WebUI")
pipeline = GesturePipeline()


@app.on_event("startup")
async def _on_startup() -> None:
    pipeline.start()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    pipeline.stop()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not INDEX_HTML.exists():
        return HTMLResponse("<h1>index.html missing</h1>", status_code=500)
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(pipeline.snapshot())


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
