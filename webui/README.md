# QwenTalk WebUI

D435 + NPU 手势识别的远程查看 Web 前端。

```
浏览器  ─┬─ /              单页 UI（Vanilla JS）
        ├─ /stream.mjpg   MJPEG 视频流（自带 NPU overlay）
        ├─ /events        SSE 状态推送（5 Hz）
        └─ /api/state     一次性 JSON

板卡    ─┬─ FastAPI / uvicorn  (server.py)
        ├─ GesturePipeline 线程  (pipeline.py)
        │   D435 → palm_detection → handpose → 3D 反投影 → 分类
        └─ NPU OpenVINO + librealsense2 (RSUSB backend)
```

## 1. 最快启动（板卡）

```bash
# 板卡端
bash ~/QwenTalk/webui_start.sh
# 默认监听 0.0.0.0:8000，NPU 推理
```

笔记本浏览器打开 `http://192.168.1.8:8000` 即可看到：

- **左侧**：实时 RGB + 21 关键点 + 手势 label + 距离 overlay
- **右上**：当前手势卡片（图标、置信度、palm score、距离、FPS、闸门状态）
- **右中**：距离尺（绿色区段 0.3-1.5 m，蓝色游标实时跟随）
- **右下**：5 手势 + 4 边界状态累计统计

## 2. 启动选项

```bash
# 自定端口 / 设备
HOST=0.0.0.0 PORT=8088 DEVICE=GPU bash webui_start.sh

# 或直接调 python -m
python -m QwenTalk.webui.server --host 0.0.0.0 --port 8000 --device NPU
```

`DEVICE` 可选 `NPU` / `GPU` / `CPU`，对应 OpenVINO compile target。

## 3. 用 systemd-run 隔离

把 WebUI 放到独立 cgroup slice，避免 LLM 推理时 NPU 抢资源：

```bash
systemd-run --user --unit=webui-gesture --slice=ai-perception.slice \
    bash ~/QwenTalk/webui_start.sh
journalctl --user -u webui-gesture -f
systemctl --user stop webui-gesture
```

## 4. 编码路径

### 4.1 当前默认：MJPEG（CPU SIMD）

`cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])` 走 libjpeg-turbo
NEON/AVX2，在 Arrow Lake-U 上 480p@30fps **CPU 占用 < 8%**，单 JPEG ≈ 30 KB。

优点：浏览器零依赖（`<img>` 直显）、零延迟（无 GOP）、抗丢帧。
缺点：**不走 Media Engine**，码率高（~7 Mbps）。

### 4.2 进阶：H.264 over WebRTC（用 Xe Media Engine）

板卡已具备：

```
✓ ffmpeg + libva 1.22 + iHD VAAPI driver
✓ gst-launch-1.0 + webrtcbin
✗ vah264enc / vaapih264enc 插件（gst-plugins-vaapi 未装）
✗ MediaMTX
```

最少改动方案是：
**ffmpeg 抓 MJPEG → h264_vaapi 硬编 → MediaMTX RTSP 入 → WebRTC 出**。

#### 步骤

```bash
# 1) MediaMTX (Go-based 全协议网关，单二进制)
mkdir -p ~/.local/bin && cd /tmp
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_amd64.tar.gz
tar xzf mediamtx_linux_amd64.tar.gz mediamtx
mv mediamtx ~/.local/bin/

# 2) 启动 MediaMTX (默认配 RTSP:8554 / WebRTC:8889 / HLS:8888)
~/.local/bin/mediamtx &

# 3) 启动 WebUI（继续提供 MJPEG）
bash ~/QwenTalk/webui_start.sh &

# 4) ffmpeg 把 MJPEG 转 H.264 VA-API 推到 MediaMTX
ffmpeg -hide_banner -loglevel warning \
  -f mjpeg -framerate 30 -i http://127.0.0.1:8000/stream.mjpg \
  -vaapi_device /dev/dri/renderD128 \
  -vf 'format=nv12,hwupload' \
  -c:v h264_vaapi -profile:v main -b:v 4M -g 30 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/gesture
```

浏览器访问：

- **WebRTC（< 200 ms 延迟）**：`http://192.168.1.8:8889/gesture`
- **HLS（~5 s 延迟，兼容性好）**：`http://192.168.1.8:8888/gesture/index.m3u8`
- **RTSP（VLC/OBS 拉流）**：`rtsp://192.168.1.8:8554/gesture`

#### 验证 VA-API 真正生效

```bash
intel_gpu_top -d render -s 100   # 板卡上看 Video 引擎占用
# 推流时 "Video" 这一行应该 > 30%；如果是 0 说明 ffmpeg 走了 CPU
```

> **注意**：`stream.mjpg` 已经被 MJPEG 客户端解码 → 重新 H.264 编码并不省 CPU。
> 如果要彻底走硬编路径，应该把 `pipeline.py` 改成 `appsrc`（GStreamer
> Python 绑定）直接喂 NumPy 数组进 `vah264enc`，跳过 JPEG 中转。
> 这是后续 P3 工作（已在 §8.5 报告"未来工作"列出）。

### 4.3 OBS Studio 玩法

OBS 30+ 支持 **WHIP** 推流：

```
OBS → 设置 → 直播 → 服务: WHIP → URL: http://192.168.1.8:8889/gesture/whip
```

也可以 OBS **拉流**（Media Source → `rtsp://192.168.1.8:8554/gesture`）做合成 / 录制 / B 站直播。

## 5. 已知问题

| 现象 | 排查 |
|------|------|
| 浏览器一片黑 | 板卡防火墙：`sudo firewall-cmd --add-port=8000/tcp --permanent && sudo firewall-cmd --reload` |
| `pyrealsense2 ImportError` | 没在 conda openvino env：`conda activate openvino` |
| `palm_detection.onnx 不存在` | `bash QwenTalk/download_headpose.sh` 重新拉模型 |
| FPS < 10 | NPU AOT 编译还没跑完，第一次启动等 60 s；或 `LRS_BACKEND=rsusb` 没 export |
| MJPEG 流卡住 | 浏览器 Tab 切到后台时部分浏览器会暂停 MJPEG，切回即恢复 |
| ffmpeg `Failed to open VAAPI device` | 用户不在 `render` 组：`sudo usermod -aG render $USER`，重登录 |

## 6. 文件清单

```
QwenTalk/webui/
├── __init__.py       # 让 -m 能找到包
├── server.py         # FastAPI / 4 个路由
├── pipeline.py       # GesturePipeline 后台线程
├── index.html        # 单页前端（无依赖）
└── README.md         # 本文档

QwenTalk/webui_start.sh  # 一键启动（含 conda + oneAPI 环境激活）
```
