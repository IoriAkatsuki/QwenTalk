#!/usr/bin/env bash
# webui_start.sh — 启动 D435 + NPU 手势 WebUI
#
# 默认: 0.0.0.0:8000, NPU 推理, MJPEG 流。
# 浏览器访问: http://<板卡IP>:8000
set -E -o pipefail

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-NPU}

cd "$(dirname "$0")"             # → ~/QwenTalk，让 webui/ 成为顶层包

# Conda openvino env (含 openvino + pyrealsense2 + fastapi)
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "openvino" ]; then
    if [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1090
        source ~/miniforge3/etc/profile.d/conda.sh
        conda activate openvino
    fi
fi

# Intel oneAPI (NPU 编译需要 libsvml.so 等运行时)
if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${ONEAPI_ROOT:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

echo "[webui] Host=$HOST Port=$PORT Device=$DEVICE"
echo "[webui] 浏览器访问: http://$(hostname -I | awk '{print $1}'):$PORT/"

exec python -m webui.server --host "$HOST" --port "$PORT" --device "$DEVICE"
