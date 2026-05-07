"""共享路径配置 — 替代各文件硬编码 `/home/intel/...`。

优先级:
  1. 环境变量 QWENTALK_ROOT (项目根) / QWENTALK_MODELS (模型根)
  2. 脚本所在目录 (Path(__file__).parent)
  3. 兜底: ~/QwenTalk

设计动机:
  原代码假设固定 `/home/intel/QwenTalk`，本机 (oasis) / 板卡 (intel) /
  其他用户 都会失败。统一从 env 或相对位置推导。
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("QWENTALK_ROOT") or Path(__file__).parent
).expanduser().resolve()

MODELS_ROOT = Path(
    os.environ.get("QWENTALK_MODELS") or PROJECT_ROOT / "models"
).expanduser().resolve()

# NPU OMZ 模型目录 (palm/handpose/yolov8n-face)
OMZ_ROOT = MODELS_ROOT / "omz"

# LLM/VLM GGUF 目录 (默认 ~/models 板卡惯例)
GGUF_ROOT = Path(
    os.environ.get("QWENTALK_GGUF") or Path.home() / "models"
).expanduser().resolve()

# SenseVoice ONNX 目录
SENSEVOICE_ROOT = Path(
    os.environ.get("SENSEVOICE_ROOT") or Path.home() / "asr_tts" / "SenseVoice-onnx"
).expanduser().resolve()

# Head-pose ONNX 目录
HEADPOSE_ROOT = Path(
    os.environ.get("HEADPOSE_ROOT") or GGUF_ROOT / "head-pose-estimation-adas-0001" / "FP16"
).expanduser().resolve()
