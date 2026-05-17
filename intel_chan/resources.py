"""D435 / NPU 资源单例 — 懒初始化 + 模块级共享。

设计动机:
  - D435 hardware_reset 9s 启动开销，多工具共享一份避免重启
  - NPU 模型编译数十毫秒，但反复 read+compile 浪费
  - atexit 注册保证进程退出时清理

只有在第一次访问时初始化。线程不安全（按 agent 单线程使用假设）。
"""
from __future__ import annotations

import atexit
import os
import time
from pathlib import Path

import numpy as np

from _paths import OMZ_ROOT as MODEL_ROOT  # B01: 不再硬编码 /home/intel

_D435 = {"pipeline": None, "align": None, "depth_scale": 0.001, "intrinsics": None}
_NPU = {"palm": None, "hand": None, "core": None}


def _parse_wh(env_value: str, default_w: int, default_h: int) -> tuple[int, int]:
    """解析 'WIDTHxHEIGHT' 环境变量，失败 fallback 到默认。"""
    try:
        w, h = env_value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return default_w, default_h


def _d435_resolution() -> tuple[tuple[int, int], tuple[int, int], int]:
    """读取 RGB / Depth 分辨率与 fps（环境变量可覆盖）。

    QWENTALK_RS_COLOR=1280x720  (默认: 1280x720)
    QWENTALK_RS_DEPTH=640x480   (默认: 640x480, D435 深度最佳模式)
    QWENTALK_RS_FPS=30          (默认: 30)
    """
    color_wh = _parse_wh(os.environ.get("QWENTALK_RS_COLOR", ""), 1280, 720)
    depth_wh = _parse_wh(os.environ.get("QWENTALK_RS_DEPTH", ""), 640, 480)
    try:
        fps = int(os.environ.get("QWENTALK_RS_FPS", "30"))
    except ValueError:
        fps = 30
    return color_wh, depth_wh, fps


def ensure_d435(*, force_reset: bool = False, warmup_frames: int = 8):
    """启动 D435 RGB+Depth pipeline。

    force_reset=True: 即使已有单例也先释放并 hardware_reset (用于异常恢复)
    warmup_frames: 启动后丢弃多少帧让流稳定 (默认 8 个)
    """
    if force_reset:
        release_d435()

    if _D435["pipeline"] is not None:
        return _D435

    import pyrealsense2 as rs

    devs = list(rs.context().query_devices())
    if not devs:
        raise RuntimeError("未检测到 D435 设备")
    for d in devs:
        d.hardware_reset()
    time.sleep(8)

    pipeline = rs.pipeline()
    config = rs.config()
    (cw, ch), (dw, dh), fps = _d435_resolution()
    config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 显式开启 RGB 自动曝光 + 自动白平衡 + 曝光优先级保 fps
    color_sensor = profile.get_device().query_sensors()[1]  # idx 1 通常是 RGB
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
    if color_sensor.supports(rs.option.auto_exposure_priority):
        color_sensor.set_option(rs.option.auto_exposure_priority, 0)  # 0=保 fps, 1=允许降 fps

    # B03 修复: warmup 失败计数，全失败则抛错而非吞掉
    fails = 0
    for _ in range(warmup_frames):
        try:
            pipeline.wait_for_frames(timeout_ms=8000)
        except RuntimeError:
            fails += 1
    if fails >= warmup_frames:
        try:
            pipeline.stop()
        except Exception:
            pass
        raise RuntimeError(f"D435 warmup 全部失败 ({fails}/{warmup_frames}) — 检查 USB 连接")

    _D435["pipeline"] = pipeline
    _D435["align"] = align
    _D435["depth_scale"] = profile.get_device().first_depth_sensor().get_depth_scale()
    _D435["intrinsics"] = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile().get_intrinsics()
    )
    atexit.register(release_d435)
    return _D435


def release_d435():
    """主动释放 D435 单例 (异常恢复或干净退出)。"""
    if _D435["pipeline"]:
        try:
            _D435["pipeline"].stop()
        except Exception:
            pass
        _D435["pipeline"] = None
        _D435["align"] = None
        _D435["intrinsics"] = None


def ensure_npu():
    """编译 palm + handpose 到 NPU。返回单例 dict。"""
    if _NPU["palm"] is not None:
        return _NPU
    import openvino as ov
    core = ov.Core()
    palm = core.read_model(str(MODEL_ROOT / "palm_detection_mediapipe_2023feb.onnx"))
    palm.reshape({palm.input(0): [1, 192, 192, 3]})
    _NPU["palm"] = core.compile_model(palm, "NPU")

    hand = core.read_model(str(MODEL_ROOT / "handpose_estimation_mediapipe_2023feb.onnx"))
    hand.reshape({hand.input(0): [1, 224, 224, 3]})
    _NPU["hand"] = core.compile_model(hand, "NPU")
    _NPU["core"] = core
    return _NPU


def grab_frames():
    """同步取一帧 RGB + Depth (米单位)。"""
    d = ensure_d435()
    aligned = d["align"].process(d["pipeline"].wait_for_frames(timeout_ms=8000))
    color = np.asanyarray(aligned.get_color_frame().get_data())
    depth_m = (np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32)
               * d["depth_scale"])
    return color, depth_m
