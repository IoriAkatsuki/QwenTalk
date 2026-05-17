#!/usr/bin/env python3
"""Xe Media Engine 硬件视频解码工具模块。

支持三种后端（按优先级自动选择）:
  1. GStreamer + VA-API  — Media Engine 硬件解码/缩放/颜色转换
  2. OpenCV + V4L2/FFMPEG — 设置 LIBVA 环境变量触发硬件加速
  3. CPU 软解码 fallback  — 普通 cv2.VideoCapture

板卡环境: libva-intel-media-driver, intel-vpl-gpu-rt, /dev/dri/renderD128

用法:
    from media_decode import create_hw_capture, create_file_decoder
    cap, backend = create_hw_capture(source=0, width=640, height=480)
    ret, frame = cap.read()
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

# VA-API 环境变量，确保 iHD 驱动被选中
_VA_ENV = {"LIBVA_DRIVER_NAME": "iHD", "LIBVA_DRIVERS_PATH": "/usr/lib64/dri"}
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".flv"}


def _run_quiet(cmd: list[str], timeout: int = 5) -> subprocess.CompletedProcess:
    """静默执行外部命令，统一超时和异常处理。"""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _check_vaapi() -> bool:
    """检查 VA-API 是否可用。"""
    if not Path("/dev/dri/renderD128").exists():
        return False
    try:
        r = _run_quiet(
            ["vainfo", "--display", "drm", "--device", "/dev/dri/renderD128"],
        )
        return r.returncode == 0 and "VAProfile" in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_gst_va_style() -> str | None:
    """检测 GStreamer VA 元素风格: "va" (新式) / "vaapi" (旧式) / None。"""
    for style, elem in [("va", "vapostproc"), ("vaapi", "vaapipostproc")]:
        try:
            if _run_quiet(["gst-inspect-1.0", elem]).returncode == 0:
                return style
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _gst_pipeline(source: int, w: int, h: int, fps: int, va: str | None) -> str:
    """构建摄像头 GStreamer pipeline。va=None 时用软件 videoconvert。"""
    postproc = {
        "va": "vapostproc", "vaapi": "vaapipostproc",
    }.get(va, "videoconvert")
    return (
        f"v4l2src device=/dev/video{source} ! "
        f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
        f"{postproc} ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
    )


def _try_open(pipeline_or_id, api=None) -> cv2.VideoCapture | None:
    """尝试打开并验证一帧可读。"""
    cap = cv2.VideoCapture(pipeline_or_id, api) if api else cv2.VideoCapture(pipeline_or_id)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        cap.release()
    return None


def create_hw_capture(
    source: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> tuple[cv2.VideoCapture, str]:
    """创建硬件加速的视频捕获，自动选择最优后端。

    Returns:
        (cap, backend_name) — backend_name 为
        "gstreamer-va" / "gstreamer-sw" / "ffmpeg-vaapi" / "cpu-v4l2"

    Raises:
        RuntimeError: 所有后端均失败
    """
    vaapi_ok = _check_vaapi()
    # B27 修复: getBuildInformation() 写 "GStreamer: NO" 时也含 "GStreamer"
    # 需要严格匹配 "GStreamer:                   YES"
    has_gst = bool(re.search(r"GStreamer:\s*YES", cv2.getBuildInformation()))

    # 1. GStreamer + VA-API 硬件后处理
    if vaapi_ok and has_gst:
        va_style = _check_gst_va_style()
        if va_style:
            cap = _try_open(
                _gst_pipeline(source, width, height, fps, va_style),
                cv2.CAP_GSTREAMER,
            )
            if cap:
                return cap, f"gstreamer-{va_style}"
        # GStreamer 软件 videoconvert fallback
        cap = _try_open(
            _gst_pipeline(source, width, height, fps, None),
            cv2.CAP_GSTREAMER,
        )
        if cap:
            return cap, "gstreamer-sw"

    # 2. V4L2 with LIBVA env (O12 修复: V4L2 raw camera 不是真硬件解码,
    # 仅 ffmpeg/libva 处理可能用 VAAPI; raw UVC 抓帧通常没硬件加速)
    if vaapi_ok:
        for k, v in _VA_ENV.items():
            os.environ.setdefault(k, v)
        cap = _try_open(source, cv2.CAP_V4L2) or _try_open(source)
        if cap:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            # 实际是 V4L2 + libva 环境就绪，并不保证硬件解码
            return cap, "v4l2-libva-env"

    # 3. CPU fallback (B26 修复: 不再重复 _try_open 第二次, 单一 cap)
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # 验证可读 1 帧 (避免设备假打开)
        ret, _ = cap.read()
        if ret:
            return cap, "cpu-v4l2"
        cap.release()

    raise RuntimeError(
        f"无法打开摄像头 /dev/video{source}，"
        "请检查设备连接和权限 (usermod -aG video $USER)"
    )


def create_file_decoder(
    filepath: str,
    hw_accel: bool = True,
) -> tuple[cv2.VideoCapture, str]:
    """创建视频文件的硬件/软件解码器。

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 所有后端均失败
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {filepath}")

    if hw_accel:
        va_style = _check_gst_va_style()
        if va_style:
            postproc = "vapostproc" if va_style == "va" else "vaapipostproc"
            pipeline = (
                f'filesrc location="{path}" ! decodebin ! '
                f"{postproc} ! video/x-raw,format=BGR ! appsink drop=0 sync=0"
            )
            cap = _try_open(pipeline, cv2.CAP_GSTREAMER)
            if cap:
                # 重新打开（验证消耗了一帧）
                cap.release()
                return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER), f"gstreamer-{va_style}"

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, "cpu-ffmpeg"
    raise RuntimeError(f"无法打开视频文件: {filepath}")


def diagnose() -> dict:
    """诊断 Media Engine / VA-API 环境，返回检测结果字典。"""
    info: dict = {
        "render_node": Path("/dev/dri/renderD128").exists(),
        "vaapi_available": _check_vaapi(),
        "gstreamer_va_style": _check_gst_va_style(),
        "opencv_gstreamer": "GStreamer" in cv2.getBuildInformation(),
    }
    # vainfo 硬件 profile 列表
    try:
        env = {**os.environ, **_VA_ENV}
        r = subprocess.run(
            ["vainfo", "--display", "drm", "--device", "/dev/dri/renderD128"],
            capture_output=True, text=True, timeout=5, env=env,
        )
        info["vainfo_profiles"] = [
            l.strip() for l in r.stdout.splitlines() if "VAProfile" in l
        ][:15]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["vainfo_profiles"] = []
    return info


# ---------------------------------------------------------------------------
# __main__ 测试入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pa = argparse.ArgumentParser(description="Xe Media Engine 解码测试")
    pa.add_argument("--mode", choices=["camera", "file", "diagnose"], default="diagnose")
    pa.add_argument("--source", type=int, default=0)
    pa.add_argument("--width", type=int, default=640)
    pa.add_argument("--height", type=int, default=480)
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--file", type=str, default="")
    a = pa.parse_args()

    if a.mode == "diagnose":
        print("=" * 55, "\n  Media Engine / VA-API 环境诊断\n" + "=" * 55)
        for k, v in diagnose().items():
            if isinstance(v, list):
                print(f"  {k}: ({len(v)} profiles)")
                for item in v[:10]:
                    print(f"    {item}")
            else:
                print(f"  {k}: {v}")

    elif a.mode == "camera":
        try:
            cap, backend = create_hw_capture(a.source, a.width, a.height, a.fps)
        except RuntimeError as e:
            print(f"[错误] {e}"); raise SystemExit(1)
        print(f"[后端] {backend}")
        # B28 修复: 跟踪 last_frame, 零帧时不再 UnboundLocalError
        n, t0, last_frame = 0, time.perf_counter(), None
        for _ in range(100):
            ok, frame = cap.read()
            if ok:
                n += 1
                last_frame = frame
        dt = time.perf_counter() - t0
        if n == 0:
            print(f"[结果] 0 帧 / {dt:.2f}s — 摄像头未输出 (后端 {backend})")
        else:
            print(f"[结果] {n} 帧 / {dt:.2f}s = {n/dt:.1f} FPS, shape={last_frame.shape}")
        cap.release()

    elif a.mode == "file":
        if not a.file:
            print("[错误] 需要 --file 参数"); raise SystemExit(1)
        cap, backend = create_file_decoder(a.file)
        print(f"[后端] {backend}")
        n, t0 = 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
        dt = time.perf_counter() - t0
        print(f"[结果] {n} 帧 / {dt:.2f}s = {n/dt:.1f} FPS")
        cap.release()
