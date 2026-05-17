#!/usr/bin/env python3
"""Intel RealSense D435 RGB-D 采集工具模块。

提供 RGB + Depth 同步采集、3D 坐标反投影、深度对齐等功能。
可作为 perception_engine 的视频源替代 V4L2 摄像头。

硬件: Intel RealSense D435/D435i/D455 (USB 3.x)
依赖: pip install pyrealsense2

用法:
    from realsense_capture import RealSenseCapture
    cam = RealSenseCapture(width=640, height=480, fps=30)
    color, depth, colormap = cam.read()
    point_3d = cam.get_3d_point(320, 240, depth)
    cam.release()
"""
from __future__ import annotations

import logging
from contextlib import suppress

import numpy as np

logger = logging.getLogger(__name__)

# 延迟导入，避免未安装时影响其他模块
_rs2 = None
_RS2_ERR: str | None = None


def _ensure_rs2():
    """延迟导入 pyrealsense2。"""
    global _rs2, _RS2_ERR
    if _rs2 is not None:
        return _rs2
    if _RS2_ERR is not None:
        raise ImportError(_RS2_ERR)
    try:
        import pyrealsense2 as rs
        _rs2 = rs
        return rs
    except ImportError as e:
        _RS2_ERR = f"pyrealsense2 未安装: {e}。请执行 pip install pyrealsense2"
        raise ImportError(_RS2_ERR) from e


def list_devices() -> list[dict]:
    """列出所有已连接的 RealSense 设备。"""
    rs = _ensure_rs2()
    return [
        {
            "name": d.get_info(rs.camera_info.name),
            "serial": d.get_info(rs.camera_info.serial_number),
            "firmware": d.get_info(rs.camera_info.firmware_version),
        }
        for d in rs.context().devices
    ]


class RealSenseCapture:
    """Intel RealSense D435 RGB + Depth 同步采集。

    线程安全: 单实例不应跨线程共享，多线程请各建独立实例。
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        serial: str | None = None,
        enable_depth: bool = True,
        align_to_color: bool = True,
        depth_quality: str = "fast",
    ) -> None:
        """
        depth_quality (O11):
          - "raw": 不跑任何滤波 (最快, ~0.5ms/frame)
          - "fast": spatial 滤波 (默认, ~2ms/frame)
          - "high": spatial+temporal+hole_filling (~5ms/frame, CPU 占用高)
        """
        rs = _ensure_rs2()
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = None
        self._depth_scale: float = 0.001
        self._intrinsics = None
        self._running = False
        self._enable_depth = enable_depth
        self._filters: list = []

        if serial:
            self._config.enable_device(serial)

        # RGB 流
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Depth 流
        if enable_depth:
            self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # 深度对齐到 RGB 坐标系
        if align_to_color and enable_depth:
            self._align = rs.align(rs.stream.color)

        # O11 修复: 深度后处理滤波器分级 (默认 "fast", 之前默认全开)
        if enable_depth and depth_quality != "raw":
            self._filters.append(rs.spatial_filter())
            if depth_quality == "high":
                self._filters.append(rs.temporal_filter())
                self._filters.append(rs.hole_filling_filter())

        self._start()

    def _start(self) -> None:
        """启动 pipeline 并读取内参。"""
        rs = _ensure_rs2()
        try:
            profile = self._pipeline.start(self._config)
        except RuntimeError as e:
            raise RuntimeError(
                f"RealSense 启动失败: {e}。检查 USB 3.x 连接、设备占用、权限"
            ) from e

        self._running = True

        if self._enable_depth:
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = cs.get_intrinsics()
        logger.info(
            "RealSense: %dx%d, fx=%.1f, scale=%.6f",
            self._intrinsics.width, self._intrinsics.height,
            self._intrinsics.fx, self._depth_scale,
        )
        # 丢弃前几帧让自动曝光稳定
        for _ in range(5):
            with suppress(Exception):
                self._pipeline.wait_for_frames(timeout_ms=1000)

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """读取一帧 RGB + Depth。

        Returns:
            (color_bgr, depth_u16, depth_colormap) — 失败时对应项为 None
        """
        if not self._running:
            return None, None, None
        try:
            fs = self._pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            return None, None, None

        if self._align:
            fs = self._align.process(fs)

        color_rs = fs.get_color_frame()
        if not color_rs:
            return None, None, None
        color = np.asanyarray(color_rs.get_data())

        depth, colormap = None, None
        if self._enable_depth:
            depth_rs = fs.get_depth_frame()
            if depth_rs:
                for f in self._filters:
                    depth_rs = f.process(depth_rs)
                depth = np.asanyarray(depth_rs.get_data())
                import cv2
                colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET,
                )
        return color, depth, colormap

    def _depth_meters(self, x: int, y: int, depth_frame: np.ndarray) -> float:
        """像素 (x,y) 的深度值（米），无效返回 -1.0。"""
        h, w = depth_frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return -1.0
        val = float(depth_frame[y, x]) * self._depth_scale
        return val if 0.01 < val < 10.0 else -1.0

    def get_distance(self, x: int, y: int, depth_frame: np.ndarray) -> float:
        """获取指定像素的深度距离（米），无效时返回 -1.0。"""
        if depth_frame is None:
            return -1.0
        return self._depth_meters(x, y, depth_frame)

    def get_3d_point(
        self, x: int, y: int, depth_frame: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """从 2D 像素 + 深度获取 3D 世界坐标 (米)，无效返回 None。"""
        rs = _ensure_rs2()
        if self._intrinsics is None or depth_frame is None:
            return None
        d = self._depth_meters(x, y, depth_frame)
        if d < 0:
            return None
        pt = rs.rs2_deproject_pixel_to_point(self._intrinsics, [float(x), float(y)], d)
        return (pt[0], pt[1], pt[2])

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def is_running(self) -> bool:
        return self._running

    def release(self) -> None:
        """停止采集并释放资源。"""
        if self._running:
            with suppress(Exception):
                self._pipeline.stop()
            self._running = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
        return False

    def __del__(self):
        self.release()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pa = argparse.ArgumentParser(description="RealSense D435 RGB-D 测试")
    pa.add_argument("--mode", choices=["list", "capture", "display"], default="list")
    pa.add_argument("--width", type=int, default=640)
    pa.add_argument("--height", type=int, default=480)
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--frames", type=int, default=100)
    a = pa.parse_args()

    if a.mode == "list":
        devs = list_devices()
        print(f"检测到 {len(devs)} 台 RealSense 设备")
        for i, d in enumerate(devs):
            print(f"  [{i}] {d['name']} SN={d['serial']} FW={d['firmware']}")

    elif a.mode == "capture":
        cam = RealSenseCapture(a.width, a.height, a.fps)
        n, t0 = 0, time.perf_counter()
        for i in range(a.frames):
            c, d, cm = cam.read()
            if c is not None:
                n += 1
            if d is not None and i % 25 == 0:
                cy, cx = d.shape[0] // 2, d.shape[1] // 2
                print(f"  帧{i}: {c.shape}, 中心距离={cam.get_distance(cx, cy, d):.3f}m")
        dt = time.perf_counter() - t0
        print(f"[结果] {n}/{a.frames} 帧, {dt:.2f}s, {n/dt:.1f} FPS")
        cam.release()

    elif a.mode == "display":
        import cv2
        cam = RealSenseCapture(a.width, a.height, a.fps)
        try:
            while True:
                c, d, cm = cam.read()
                if c is None:
                    continue
                show = np.hstack([c, cm]) if cm is not None else c
                cv2.imshow("RealSense (RGB | Depth)", show)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
