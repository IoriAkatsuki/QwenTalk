"""D435 + NPU 推理后台线程，把每帧 JPEG 与状态推到队列。

被 webui/server.py 复用。逻辑直接借用 gesture3d_test 的纯函数，
不重复实现规则分类器与距离闸门。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyrealsense2 as rs  # noqa: E402

from gesture3d_test import (  # noqa: E402
    DIST_MAX_M,
    DIST_MIN_M,
    PALM_CONF_THRESHOLD,
    classify_gesture,
    estimate_handpose,
    estimate_wrist_distance,
    load_npu_models,
    palm_detection_score,
)
from resources import ensure_d435, grab_frames  # noqa: E402

from .encoder import make_encoder  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class FrameState:
    gesture: str = "init"
    conf: float = 0.0
    wrist_m: float = -1.0
    palm_score: float = 0.0
    fps: float = 0.0
    frame_idx: int = 0
    stats: dict = field(default_factory=dict)
    encoder_backend: str = "unknown"


class GesturePipeline:
    """后台线程：D435 → palm gating → handpose → 3D 反投影 → 分类 → MJPEG。"""

    def __init__(self, device: str = "NPU", palm_threshold: float = PALM_CONF_THRESHOLD):
        self.device = device
        self.palm_threshold = palm_threshold
        self.frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.state = FrameState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._encoder = None  # 在 _run 里 lazy 初始化（要等 D435 出第一帧才知道分辨率）

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gesture-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> dict:
        with self._lock:
            s = self.state
            return {
                "gesture": s.gesture, "conf": s.conf,
                "wrist_m": s.wrist_m, "palm_score": s.palm_score,
                "fps": s.fps, "frame_idx": s.frame_idx,
                "stats": dict(s.stats),
                "dist_min_m": DIST_MIN_M, "dist_max_m": DIST_MAX_M,
                "palm_threshold": self.palm_threshold,
                "encoder_backend": s.encoder_backend,
            }

    def latest_jpeg(self, timeout: float = 2.0) -> bytes | None:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        d435 = ensure_d435(warmup_frames=10)
        palm_c, hand_c = load_npu_models(self.device)
        intrinsics = d435["intrinsics"]
        fps_t0, fps_count = time.perf_counter(), 0

        # 第一帧拿到分辨率后再造编码器（VA-API 需要预先告诉 ffmpeg 尺寸）
        first_color, first_depth = grab_frames()
        h, w = first_color.shape[:2]
        self._encoder = make_encoder(w, h, fps=30)
        with self._lock:
            self.state.encoder_backend = self._encoder.backend
        print(f"[pipeline] encoder backend = {self._encoder.backend}")
        # 处理第一帧
        result = self._infer_frame(first_color, first_depth, palm_c, hand_c, intrinsics)
        self._draw_overlay(first_color, result)
        self._publish_jpeg(first_color)
        fps_count = 1
        fps_count, fps_t0 = self._update_state(result, fps_count, fps_t0)

        while not self._stop_event.is_set():
            try:
                color, depth = grab_frames()
            except RuntimeError:
                continue
            result = self._infer_frame(color, depth, palm_c, hand_c, intrinsics)
            self._draw_overlay(color, result)
            self._publish_jpeg(color)
            fps_count += 1
            fps_count, fps_t0 = self._update_state(result, fps_count, fps_t0)

    def _infer_frame(self, color, depth, palm_c, hand_c, intrinsics) -> dict:
        h, w = color.shape[:2]
        palm_score = palm_detection_score(palm_c, color)
        if palm_score < self.palm_threshold:
            return {"gesture": "no_palm", "conf": 0.0, "wrist_m": -1.0,
                    "palm_score": palm_score, "landmarks": None}

        margin = min(h, w) // 4
        crop = color[margin:h - margin, margin:w - margin]
        landmarks = estimate_handpose(hand_c, crop)
        pts_3d = self._deproject(landmarks, depth, intrinsics, h, w, margin)
        gesture, conf = classify_gesture(landmarks, pts_3d=pts_3d)
        wrist = estimate_wrist_distance(pts_3d)
        return {"gesture": gesture, "conf": conf, "wrist_m": wrist,
                "palm_score": palm_score, "landmarks": landmarks, "margin": margin}

    @staticmethod
    def _deproject(landmarks, depth, intrinsics, h: int, w: int, margin: int) -> np.ndarray:
        pts = np.zeros((21, 3), dtype=np.float32)
        for i in range(min(21, landmarks.shape[0])):
            px = int(np.clip(landmarks[i][0] * (w - 2 * margin) + margin, 0, w - 1))
            py = int(np.clip(landmarks[i][1] * (h - 2 * margin) + margin, 0, h - 1))
            d_m = float(depth[py, px])
            if 0.05 < d_m < 5.0:
                pts[i] = rs.rs2_deproject_pixel_to_point(
                    intrinsics, [float(px), float(py)], d_m,
                )
        return pts

    @staticmethod
    def _draw_overlay(canvas: np.ndarray, result: dict) -> None:
        h, w = canvas.shape[:2]
        landmarks = result.get("landmarks")
        gesture, conf = result["gesture"], result["conf"]
        wrist = result["wrist_m"]
        if landmarks is not None:
            margin = result["margin"]
            pt_color = (0, 255, 0) if conf > 0.5 else (128, 128, 128)
            for i in range(min(21, landmarks.shape[0])):
                px = int(np.clip(landmarks[i][0] * (w - 2 * margin) + margin, 0, w - 1))
                py = int(np.clip(landmarks[i][1] * (h - 2 * margin) + margin, 0, h - 1))
                cv2.circle(canvas, (px, py), 3, pt_color, -1)
        cv2.putText(canvas, f"{gesture} ({conf:.0%})", (10, 40), FONT, 1.2, (0, 0, 255), 2)
        z_in_range = DIST_MIN_M <= wrist <= DIST_MAX_M
        z_color = (0, 255, 0) if z_in_range else (0, 0, 255)
        cv2.putText(canvas, f"Z={wrist:.2f}m  palm={result['palm_score']:.2f}",
                    (10, 80), FONT, 0.7, z_color, 2)

    def _publish_jpeg(self, frame: np.ndarray) -> None:
        if self._encoder is None:
            return
        data = self._encoder.encode(frame)
        if not data:
            return
        try:
            self.frame_queue.put_nowait(data)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass

    def _update_state(self, result: dict, fps_count: int, fps_t0: float) -> tuple[int, float]:
        elapsed = time.perf_counter() - fps_t0
        with self._lock:
            self.state.gesture = result["gesture"]
            self.state.conf = result["conf"]
            self.state.wrist_m = result["wrist_m"]
            self.state.palm_score = result["palm_score"]
            self.state.frame_idx += 1
            self.state.stats[result["gesture"]] = self.state.stats.get(result["gesture"], 0) + 1
            if elapsed >= 1.0:
                self.state.fps = fps_count / elapsed
        if elapsed >= 1.0:
            return 0, time.perf_counter()
        return fps_count, fps_t0
