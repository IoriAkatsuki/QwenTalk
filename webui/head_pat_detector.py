"""摸摸头交互检测 — D435 手部 3D + Live2D 形象头部投影区命中。

触发条件 (按优先级):
  1. 手在画面**上方区域**（屏幕 y < HEAD_ZONE_Y_MAX）
  2. 距离 30-80cm（DEPTH_MIN..MAX，确认是"伸手摸"而非远处晃）
  3. 在区域内持续 ≥ DWELL_MS_MIN（避免一闪而过误触）

输出 PatEvent → 由 event_bus 转给前端（Live2D 害羞动画 + TTS）。

依赖：
  - GesturePipeline 输出的 21 个 handpose 3D keypoints (米制)
  - 帧分辨率（W, H）用于屏幕坐标归一化
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

# 调参常量（魔法数字避免，集中管理）
HEAD_ZONE_Y_MAX = 0.45   # 屏幕 y 归一化 [0,1]，顶部 45% 算"头顶区"
HEAD_ZONE_X_MIN = 0.30   # 左右居中带：x ∈ [0.30, 0.70]
HEAD_ZONE_X_MAX = 0.70
DEPTH_MIN_M = 0.30       # 30cm，避免镜头前一拳塞死
DEPTH_MAX_M = 0.80       # 80cm，超出认为是远处随便挥
DWELL_MS_MIN = 500       # 0.5s 持续才触发
DWELL_MS_FAST_LEAVE = 200  # 离开 200ms 重置
COOLDOWN_MS = 3000       # 触发后 3s 内不再重复触发（避免连发）


@dataclass
class PatEvent:
    timestamp: float
    dwell_ms: int
    palm_x_norm: float   # 屏幕 x [0,1]
    palm_y_norm: float
    depth_m: float


class HeadPatDetector:
    """事件驱动的摸摸头检测器。

    用法:
        det = HeadPatDetector(frame_w=640, frame_h=480)
        det.on_pat = lambda ev: send_to_live2d("blush_motion")
        # 每帧调用:
        det.update(palm_landmark_xy_norm, palm_3d_xyz_m)
    """

    def __init__(
        self,
        frame_w: int = 640,
        frame_h: int = 480,
        on_pat: Optional[Callable[[PatEvent], None]] = None,
    ) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.on_pat = on_pat
        self._enter_ms: Optional[int] = None
        self._last_in_zone_ms: int = 0
        self._last_emit_ms: int = 0

    def update(
        self,
        palm_x_norm: float | None,
        palm_y_norm: float | None,
        depth_m: float | None,
    ) -> bool:
        """每帧调用一次。返回是否触发了新的 PatEvent。"""
        now_ms = int(time.time() * 1000)

        if not self._is_in_head_zone(palm_x_norm, palm_y_norm, depth_m):
            self._maybe_reset(now_ms)
            return False

        # 在头部区域内
        self._last_in_zone_ms = now_ms
        if self._enter_ms is None:
            self._enter_ms = now_ms
            return False

        dwell = now_ms - self._enter_ms
        if dwell < DWELL_MS_MIN:
            return False

        # cooldown 中不重复触发
        if now_ms - self._last_emit_ms < COOLDOWN_MS:
            return False

        # 触发
        self._last_emit_ms = now_ms
        event = PatEvent(
            timestamp=time.time(),
            dwell_ms=dwell,
            palm_x_norm=float(palm_x_norm),
            palm_y_norm=float(palm_y_norm),
            depth_m=float(depth_m),
        )
        if self.on_pat is not None:
            try:
                self.on_pat(event)
            except Exception:  # noqa: BLE001 — handler 不能炸 detector
                pass
        return True

    def _is_in_head_zone(
        self,
        x: float | None,
        y: float | None,
        depth: float | None,
    ) -> bool:
        if x is None or y is None or depth is None:
            return False
        if not (HEAD_ZONE_X_MIN <= x <= HEAD_ZONE_X_MAX):
            return False
        if y >= HEAD_ZONE_Y_MAX:
            return False
        return DEPTH_MIN_M <= depth <= DEPTH_MAX_M

    def _maybe_reset(self, now_ms: int) -> None:
        """离开区域时 — 若已离开 > FAST_LEAVE，重置 enter 时钟。"""
        if self._enter_ms is None:
            return
        if now_ms - self._last_in_zone_ms > DWELL_MS_FAST_LEAVE:
            self._enter_ms = None


def palm_center_from_landmarks(
    landmarks_norm,  # shape (21, 2) or (21, 3), normalized [0,1] in image
    pts_3d,          # shape (21, 3), meters (or None)
) -> tuple[float | None, float | None, float | None]:
    """从 21 个 handpose 关键点估算手心位置 + 深度。

    用 MCP 关节平均（idx 0/5/9/13/17 是手掌根 + 各指根），稳定性比 wrist 好。
    """
    if landmarks_norm is None or len(landmarks_norm) < 18:
        return None, None, None
    palm_indices = [0, 5, 9, 13, 17]
    xs = [float(landmarks_norm[i][0]) for i in palm_indices]
    ys = [float(landmarks_norm[i][1]) for i in palm_indices]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    depth = None
    if pts_3d is not None and len(pts_3d) >= 18:
        zs = [float(pts_3d[i][2]) for i in palm_indices if pts_3d[i][2] > 0.01]
        if zs:
            depth = sum(zs) / len(zs)
    return cx, cy, depth
