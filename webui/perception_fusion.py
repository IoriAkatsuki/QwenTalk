"""统一 perception 状态融合 — D435 手势 + NPU 表情/眼动/头姿 + 距离 + 手部 3D。

设计：
  GesturePipeline (已有)        ┐
  EmotionSource (占位 → NPU)   ─┼─→ PerceptionFusion → 订阅者 (WS broadcast / event_bus)
  GazeSource (占位 → NPU)      ─┤
  FacePoseSource (占位 → NPU)  ─┘

板卡 NPU 上线前用占位 source 跑通接口；上线后替换 source 实现即可。
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional


@dataclass
class HandState:
    gesture: str = "init"
    conf: float = 0.0
    wrist_m: float = -1.0
    palm_score: float = 0.0
    palm_3d: tuple[float, float, float] | None = None  # 手心 3D 米制 (摸摸头要)


@dataclass
class EmotionState:
    label: str = "neutral"
    confidence: float = 0.0
    # 7 类概率：neutral / happy / sad / angry / surprised / disgusted / fearful
    probs: dict[str, float] = field(default_factory=dict)


@dataclass
class GazeState:
    vec_3d: tuple[float, float, float] = (0.0, 0.0, -1.0)  # 朝向单位向量
    screen_xy: tuple[float, float] = (0.5, 0.5)  # 屏幕投影归一化 [0,1]
    on_screen: bool = False
    dwell_ms: int = 0


@dataclass
class FacePoseState:
    yaw_deg: float = 0.0  # 头左右 (-90..+90)
    pitch_deg: float = 0.0  # 头上下
    roll_deg: float = 0.0  # 头侧倾
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h) in pixels


@dataclass
class PerceptionState:
    timestamp: float = 0.0
    hand: HandState = field(default_factory=HandState)
    emotion: EmotionState = field(default_factory=EmotionState)
    gaze: GazeState = field(default_factory=GazeState)
    face: FacePoseState = field(default_factory=FacePoseState)
    distance_m: float = -1.0  # 用户与摄像头距离 (D435 9 宫格 center 中位)
    user_present: bool = False  # face detected + distance in valid range
    event: str = ""  # 单帧事件标签 (head.pat / gesture.wave / ...)，前端按此触发动画

    def to_dict(self) -> dict:
        return asdict(self)


class _NullEmotionSource:
    """占位 emotion source — 等板卡 NPU emotions-0003 接入后替换。"""

    def snapshot(self) -> EmotionState:
        return EmotionState()


class _NullGazeSource:
    """占位 gaze source — 等板卡 NPU gaze-estimation-adas-0002 接入后替换。"""

    def snapshot(self) -> GazeState:
        return GazeState()


class _NullFacePoseSource:
    """占位 face-pose source — 等板卡 NPU head-pose-estimation-adas-0001 接入后替换。"""

    def snapshot(self) -> FacePoseState:
        return FacePoseState()


def _extract_hand(gesture_snap: dict) -> HandState:
    """从 GesturePipeline.snapshot() 转 HandState。"""
    return HandState(
        gesture=gesture_snap.get("gesture", "init"),
        conf=float(gesture_snap.get("conf", 0.0)),
        wrist_m=float(gesture_snap.get("wrist_m", -1.0)),
        palm_score=float(gesture_snap.get("palm_score", 0.0)),
        palm_3d=gesture_snap.get("palm_3d"),
    )


class PerceptionFusion:
    """5-10Hz 后台循环，融合各 source → PerceptionState → 通知订阅者。

    用法:
        fusion = PerceptionFusion(gesture_pipeline)
        fusion.subscribe(lambda s: ws_broadcast(s.to_dict()))
        fusion.start()
    """

    def __init__(
        self,
        gesture_pipeline,
        emotion_source=None,
        gaze_source=None,
        face_pose_source=None,
        rate_hz: int = 5,
    ) -> None:
        self._gp = gesture_pipeline
        self._emo = emotion_source or _NullEmotionSource()
        self._gaze = gaze_source or _NullGazeSource()
        self._face = face_pose_source or _NullFacePoseSource()
        self._interval = 1.0 / max(rate_hz, 1)
        self._subs: list[Callable[[PerceptionState], None]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest = PerceptionState()
        self._pending_events: list[str] = []  # 下一帧要塞进 state.event 的事件队列

    def subscribe(self, callback: Callable[[PerceptionState], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    # codex review #3: 5Hz pop 一个 + 无界队列 → 队尾事件陈旧。限长 + 同类型去重
    _PENDING_EVENTS_MAX = 8

    def push_event(self, event_type: str) -> None:
        """外部 detector / event_bus 触发的单帧事件注入。

        节流：同类型只保留最新位置；队列上限 8 防暴发陈旧。
        例：5s 内 user.arrived 触发 10 次只保留 1 个，避免前端连发 10 次动画。
        """
        with self._lock:
            if event_type in self._pending_events:
                self._pending_events.remove(event_type)
            self._pending_events.append(event_type)
            if len(self._pending_events) > self._PENDING_EVENTS_MAX:
                self._pending_events.pop(0)

    def snapshot(self) -> PerceptionState:
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="perception-fusion", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            # MED-2 fix: 守住 _build_state — 任何 source 抛异常都不能杀死 perception 线程。
            try:
                state = self._build_state()
            except Exception:  # noqa: BLE001
                self._stop.wait(self._interval)
                continue
            with self._lock:
                self._latest = state
                callbacks = list(self._subs)
            for cb in callbacks:
                try:
                    cb(state)
                except Exception:  # noqa: BLE001 — broadcaster 不能因一个订阅崩
                    pass
            self._stop.wait(self._interval)

    def _build_state(self) -> PerceptionState:
        gesture_snap = self._gp.snapshot() if self._gp is not None else {}
        hand = _extract_hand(gesture_snap)
        emotion = self._emo.snapshot()
        gaze = self._gaze.snapshot()
        face = self._face.snapshot()
        distance = float(gesture_snap.get("user_distance_m", hand.wrist_m))
        user_present = (
            face.bbox is not None
            or (0.2 < distance < 5.0)
        )
        # 取一个 pending event（每帧只发一个，避免高频拥塞）
        event = ""
        with self._lock:
            if self._pending_events:
                event = self._pending_events.pop(0)
        return PerceptionState(
            timestamp=time.time(),
            hand=hand,
            emotion=emotion,
            gaze=gaze,
            face=face,
            distance_m=distance,
            user_present=user_present,
            event=event,
        )
