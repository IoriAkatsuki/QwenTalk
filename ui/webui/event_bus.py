"""轻量事件总线 — perception 事件 → 触发规则 → 异步分派 callback。

适用场景：
  - 摸摸头 PatEvent → Live2D 害羞动画 + TTS
  - 距离接近 (< 1m + 持续 1s) → 唤醒 "你来啦"
  - gaze 离开屏幕 > 5s → "看着我嘛"
  - 用户长时间不说话 → 主动开口

不依赖第三方 EventBus 库；线程安全（publish 可从 perception thread 调）。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    type: str
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


Handler = Callable[[Event], None]


class EventBus:
    """同步分派的事件总线 — handler 在 publish 线程上跑。

    use case 多为短回调（推 WS 一帧、记一条 memory）；要重活就在 handler
    里再丢 Thread。保持 bus 自身简单。
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[event_type].append(handler)

    def publish(self, event_type: str, payload: dict | None = None) -> None:
        event = Event(type=event_type, payload=payload or {})
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
        for h in handlers:
            try:
                h(event)
            except Exception:  # noqa: BLE001 — 单个 handler 故障不影响其他
                pass


# ============== 触发规则 ==============

@dataclass
class _DistanceWakeState:
    in_zone_since: float = 0.0
    last_state: str = "FAR"  # FAR / APPROACH / ACTIVE


class DistanceWakeRule:
    """距离唤醒状态机：FAR(>2m) → APPROACH(1-2m) → ACTIVE(<1m & >0.3m + 持续 0.8s)."""

    APPROACH_M = 2.0
    ACTIVE_M = 1.0
    DWELL_S = 0.8

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._state = _DistanceWakeState()

    def update(self, distance_m: float) -> None:
        # HIGH-1 fix: 无效值（< 0.2 m 通常是 D435 返回 -1 / 太近遮挡）不当 APPROACH 处理；
        # 当前状态保持，避免每帧误触发 user.approaching。
        if distance_m < 0.2:
            return
        now = time.time()
        prev = self._state.last_state
        if distance_m < self.ACTIVE_M:
            if self._state.in_zone_since == 0.0:
                self._state.in_zone_since = now
            if now - self._state.in_zone_since >= self.DWELL_S and prev != "ACTIVE":
                self._bus.publish("user.arrived", {"distance_m": distance_m})
                self._state.last_state = "ACTIVE"
        elif distance_m < self.APPROACH_M:
            self._state.in_zone_since = 0.0
            if prev == "FAR":
                self._bus.publish("user.approaching", {"distance_m": distance_m})
            self._state.last_state = "APPROACH"
        else:
            self._state.in_zone_since = 0.0
            if prev == "ACTIVE":
                self._bus.publish("user.left", {"distance_m": distance_m})
            self._state.last_state = "FAR"


@dataclass
class _GazeAwayState:
    away_since: float = 0.0
    away_event_fired: bool = False


class GazeAwayRule:
    """凝视离开屏幕超 THRESHOLD_S 触发一次 gaze.away；回归触发 gaze.back。"""

    AWAY_THRESHOLD_S = 5.0

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._state = _GazeAwayState()

    def update(self, on_screen: bool) -> None:
        now = time.time()
        if on_screen:
            if self._state.away_event_fired:
                self._bus.publish("gaze.back", {})
            self._state.away_since = 0.0
            self._state.away_event_fired = False
            return
        if self._state.away_since == 0.0:
            self._state.away_since = now
        elif (now - self._state.away_since >= self.AWAY_THRESHOLD_S
              and not self._state.away_event_fired):
            self._bus.publish("gaze.away", {"away_s": now - self._state.away_since})
            self._state.away_event_fired = True


@dataclass
class _SilenceState:
    last_user_input_ts: float = field(default_factory=time.time)
    fired: bool = False


class SilenceRule:
    """长时间用户不输入 → 触发一次 user.silent。"""

    SILENCE_THRESHOLD_S = 30.0

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._state = _SilenceState()

    def mark_user_input(self) -> None:
        self._state.last_user_input_ts = time.time()
        self._state.fired = False

    def tick(self, user_present: bool = True) -> None:
        # MED-1 fix: 用户不在场时不触发 silent（避免对空气说"在想什么呢"）。
        if not user_present:
            return
        now = time.time()
        elapsed = now - self._state.last_user_input_ts
        if elapsed >= self.SILENCE_THRESHOLD_S and not self._state.fired:
            self._bus.publish("user.silent", {"silent_s": elapsed})
            self._state.fired = True


class TriggerRules:
    """聚合所有规则；可被 perception_fusion 订阅以接收 PerceptionState。"""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.distance = DistanceWakeRule(bus)
        self.gaze_away = GazeAwayRule(bus)
        self.silence = SilenceRule(bus)

    def on_perception(self, state: Any) -> None:
        """订阅 PerceptionFusion 的回调入口。"""
        try:
            user_present = bool(getattr(state, "user_present", False))
            self.distance.update(getattr(state, "distance_m", -1.0))
            gaze = getattr(state, "gaze", None)
            if gaze is not None:
                self.gaze_away.update(bool(getattr(gaze, "on_screen", False)))
            self.silence.tick(user_present=user_present)
        except Exception:  # noqa: BLE001
            pass
