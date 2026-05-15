"""Intel 酱 过夜代码 smoke tests — 不依赖板卡硬件，本机可跑。

覆盖：
  1. 所有新模块能 import 不抛异常
  2. PerceptionFusion: mock GesturePipeline → 5Hz tick → 订阅者收到 state
  3. EventBus: subscribe + publish 路径
  4. DistanceWake / GazeAway / SilenceRule 状态机切换正确
  5. HeadPatDetector: 进入头顶区 + 持续 → 触发 PatEvent
  6. IntelChanContext + build_system_prompt 能生成合理 prompt
  7. FastAPI server import + 路由列表

跑法:
    cd ~/Documents/Intel/QwenTalk
    python -m pytest tests/test_smoke.py -v
    # or:
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestImports(unittest.TestCase):
    """1. 所有 overnight 新模块能 import."""

    def test_perception_fusion_imports(self):
        from webui import perception_fusion
        self.assertTrue(hasattr(perception_fusion, "PerceptionFusion"))
        self.assertTrue(hasattr(perception_fusion, "PerceptionState"))

    def test_head_pat_detector_imports(self):
        from webui import head_pat_detector
        self.assertTrue(hasattr(head_pat_detector, "HeadPatDetector"))
        self.assertTrue(hasattr(head_pat_detector, "palm_center_from_landmarks"))

    def test_event_bus_imports(self):
        from webui import event_bus
        self.assertTrue(hasattr(event_bus, "EventBus"))
        self.assertTrue(hasattr(event_bus, "TriggerRules"))

    def test_intel_chan_persona_imports(self):
        import intel_chan_persona
        self.assertTrue(hasattr(intel_chan_persona, "build_system_prompt"))
        self.assertTrue(hasattr(intel_chan_persona, "pick_proactive_line"))


class TestPerceptionFusion(unittest.TestCase):
    """2. PerceptionFusion mock GesturePipeline → 订阅收到 state."""

    def test_fusion_publishes_to_subscriber(self):
        from webui.perception_fusion import PerceptionFusion, PerceptionState

        gp = MagicMock()
        gp.snapshot.return_value = {
            "gesture": "point", "conf": 0.8,
            "wrist_m": 0.5, "palm_score": 0.7,
        }
        fusion = PerceptionFusion(gp, rate_hz=20)

        received = []
        fusion.subscribe(lambda s: received.append(s))
        fusion.start()
        time.sleep(0.25)  # 5 ticks at 20Hz
        fusion.stop()

        self.assertGreater(len(received), 2, "至少应收到 2 帧")
        first = received[0]
        self.assertIsInstance(first, PerceptionState)
        self.assertEqual("point", first.hand.gesture)


class TestEventBus(unittest.TestCase):
    """3. EventBus subscribe + publish 路径."""

    def test_subscribe_publish(self):
        from webui.event_bus import EventBus

        bus = EventBus()
        seen = []
        bus.subscribe("test.event", lambda ev: seen.append(ev))
        bus.publish("test.event", {"x": 1})
        bus.publish("other.event", {"y": 2})

        self.assertEqual(1, len(seen), "只该收到 test.event 一次")
        self.assertEqual("test.event", seen[0].type)
        self.assertEqual(1, seen[0].payload["x"])

    def test_handler_exception_isolated(self):
        from webui.event_bus import EventBus

        bus = EventBus()
        good_calls = []
        bus.subscribe("e", lambda ev: (_ for _ in ()).throw(RuntimeError("boom")))
        bus.subscribe("e", lambda ev: good_calls.append(ev))
        bus.publish("e", {})

        self.assertEqual(1, len(good_calls), "崩了一个 handler 不该影响别的")


class TestDistanceWakeRule(unittest.TestCase):
    """4a. 距离唤醒状态机：FAR → APPROACH → ACTIVE."""

    def test_arrived_after_dwell(self):
        from webui.event_bus import EventBus, DistanceWakeRule

        bus = EventBus()
        events = []
        bus.subscribe("user.arrived", lambda ev: events.append(ev))
        rule = DistanceWakeRule(bus)

        rule.update(3.0)  # FAR
        rule.update(0.5)  # 进入 ACTIVE 区
        time.sleep(0.85)  # 等 dwell 0.8s 过
        rule.update(0.5)  # 再 tick

        self.assertEqual(1, len(events), "应触发一次 user.arrived")

    def test_left_event(self):
        from webui.event_bus import EventBus, DistanceWakeRule

        bus = EventBus()
        events = []
        bus.subscribe("user.left", lambda ev: events.append(ev))
        rule = DistanceWakeRule(bus)

        rule.update(0.5)
        time.sleep(0.85)
        rule.update(0.5)        # ACTIVE
        rule.update(3.0)        # FAR (留下后)
        self.assertEqual(1, len(events), "ACTIVE → FAR 应触发 user.left")


class TestGazeAwayRule(unittest.TestCase):
    """4b. 凝视离开规则."""

    def test_away_then_back(self):
        from webui.event_bus import EventBus, GazeAwayRule

        bus = EventBus()
        away_events = []
        back_events = []
        bus.subscribe("gaze.away", lambda ev: away_events.append(ev))
        bus.subscribe("gaze.back", lambda ev: back_events.append(ev))

        rule = GazeAwayRule(bus)
        rule.AWAY_THRESHOLD_S = 0.1  # 缩短阈值方便测试
        rule.update(False)
        time.sleep(0.15)
        rule.update(False)         # 触发 gaze.away
        rule.update(True)          # 回归触发 gaze.back

        self.assertEqual(1, len(away_events))
        self.assertEqual(1, len(back_events))


class TestHeadPatDetector(unittest.TestCase):
    """5. 摸摸头检测器 — 进入头顶区 + 持续 → 触发 PatEvent."""

    def test_trigger_after_dwell(self):
        from webui import head_pat_detector
        head_pat_detector.DWELL_MS_MIN = 100  # 加速测试
        from webui.head_pat_detector import HeadPatDetector

        events = []
        det = HeadPatDetector(on_pat=lambda ev: events.append(ev))

        det.update(0.5, 0.2, 0.5)            # 头顶区 + 深度 OK，开始计时
        time.sleep(0.12)                      # 等 dwell
        det.update(0.5, 0.2, 0.5)            # 触发

        self.assertEqual(1, len(events))
        self.assertGreaterEqual(events[0].dwell_ms, 100)

    def test_no_trigger_outside_zone(self):
        from webui.head_pat_detector import HeadPatDetector

        events = []
        det = HeadPatDetector(on_pat=lambda ev: events.append(ev))
        det.update(0.5, 0.8, 0.5)            # y 太大不在头顶
        det.update(0.5, 0.8, 0.5)
        det.update(0.5, 0.2, 5.0)            # 深度太远
        self.assertEqual(0, len(events))

    def test_palm_center_from_landmarks(self):
        from webui.head_pat_detector import palm_center_from_landmarks

        # 21 keypoints, 5 个 MCP 在 (0.5, 0.5) 周围
        import numpy as np
        lm = np.zeros((21, 2), dtype=float)
        for i in [0, 5, 9, 13, 17]:
            lm[i] = (0.5, 0.5)
        pts_3d = np.zeros((21, 3), dtype=float)
        for i in [0, 5, 9, 13, 17]:
            pts_3d[i] = (0.0, 0.0, 0.6)

        cx, cy, depth = palm_center_from_landmarks(lm, pts_3d)
        self.assertAlmostEqual(0.5, cx)
        self.assertAlmostEqual(0.5, cy)
        self.assertAlmostEqual(0.6, depth, places=2)


class TestIntelChanPersona(unittest.TestCase):
    """6. system prompt + proactive 台词."""

    def test_build_system_prompt_basic(self):
        from intel_chan_persona import IntelChanContext, build_system_prompt

        ctx = IntelChanContext(
            user_name="评委", distance_m=0.8,
            emotion_label="happy", intimacy_score=5,
        )
        sp = build_system_prompt(ctx)
        self.assertIn("Intel 酱", sp)
        self.assertIn("评委", sp)
        self.assertIn("0.8", sp)
        self.assertIn("happy", sp)

    def test_proactive_line_event_types(self):
        from intel_chan_persona import pick_proactive_line

        for et in ["user.arrived", "user.left", "gaze.away", "head.pat"]:
            line = pick_proactive_line(et)
            self.assertTrue(len(line) > 0, f"{et} 应该有台词")


def _has_realsense() -> bool:
    try:
        import pyrealsense2  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_realsense(), "pyrealsense2 not installed (run on board)")
class TestServerImports(unittest.TestCase):
    """7. FastAPI server 能 import + 路由能列出（需板卡环境）."""

    def test_server_module_imports(self):
        from webui import server  # noqa: F401
        from webui.server import app
        paths = {route.path for route in app.routes}
        for expected in ("/", "/intel_chan", "/api/state",
                         "/stream.mjpg", "/events", "/ws/perception", "/ws/chat"):
            self.assertIn(expected, paths, f"missing route {expected}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
