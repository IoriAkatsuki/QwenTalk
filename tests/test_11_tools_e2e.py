"""11 个 in-process function tools 的端到端回归测试。

覆盖：
  1. tool registration: TOOLS dict 11 个 + TOOL_SCHEMAS 11 个 + 名字对齐
  2. 每个 tool 单测（mock provider 注入 FakePerception + FakeMemoryStore）
     - 跳过需要外网的 web_search / get_weather
  3. Memory L1 / L2 跨"会话"持久化（tmp_path 真 SQLite）
  4. multi-step ReAct（mock LLM endpoint 返回连续 tool_calls，验证多轮）

约束：
  - 不 mock 数据库；用 tmp_path SQLite 真跑
  - 不假设板卡可达；纯本机能跑
  - 不修改任何源码（除了新增本文件）

跑法:
    cd ~/Documents/Intel/QwenTalk
    python -m pytest tests/test_11_tools_e2e.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fakes — 模拟 perception / memory_store provider
# ---------------------------------------------------------------------------

def _make_fake_perception_snapshot():
    """构造一份完整的 PerceptionState 样本（含 face/hand/emotion/gaze）。"""
    from webui.perception_fusion import (
        EmotionState, FacePoseState, GazeState, HandState, PerceptionState,
    )
    return PerceptionState(
        timestamp=1715923200.0,
        hand=HandState(
            gesture="point", conf=0.92, wrist_m=0.55,
            palm_score=0.81, palm_3d=(0.1, 0.2, 0.55),
        ),
        emotion=EmotionState(label="happy", confidence=0.78, probs={"happy": 0.78}),
        gaze=GazeState(vec_3d=(0.0, 0.0, -1.0), screen_xy=(0.5, 0.5),
                       on_screen=True, dwell_ms=300),
        face=FacePoseState(yaw_deg=5.0, pitch_deg=-3.0, roll_deg=1.0,
                           bbox=(100, 120, 80, 80)),
        distance_m=0.8,
        user_present=True,
        event="",
    )


class FakePerception:
    """轻量 perception provider —— 只需 snapshot() + push_event()."""

    def __init__(self):
        self._snap = _make_fake_perception_snapshot()
        self.events: list[str] = []

    def snapshot(self):
        return self._snap

    def push_event(self, event_type: str) -> None:
        self.events.append(event_type)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_perception():
    """供需要 perception 的 tool 用。"""
    return FakePerception()


@pytest.fixture
def real_memory_store(tmp_path):
    """tmp_path 真 SQLite —— 不 mock 数据库。"""
    from webui.memory import MemoryStore
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(str(db_path))
    yield store
    store.close()


@pytest.fixture
def wired_tool_impls(fake_perception, real_memory_store):
    """注入 perception + memory_store 到 tool_impls 全局变量。

    yield 后还原为 None 避免污染其他测试。
    """
    from webui import tool_impls
    tool_impls.set_perception(fake_perception)
    tool_impls.set_memory_store(real_memory_store)
    yield tool_impls
    tool_impls.set_perception(None)
    tool_impls.set_memory_store(None)


# ---------------------------------------------------------------------------
# 1. Tool registration —— 11 个 key + 11 个 schema 一一对应
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """验证 TOOLS dict 与 TOOL_SCHEMAS 数量与名字一一对应。"""

    def test_tools_dict_has_11_entries(self):
        """TOOLS dict 必须正好 11 个 in-process tool。"""
        from webui.chat_tools import TOOLS
        assert 11 == len(TOOLS)

    def test_tool_schemas_has_11_entries(self):
        """webui 暴露的 TOOL_SCHEMAS 应过滤为 11 个（与 TOOLS 对齐）。"""
        from webui.chat_tools import TOOL_SCHEMAS
        assert 11 == len(TOOL_SCHEMAS)

    def test_tools_and_schemas_names_match(self):
        """每个 tool 名字必须有对应 schema，反之亦然。"""
        from webui.chat_tools import TOOLS, TOOL_SCHEMAS
        tool_names = set(TOOLS.keys())
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert tool_names == schema_names

    def test_expected_tool_names_present(self):
        """断言 11 个具体 tool 名字都在（防止重构丢工具）。"""
        from webui.chat_tools import TOOLS
        expected = {
            "web_search", "get_weather",
            "get_temperature", "get_system_info",
            "get_gesture", "get_distance", "get_scene", "identify_user",
            "speak", "memory_store", "memory_recall",
        }
        assert expected == set(TOOLS.keys())

    def test_tools_are_callable(self):
        """每个 TOOLS 值都是 callable。"""
        from webui.chat_tools import TOOLS
        for name, fn in TOOLS.items():
            assert callable(fn), f"{name} 不是 callable"

    def test_schemas_have_required_fields(self):
        """每个 schema 必须有 function.name + function.parameters。"""
        from webui.chat_tools import TOOL_SCHEMAS
        for s in TOOL_SCHEMAS:
            assert "function" == s["type"]
            fn = s["function"]
            assert "name" in fn
            assert "parameters" in fn
            assert "object" == fn["parameters"]["type"]


# ---------------------------------------------------------------------------
# 2. 每个 tool 单测（mock provider；跳过外网）
# ---------------------------------------------------------------------------

class TestSystemTools:
    """get_temperature / get_system_info —— 委托 tools.py，读真 sysfs/psutil。"""

    def test_get_temperature_returns_zones(self):
        """get_temperature 应返回 zones_celsius dict（本机有 thermal_zone）。"""
        from webui import tool_impls
        result = tool_impls.get_temperature()
        # 容忍本机没 thermal_zone 的情况
        assert isinstance(result, dict)
        if "error" not in result:
            assert "zones_celsius" in result or "max_celsius" in result

    def test_get_system_info_cpu(self):
        """get_system_info('cpu') 应返回 cpu_percent + cpu_count。"""
        from webui import tool_impls
        result = tool_impls.get_system_info("cpu")
        assert "cpu_count" in result
        assert isinstance(result["cpu_count"], int)
        assert result["cpu_count"] > 0

    def test_get_system_info_memory(self):
        """get_system_info('memory') 应返回 total_gb + percent。"""
        from webui import tool_impls
        result = tool_impls.get_system_info("memory")
        assert "total_gb" in result
        assert "percent" in result

    def test_get_system_info_unknown_metric(self):
        """未知 metric 应返回 error dict 而非抛异常。"""
        from webui import tool_impls
        result = tool_impls.get_system_info("nonsense")
        assert "error" in result


class TestPerceptionTools:
    """get_gesture / get_distance / get_scene / identify_user —— 需 perception。"""

    def test_get_gesture_reads_snapshot(self, wired_tool_impls):
        """get_gesture 应返回 perception snapshot 里的 hand 字段。"""
        result = wired_tool_impls.get_gesture()
        assert "point" == result["gesture"]
        assert 0.92 == result["confidence"]
        assert 0.55 == result["wrist_distance_m"]
        assert 0.81 == result["palm_score"]
        assert (0.1, 0.2, 0.55) == result["palm_3d_m"]

    def test_get_gesture_no_perception_returns_error(self):
        """provider 未注入时返回 error，不抛异常。"""
        from webui import tool_impls
        tool_impls.set_perception(None)
        result = tool_impls.get_gesture()
        assert "error" in result

    def test_get_distance_user_region(self, wired_tool_impls):
        """get_distance('user') 读 snap.distance_m。"""
        result = wired_tool_impls.get_distance(region="user")
        assert "user" == result["region"]
        assert 0.8 == result["distance_m"]

    def test_get_distance_hand_region(self, wired_tool_impls):
        """get_distance('hand') 读 snap.hand.wrist_m。"""
        result = wired_tool_impls.get_distance(region="hand")
        assert "hand" == result["region"]
        assert 0.55 == result["distance_m"]

    def test_get_distance_invalid_distance(self, fake_perception, wired_tool_impls):
        """distance ≤ 0 时返回 None + note。"""
        fake_perception._snap.distance_m = -1.0
        result = wired_tool_impls.get_distance(region="user")
        assert result["distance_m"] is None
        assert "note" in result

    def test_get_scene_describes_user(self, wired_tool_impls):
        """get_scene 应输出含距离 + 手势 + 表情 的中文描述。"""
        result = wired_tool_impls.get_scene()
        assert result["user_present"] is True
        desc = result["description"]
        assert "0.8" in desc  # 距离
        assert "point" in desc  # 手势
        assert "happy" in desc  # 表情

    def test_get_scene_no_user(self, fake_perception, wired_tool_impls):
        """user_present=False 时应描述 '视野内没有人'。"""
        fake_perception._snap.user_present = False
        result = wired_tool_impls.get_scene()
        assert "视野内没有人" == result["description"]

    def test_identify_user_returns_face_pose(self, wired_tool_impls):
        """identify_user 应返回 head_pose_deg + emotion + gaze_on_screen。"""
        result = wired_tool_impls.identify_user()
        assert result["face_detected"] is True
        assert 5.0 == result["head_pose_deg"]["yaw"]
        assert "happy" == result["emotion"]
        assert result["gaze_on_screen"] is True

    def test_identify_user_no_face_detected(self):
        """face.bbox=None 时 face_detected=False，不抛异常（TG6 边界）。"""
        from webui.perception_fusion import (
            EmotionState, FacePoseState, GazeState, HandState, PerceptionState,
        )
        from webui import tool_impls

        class _FacelessPerc:
            def snapshot(self):
                return PerceptionState(
                    timestamp=0, hand=HandState(), emotion=EmotionState(),
                    gaze=GazeState(), face=FacePoseState(bbox=None),
                    distance_m=-1, user_present=False, event="",
                )

            def push_event(self, e): pass

        tool_impls.set_perception(_FacelessPerc())
        result = tool_impls.identify_user()
        assert result["face_detected"] is False
        assert "note" in result


class TestSpeakTool:
    """speak 通过 perception.push_event 投递到前端 WS。"""

    def test_speak_queues_event(self, fake_perception, wired_tool_impls):
        """speak('你好') 应 push 'speak:|你好' 事件。"""
        result = wired_tool_impls.speak(text="你好")
        assert result["queued"] is True
        assert "你好" == result["text"]
        assert any("speak:|你好" == e for e in fake_perception.events)

    def test_speak_truncates_long_text(self, fake_perception, wired_tool_impls):
        """超过 100 字应截断；不抛异常。"""
        long_text = "啊" * 200
        result = wired_tool_impls.speak(text=long_text)
        assert 100 == len(result["text"])
        assert result["queued"] is True

    def test_speak_empty_text_errors(self, wired_tool_impls):
        """空字符串应返回 error。"""
        result = wired_tool_impls.speak(text="")
        assert "error" in result

    def test_speak_no_perception_errors(self):
        """provider 未注入时返回 error。"""
        from webui import tool_impls
        tool_impls.set_perception(None)
        result = tool_impls.speak(text="hi")
        assert "error" in result

    def test_speak_push_event_raises_returns_error(self):
        """push_event 抛异常应被 catch 返回 error dict，不冒泡破坏 ReAct loop（M3）。"""
        from unittest.mock import MagicMock
        from webui import tool_impls
        perc = MagicMock()
        perc.push_event = MagicMock(side_effect=RuntimeError("event queue full"))
        tool_impls.set_perception(perc)
        result = tool_impls.speak(text="测试")
        assert "error" in result
        assert "RuntimeError" in result["error"]
        assert "queue full" in result["error"]


class TestMemoryTools:
    """memory_store / memory_recall —— 真 SQLite 持久化。"""

    def test_memory_store_writes_fact(self, real_memory_store, wired_tool_impls):
        """memory_store 应写入 persona_facts 表。"""
        result = wired_tool_impls.memory_store(
            content="用户喜欢喝美式咖啡", importance=0.8,
        )
        assert result["stored"] is True
        assert 0.8 == result["importance"]
        # 直查 store
        facts = real_memory_store.get_persona_facts(top_n=10)
        assert 1 == len(facts)
        assert "用户喜欢喝美式咖啡" == facts[0]["value"]

    def test_memory_store_with_explicit_key(self, real_memory_store, wired_tool_impls):
        """显式 key 应覆盖默认 hash key。"""
        wired_tool_impls.memory_store(
            content="生日 5 月 17 日", key="birthday", importance=0.9,
        )
        facts = real_memory_store.get_persona_facts(top_n=10)
        keys = [f["key"] for f in facts]
        assert "birthday" in keys

    def test_memory_store_clamps_importance(self, wired_tool_impls):
        """importance 超界应 clamp 到 [0, 1]。"""
        r1 = wired_tool_impls.memory_store(content="x", importance=5.0)
        r2 = wired_tool_impls.memory_store(content="y", importance=-1.0)
        assert 1.0 == r1["importance"]
        assert 0.0 == r2["importance"]

    def test_memory_store_empty_content_errors(self, wired_tool_impls):
        """空 content 应 error，不写库。"""
        result = wired_tool_impls.memory_store(content="")
        assert "error" in result

    def test_memory_recall_returns_facts_by_importance(self, wired_tool_impls):
        """memory_recall 按 importance 排序返回。"""
        wired_tool_impls.memory_store(content="低", importance=0.1, key="low")
        wired_tool_impls.memory_store(content="高", importance=0.9, key="high")
        wired_tool_impls.memory_store(content="中", importance=0.5, key="mid")

        result = wired_tool_impls.memory_recall(top_n=3)
        assert 3 == result["count"]
        values = [f["value"] for f in result["facts"]]
        assert "高" == values[0]  # importance 最高排首位

    def test_memory_recall_empty_db(self, wired_tool_impls):
        """空库 recall 应返回 count=0 而非崩。"""
        result = wired_tool_impls.memory_recall(top_n=5)
        assert 0 == result["count"]
        assert [] == result["facts"]

    def test_memory_recall_no_store_errors(self):
        """store provider 未注入应返回 error。"""
        from webui import tool_impls
        tool_impls.set_memory_store(None)
        result = tool_impls.memory_recall()
        assert "error" in result


class TestExternalTools:
    """web_search / get_weather —— 需外网，默认 skip；mock 模式只验证调用路径。"""

    @pytest.mark.skip(reason="web_search 依赖 cn.bing.com 外网；非 CI 跑可手动开")
    def test_web_search_returns_results(self):
        from webui import tool_impls
        result = tool_impls.web_search("python", max_results=2)
        assert "results" in result or "error" in result

    @pytest.mark.skip(reason="get_weather 依赖 wttr.in 外网；非 CI 跑可手动开")
    def test_get_weather_returns_temp(self):
        from webui import tool_impls
        result = tool_impls.get_weather(city="Beijing")
        assert "temp_c" in result or "error" in result

    def test_web_search_handles_network_failure(self):
        """模拟外网失败：urllib 抛错应转为 error dict。"""
        from webui import tool_impls
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            result = tool_impls.web_search("test")
        assert "error" in result

    def test_get_weather_handles_network_failure(self):
        """模拟 wttr.in 网络失败应返回 error。"""
        from webui import tool_impls
        with patch("requests.get", side_effect=OSError("dns fail")):
            result = tool_impls.get_weather(city="Beijing")
        assert "error" in result

    def test_web_search_parses_mock_html(self):
        """mock urlopen 返回固定 HTML，验证 happy-path parsing 不漂移（M4）。"""
        from unittest.mock import MagicMock
        from webui import tool_impls
        mock_html = (
            '<html><body><li class="b_algo">'
            '<h2><a href="x">Python 官网</a></h2>'
            '<p>Python is a programming language</p></li>'
            '<li class="b_algo"><h2>Real Python</h2>'
            '<p>tutorials and articles</p></li></body></html>'
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_html
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool_impls.web_search("python", max_results=2)
        assert "results" in result
        assert 2 == len(result["results"])
        assert "Python" in result["results"][0]["title"]
        assert "cn.bing.com" == result["source"]

    def test_get_weather_parses_mock_json(self):
        """mock wttr.in 返回固定 JSON，验证 happy-path parsing 不漂移（M4）。"""
        from unittest.mock import MagicMock
        from webui import tool_impls
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "current_condition": [{
                "temp_C": "20", "FeelsLikeC": "18",
                "weatherDesc": [{"value": "Sunny"}],
                "humidity": "60", "windspeedKmph": "5",
            }]
        }
        with patch("requests.get", return_value=mock_resp):
            result = tool_impls.get_weather(city="Beijing")
        assert "Beijing" == result["city"]
        assert "20" == result["temp_c"]
        assert "Sunny" == result["desc"]
        assert "60" == result["humidity_pct"]
        assert "wttr.in" == result["source"]


# ---------------------------------------------------------------------------
# 3. Memory L1 / L2 跨"会话"
# ---------------------------------------------------------------------------

class TestMemoryAcrossSessions:
    """L1 (turns) + L2 (persona_facts) 跨 session 持久化与 hydrate。"""

    def test_l1_recall_recent_turns_across_sessions(self, real_memory_store):
        """会话 1 写 turns → end_session → 会话 2 start_session 应能拿到上轮历史。"""
        store = real_memory_store

        # 会话 1
        sid1 = store.start_session(user_key="default")
        store.append_turn(sid1, "user", "你好")
        store.append_turn(sid1, "assistant", "你好呀")
        store.append_turn(sid1, "user", "今天天气怎么样")
        store.append_turn(sid1, "assistant", "晴天 22 度")
        store.end_session(sid1)

        # 会话 2 — 模拟 reconnect
        sid2 = store.start_session(user_key="default")
        recent = store.recall_recent_turns(user_key="default", max_turns=6)

        # 4 条 turn 应全部恢复（按时序正序）
        assert 4 == len(recent)
        assert "user" == recent[0]["role"]
        assert "你好" == recent[0]["content"]
        assert "assistant" == recent[-1]["role"]
        assert "晴天 22 度" == recent[-1]["content"]
        store.end_session(sid2)

    def test_l1_only_returns_user_assistant_roles(self, real_memory_store):
        """recall_recent_turns 应过滤掉 tool 角色（只回 user/assistant）。"""
        store = real_memory_store
        sid = store.start_session()
        store.append_turn(sid, "user", "查温度")
        store.append_turn(sid, "tool", '{"max_celsius": 60}')
        store.append_turn(sid, "assistant", "60 度")
        store.end_session(sid)

        recent = store.recall_recent_turns(max_turns=10)
        roles = [t["role"] for t in recent]
        assert "tool" not in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_l2_persona_facts_persist_across_sessions(self, real_memory_store):
        """L2 facts 跨 session 持久；新会话直接 get_persona_facts 可读到。"""
        store = real_memory_store

        sid1 = store.start_session()
        store.upsert_persona_fact("name", "小明", importance=0.9)
        store.upsert_persona_fact("hobby", "弹钢琴", importance=0.6)
        store.end_session(sid1)

        # 新 session 完全独立但 facts 仍可读
        sid2 = store.start_session()
        facts = store.get_persona_facts(top_n=5)
        assert 2 == len(facts)
        keys = {f["key"]: f["value"] for f in facts}
        assert "小明" == keys["name"]
        assert "弹钢琴" == keys["hobby"]
        store.end_session(sid2)

    def test_l2_upsert_updates_existing_key(self, real_memory_store):
        """同 key 重写应 update 而非新增（PRIMARY KEY ON CONFLICT）。"""
        store = real_memory_store
        store.upsert_persona_fact("name", "小明", importance=0.5)
        store.upsert_persona_fact("name", "小红", importance=0.9)
        facts = store.get_persona_facts(top_n=5)
        assert 1 == len(facts)
        assert "小红" == facts[0]["value"]
        assert 0.9 == facts[0]["importance"]

    def test_l1_l2_combined_hydrate_flow(self, real_memory_store, wired_tool_impls):
        """模拟完整 hydrate：会话 1 写 turn + 写 fact → 会话 2 同时 recall。"""
        store = real_memory_store

        # 会话 1：tool 路径写 L2 + 直接 API 写 L1
        sid1 = store.start_session()
        wired_tool_impls.memory_store(
            content="用户偏好深色主题", importance=0.7, key="theme",
        )
        store.append_turn(sid1, "user", "记住我喜欢深色")
        store.append_turn(sid1, "assistant", "好的，已记住")
        store.end_session(sid1)

        # 会话 2：双层都能恢复
        sid2 = store.start_session()
        l1 = store.recall_recent_turns(max_turns=6)
        l2_result = wired_tool_impls.memory_recall(top_n=3)

        assert len(l1) >= 2
        assert l2_result["count"] >= 1
        assert any("深色主题" in f["value"] for f in l2_result["facts"])
        store.end_session(sid2)


# ---------------------------------------------------------------------------
# 4. multi-step ReAct —— mock LLM endpoint 返回连续 tool_calls
# ---------------------------------------------------------------------------

def _sse_chunk(payload: dict) -> str:
    """构造一个 SSE data line。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n"


def _build_tool_call_round_lines(tool_name: str, args: str, idx: int = 0) -> list[str]:
    """mock 一轮 SSE：assistant 触发一个 tool_call。"""
    chunks = [
        # delta: 开始 tool_call (id + name)
        _sse_chunk({"choices": [{"delta": {"tool_calls": [{
            "index": idx, "id": f"call_{idx}", "type": "function",
            "function": {"name": tool_name, "arguments": ""},
        }]}}]}),
        # delta: 追加 arguments
        _sse_chunk({"choices": [{"delta": {"tool_calls": [{
            "index": idx, "function": {"arguments": args},
        }]}}]}),
        # finish_reason
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]\n",
    ]
    return chunks


def _build_final_content_lines(text: str) -> list[str]:
    """mock 一轮 SSE：assistant 输出最终 content + stop。"""
    return [
        _sse_chunk({"choices": [{"delta": {"content": text}}]}),
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]\n",
    ]


class _FakeSSEResponse:
    """模拟 requests.post(stream=True) 的 context manager。"""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self.encoding = "utf-8"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            for sub in line.split("\n"):
                if sub:
                    yield sub


class TestReActLoop:
    """multi-step ReAct loop —— mock LLM 返回 tool_call → final content 两轮。"""

    def test_two_round_react_calls_tool_then_answers(
        self, wired_tool_impls,
    ):
        """ReAct: 第 1 轮 tool_call(get_temperature) → 第 2 轮 final content。

        验证：
          - 收到 tool_call / tool_result 事件
          - 最后 yield 'done' + 完整 content
          - history in-place 累加 user/assistant/tool/assistant 共 4 条
        """
        from webui import chat_tools

        # 两轮 SSE：第一轮调 get_temperature；第二轮直接出 content
        round1 = _build_tool_call_round_lines("get_temperature", "{}")
        round2 = _build_final_content_lines("当前温度 71 度，注意散热。")
        responses = iter([_FakeSSEResponse(round1), _FakeSSEResponse(round2)])

        def fake_post(*args, **kwargs):
            return next(responses)

        history: list = []
        events = []
        with patch.object(chat_tools.requests, "post", side_effect=fake_post):
            for evt in chat_tools.chat_with_tools_stream(
                prompt="板子热不热", system="test system",
                max_tokens=100, disable_thinking=True, history=history,
            ):
                events.append(evt)

        kinds = [e[0] for e in events]
        # tool_call + tool_result + 至少一个 sentence + done
        assert "tool_call" in kinds
        assert "tool_result" in kinds
        assert "done" == kinds[-1]

        # tool_call 名字应是 get_temperature
        tc_evt = next(e for e in events if e[0] == "tool_call")
        assert "get_temperature" == tc_evt[1]["name"]

        # tool_result payload 是 dict
        tr_evt = next(e for e in events if e[0] == "tool_result")
        assert isinstance(tr_evt[1]["result"], dict)

        # history 累加：user / assistant(tool_calls) / tool / assistant(content)
        roles = [m["role"] for m in history]
        assert "user" == roles[0]
        assert "tool" in roles
        assert "assistant" == roles[-1]

    def test_react_immediate_answer_no_tool_call(self, wired_tool_impls):
        """LLM 直接返回 content（不调 tool）应单轮结束。"""
        from webui import chat_tools

        single = _build_final_content_lines("你好呀！")
        responses = iter([_FakeSSEResponse(single)])

        history: list = []
        events = []
        with patch.object(chat_tools.requests, "post",
                          side_effect=lambda *a, **k: next(responses)):
            for evt in chat_tools.chat_with_tools_stream(
                prompt="你好", system="s", max_tokens=50,
                disable_thinking=True, history=history,
            ):
                events.append(evt)

        kinds = [e[0] for e in events]
        assert "tool_call" not in kinds
        assert "done" == kinds[-1]
        assert 2 == len(history)  # user + assistant
        assert "assistant" == history[-1]["role"]

    def test_exec_tool_handles_invalid_json_args(self):
        """_exec_tool 收到非法 JSON args 应返回 error dict 不抛（TG5 容错）。"""
        from webui import chat_tools
        result = chat_tools._exec_tool("get_temperature", "{this is not json")
        assert "error" in result
        err = result["error"].lower()
        assert "json" in err or "decode" in err

    def test_exec_tool_unknown_tool_returns_error(self):
        """未知 tool name 应返回 error dict 不抛。"""
        from webui import chat_tools
        result = chat_tools._exec_tool("nonexistent_tool_xyz", "{}")
        assert "error" in result
        assert "unknown" in result["error"].lower()

    def test_react_max_iterations_guard(self, wired_tool_impls):
        """LLM 死循环调 tool 时应触发 MAX_ITERATIONS 保护并 fail loud。"""
        from webui import chat_tools

        # 永远返回 tool_call，迫使触发上限
        def fake_post(*args, **kwargs):
            return _FakeSSEResponse(
                _build_tool_call_round_lines("get_system_info", '{"metric":"cpu"}')
            )

        history: list = []
        events = []
        with patch.object(chat_tools.requests, "post", side_effect=fake_post):
            for evt in chat_tools.chat_with_tools_stream(
                prompt="无限循环", system="s", max_tokens=50,
                disable_thinking=True, history=history,
            ):
                events.append(evt)

        kinds = [e[0] for e in events]
        assert "done" == kinds[-1]
        # 至少 MAX_ITERATIONS 轮 tool_call
        tool_calls = [e for e in events if e[0] == "tool_call"]
        assert chat_tools.MAX_ITERATIONS == len(tool_calls)
        # 最后一个 sentence 是 fail-loud 文案
        last_sentence = next(
            (e for e in reversed(events) if e[0] == "sentence"), None,
        )
        assert last_sentence is not None
        # M1 收紧：契约绑定 chat_tools.py:244 fail-loud 文案 + MAX_ITERATIONS 字面
        assert "尝试调用" in last_sentence[1]
        assert str(chat_tools.MAX_ITERATIONS) in last_sentence[1]


# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
