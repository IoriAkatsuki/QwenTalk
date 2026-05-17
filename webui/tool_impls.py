"""LLM tool 实现 — in-process function calling，9 个工具集中维护。

依赖 server.py lifespan 注入的 3 个 provider：pipeline / perception / memory_store。
各 tool 接受 dict args，返回 dict result（错误也是 dict 不抛异常 — 给模型读）。
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse
import urllib.request

import requests

WTTR_TIMEOUT_S = 8.0
BING_TIMEOUT_S = 8.0
BING_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
SPEAK_MAX_CHARS = 100
MEMORY_RECALL_MAX = 20

# Providers — server.py lifespan 注入；未注入时各 tool 自报 error 不崩
_pipeline = None
_perception = None
_memory_store = None


def set_pipeline(p) -> None:
    global _pipeline
    _pipeline = p


def set_perception(p) -> None:
    global _perception
    _perception = p


def set_memory_store(m) -> None:
    global _memory_store
    _memory_store = m


# ============== 网络 / 外部 API ==============

def web_search(query: str, max_results: int = 3) -> dict:
    """Bing 中国版 HTML 抓取（板卡国内可达；DDG/SearXNG 不可用的 fallback）。"""
    try:
        url = f"https://cn.bing.com/search?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": BING_UA})
        html = urllib.request.urlopen(req, timeout=BING_TIMEOUT_S).read()
        html = html.decode("utf-8", errors="ignore")
        blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
        results = []
        for b in blocks[:max_results]:
            h2 = re.search(r"<h2[^>]*>(.*?)</h2>", b, re.S)
            p = re.search(r"<p[^>]*>(.*?)</p>", b, re.S)
            title = re.sub(r"<[^>]+>", "", h2.group(1) if h2 else "")
            title = re.sub(r"&ensp;|&nbsp;|&amp;", " ", title)
            title = re.sub(r"\s+", " ", title).strip()
            snippet = re.sub(r"<[^>]+>", "", p.group(1) if p else "")
            snippet = re.sub(r"&ensp;|&#0183;|&nbsp;|&amp;", " ", snippet)
            snippet = re.sub(r"\s+", " ", snippet)[:200]
            if snippet or title:
                results.append({"title": title[:80], "snippet": snippet})
        if not results:
            return {"error": "Bing 未返回结果"}
        return {"query": query, "results": results, "source": "cn.bing.com"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Bing 失败: {type(e).__name__}: {e}"}


def get_weather(city: str = "Beijing") -> dict:
    """wttr.in 天气查询（无 KEY，板卡国内可达）。"""
    try:
        r = requests.get(
            f"https://wttr.in/{city}?format=j1", timeout=WTTR_TIMEOUT_S,
            headers={"User-Agent": "curl/7.0"},
        )
        r.raise_for_status()
        d = r.json()
        c = d["current_condition"][0]
        return {
            "city": city, "temp_c": c["temp_C"],
            "feels_like_c": c["FeelsLikeC"],
            "desc": c["weatherDesc"][0]["value"],
            "humidity_pct": c["humidity"],
            "wind_kmh": c.get("windspeedKmph", "n/a"),
            "source": "wttr.in",
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"wttr.in 失败: {type(e).__name__}: {e}"}


# ============== 系统状态 ==============

def get_temperature() -> dict:
    """读 thermal_zone 温度 — 包装 tools.py 同名函数。"""
    from tools import get_temperature as _impl
    return _impl()


def get_system_info(metric: str) -> dict:
    """系统状态 (cpu/memory/disk/load) — 包装 tools.py。"""
    from tools import get_system_info as _impl
    return _impl(metric)


# ============== D435 + NPU 感知 ==============

def get_gesture() -> dict:
    """读当前手势识别状态（D435 + NPU pipeline 已聚合的最新帧）。"""
    if _perception is None:
        return {"error": "perception 未注入"}
    snap = _perception.snapshot()
    h = snap.hand
    return {
        "gesture": h.gesture,
        "confidence": round(h.conf, 2),
        "wrist_distance_m": round(h.wrist_m, 2) if h.wrist_m > 0 else None,
        "palm_score": round(h.palm_score, 2),
        "palm_3d_m": h.palm_3d,
    }


def get_distance(region: str = "user") -> dict:
    """读用户距摄像头距离。region: 'user' (整体) | 'hand' (手部)。

    简化版（不做 D435 9 宫格细分，避免占用 perception 线程的 D435）。
    """
    if _perception is None:
        return {"error": "perception 未注入"}
    snap = _perception.snapshot()
    d = snap.hand.wrist_m if region == "hand" else snap.distance_m
    if d <= 0:
        return {"region": region, "distance_m": None, "note": "未检测到距离"}
    return {"region": region, "distance_m": round(d, 2)}


def get_scene() -> dict:
    """综合 perception 当前状态描述视野内的人和动作（不调 VLM，避免硬件竞争）。"""
    if _perception is None:
        return {"error": "perception 未注入"}
    snap = _perception.snapshot()
    parts = []
    if snap.user_present:
        if snap.distance_m > 0:
            parts.append(f"用户在 {snap.distance_m:.1f} 米处")
        else:
            parts.append("用户在视野内")
        if snap.hand.gesture and snap.hand.gesture != "init":
            parts.append(f"手势 {snap.hand.gesture}（置信度 {snap.hand.conf:.0%}）")
        if snap.emotion.label != "neutral":
            parts.append(f"表情 {snap.emotion.label}")
        if snap.face.bbox is not None:
            parts.append("已识别面部")
        if snap.gaze.on_screen:
            parts.append("正在看屏幕")
    else:
        parts.append("视野内没有人")
    return {
        "description": "；".join(parts) or "无观察",
        "user_present": snap.user_present,
        "raw": {
            "distance_m": snap.distance_m,
            "gesture": snap.hand.gesture,
            "emotion": snap.emotion.label,
        },
    }


def identify_user() -> dict:
    """读当前用户面部 + 表情 + 头姿（NPU age/gender 待接入时返回占位）。"""
    if _perception is None:
        return {"error": "perception 未注入"}
    snap = _perception.snapshot()
    face = snap.face
    return {
        "face_detected": face.bbox is not None,
        "head_pose_deg": {
            "yaw": round(face.yaw_deg, 1),
            "pitch": round(face.pitch_deg, 1),
            "roll": round(face.roll_deg, 1),
        },
        "emotion": snap.emotion.label,
        "emotion_confidence": round(snap.emotion.confidence, 2),
        "gaze_on_screen": snap.gaze.on_screen,
        "note": "age/gender 占位（NPU age-gender-recognition 待接入）",
    }


# ============== TTS / 主动输出 ==============

def speak(text: str) -> dict:
    """通过 perception WS 事件让前端用 Web Speech API 说出 text。

    text 限 100 字以内。前端 onmessage 收到 event="speak:|<text>" 即触发 utterance。
    """
    if _perception is None:
        return {"error": "perception 未注入"}
    text = (text or "").strip()
    if not text:
        return {"error": "empty text"}
    if len(text) > SPEAK_MAX_CHARS:
        text = text[:SPEAK_MAX_CHARS]
    _perception.push_event(f"speak:|{text}")
    return {"queued": True, "text": text}


# ============== L2 长期记忆 ==============

def memory_store(content: str, importance: float = 0.5,
                 key: str | None = None) -> dict:
    """把一条事实写入 L2 persona facts；模型可自决记关键信息（用户偏好/姓名等）。"""
    if _memory_store is None:
        return {"error": "memory_store 未注入"}
    content = (content or "").strip()
    if not content:
        return {"error": "empty content"}
    importance = max(0.0, min(1.0, float(importance)))
    if not key:
        key = (content[:30] + ":"
               + hashlib.md5(content.encode()).hexdigest()[:6])
    _memory_store.upsert_persona_fact(key, content, importance)
    return {"stored": True, "key": key, "importance": importance}


def memory_recall(query: str = "", top_n: int = 5) -> dict:
    """查 L2 persona facts。当前简化：返回 top-N by importance（无 query filter）。"""
    if _memory_store is None:
        return {"error": "memory_store 未注入"}
    top_n = max(1, min(MEMORY_RECALL_MAX, int(top_n)))
    facts = _memory_store.get_persona_facts(top_n=top_n)
    return {
        "query": query, "count": len(facts),
        "facts": [{"key": f["key"], "value": f["value"],
                   "importance": f.get("importance", 0.5)} for f in facts],
    }
