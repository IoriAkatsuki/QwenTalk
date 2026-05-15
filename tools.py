#!/usr/bin/env python3
"""Agent 可调用的 7 个工具 — Intel DK-2500 五引擎能力封装。

实现委托:
  - safe_eval.compute: AST 安全数学求值
  - resources.ensure_d435/ensure_npu/grab_frames: 重资源单例
  - tool_schemas.TOOL_SCHEMAS: OpenAI tool_calls schema
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from resources import ensure_d435, ensure_npu, grab_frames
from safe_eval import compute
from tool_schemas import TOOL_SCHEMAS
from edge_memory import (
    decay_memory,
    memory_stats,
    recall_memory,
    reinforce_memory,
    store_memory,
)


# ============================================================================
# 工具 1: web_search — DuckDuckGo 即时问答
# ============================================================================
def web_search(query: str, max_results: int = 3) -> dict:
    """DuckDuckGo 即时问答 + 相关主题。"""
    import urllib.parse
    import requests
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
        data = requests.get(url, timeout=10).json()
        results = []
        if data.get("Abstract"):
            results.append({"title": data.get("Heading", ""), "snippet": data["Abstract"]})
        for t in (data.get("RelatedTopics") or [])[:max_results - len(results)]:
            if isinstance(t, dict) and t.get("Text"):
                results.append({"title": t.get("FirstURL", "").split("/")[-1],
                                "snippet": t["Text"]})
        return {"query": query, "results": results[:max_results] or [{"snippet": "无结果"}]}
    except Exception as e:
        return {"error": f"搜索失败: {e}"}


# ============================================================================
# 工具 2: calculate — AST 安全数学求值
# ============================================================================
def calculate(expression: str) -> dict:
    """AST 安全数学求值（仅算术 + math 子集白名单）。"""
    try:
        return {"expression": expression, "result": compute(expression)}
    except Exception as e:
        return {"error": f"计算失败: {e}"}


# ============================================================================
# 工具 3: get_system_info — psutil
# ============================================================================
def get_system_info(metric: str) -> dict:
    """获取系统状态: cpu/memory/disk/load。"""
    try:
        import psutil
        if metric == "cpu":
            return {"cpu_percent": psutil.cpu_percent(interval=0.5),
                    "cpu_count": psutil.cpu_count(),
                    "freq_mhz": int(psutil.cpu_freq().current) if psutil.cpu_freq() else 0}
        if metric == "memory":
            m = psutil.virtual_memory()
            return {"total_gb": round(m.total / 1e9, 1),
                    "used_gb": round(m.used / 1e9, 1),
                    "available_gb": round(m.available / 1e9, 1),
                    "percent": m.percent}
        if metric == "disk":
            d = psutil.disk_usage("/")
            return {"total_gb": round(d.total / 1e9, 1),
                    "free_gb": round(d.free / 1e9, 1),
                    "percent": d.percent}
        if metric == "load":
            return {"load_1min": round(psutil.getloadavg()[0], 2),
                    "load_5min": round(psutil.getloadavg()[1], 2)}
        return {"error": f"未知 metric: {metric}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 工具 4: get_temperature — sysfs thermal zones
# ============================================================================
def get_temperature() -> dict:
    """通过 sysfs 读取 thermal_zone 温度。"""
    try:
        zones = {}
        for p in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                temp = int((p / "temp").read_text().strip()) / 1000.0
                name = (p / "type").read_text().strip()
                zones[name] = round(temp, 1)
            except Exception:
                continue
        if not zones:
            return {"error": "无 thermal_zone 数据"}
        return {"zones_celsius": zones, "max_celsius": max(zones.values())}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 工具 5: get_distance — D435 区域距离
# ============================================================================
_REGIONS = {
    "center": lambda h, w: (h // 2 - 30, h // 2 + 30, w // 2 - 30, w // 2 + 30),
    "left":   lambda h, w: (h // 2 - 30, h // 2 + 30, 0, 60),
    "right":  lambda h, w: (h // 2 - 30, h // 2 + 30, w - 60, w),
    "top":    lambda h, w: (0, 60, w // 2 - 30, w // 2 + 30),
    "bottom": lambda h, w: (h - 60, h, w // 2 - 30, w // 2 + 30),
}


def get_distance(region: str = "center") -> dict:
    """D435 测量指定区域距离（center/left/right/top/bottom），单位米。"""
    try:
        if region not in _REGIONS:
            return {"error": f"未知 region: {region}"}
        _, depth_m = grab_frames()
        h, w = depth_m.shape
        y0, y1, x0, x1 = _REGIONS[region](h, w)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[(patch > 0.1) & (patch < 8.0)]
        if len(valid) < 100:
            return {"region": region, "distance_m": None, "note": "深度数据不足"}
        return {"region": region,
                "distance_m": round(float(np.median(valid)), 3),
                "min_m": round(float(np.min(valid)), 3)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 工具 6: get_gesture — 单帧 NPU 手势识别 + 距离过滤
# ============================================================================
def get_gesture() -> dict:
    """检测当前手势（带距离闸门 0.3-1.5m）。"""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from gesture3d_test import classify_gesture, DIST_MIN_M, DIST_MAX_M, WRIST
        import pyrealsense2 as rs

        npu = ensure_npu()
        color, depth_m = grab_frames()
        h, w = color.shape[:2]
        margin = min(h, w) // 4
        crop = color[margin:h - margin, margin:w - margin]

        inp = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        result = npu["hand"](inp[np.newaxis])
        landmarks = result[npu["hand"].output(0)].reshape(-1, 3)

        pts_3d = np.zeros((21, 3), dtype=np.float32)
        d = ensure_d435()
        for i in range(min(21, landmarks.shape[0])):
            px = int(np.clip(landmarks[i][0] * (w - 2 * margin) + margin, 0, w - 1))
            py = int(np.clip(landmarks[i][1] * (h - 2 * margin) + margin, 0, h - 1))
            d_m = float(depth_m[py, px])
            if 0.05 < d_m < 5.0:
                pts_3d[i] = rs.rs2_deproject_pixel_to_point(
                    d["intrinsics"], [float(px), float(py)], d_m,
                )
        wrist_dist = float(pts_3d[WRIST][2]) if pts_3d[WRIST][2] > 0 else -1.0
        gesture, conf = classify_gesture(
            landmarks, pts_3d=pts_3d,
            wrist_distance=wrist_dist if wrist_dist > 0 else None,
        )
        return {"gesture": gesture, "confidence": round(float(conf), 2),
                "wrist_distance_m": round(wrist_dist, 3) if wrist_dist > 0 else None,
                "valid_range_m": [DIST_MIN_M, DIST_MAX_M]}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 工具 7: get_scene — D435 + SmolVLM2 场景理解
# ============================================================================
def get_scene(vlm_url: str = "http://127.0.0.1:8081/completion") -> dict:
    """D435 RGB-D + SmolVLM2 场景理解。"""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from scene3d import depth_stats_grid, closest_region, build_depth_prompt
        import requests

        color, depth_m = grab_frames()
        cells = depth_stats_grid(depth_m)
        closest = closest_region(depth_m)
        depth_ctx = build_depth_prompt(cells, closest)

        _, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(jpeg.tobytes()).decode()
        prompt = (
            f"<|im_start|>User: <__image__>{depth_ctx}\n"
            "结合图像和深度信息，用中文一句话描述场景。"
            f"<end_of_utterance>\n<|im_start|>Assistant:"
        )
        r = requests.post(vlm_url, json={
            "prompt": prompt, "image_data": [{"data": b64, "id": 1}],
            "n_predict": 80, "temperature": 0.3,
            "stop": ["<end_of_utterance>"],
        }, timeout=30)
        scene = r.json().get("content", "").strip() if r.ok else f"VLM HTTP {r.status_code}"
        return {"scene": scene,
                "closest_distance_m": round(closest[2], 3) if closest else None,
                "depth_summary": {f"{c['row']},{c['col']}": round(c['median_m'], 2)
                                  for c in cells if c['median_m'] > 0}}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 工具 8-11: edge memory — SQLite 短/长期记忆 + 指数衰减
# ============================================================================
def memory_store(
    content: str,
    memory_type: str = "short_term",
    importance: float = 0.5,
    tags: list[str] | None = None,
    source: str = "agent",
    half_life_hours: float | None = None,
    summary: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict:
    """写入 SQLite 记忆。短期/长期用不同默认半衰期。"""
    return store_memory(
        content=content,
        memory_type=memory_type,
        importance=importance,
        tags=tags,
        source=source,
        half_life_hours=half_life_hours,
        summary=summary,
        metadata=metadata,
    )


def memory_recall(
    query: str,
    memory_type: str = "any",
    top_k: int = 5,
    reinforce: bool = True,
    include_decayed: bool = False,
) -> dict:
    """按相关性、重要性、访问强化和 exp 衰减检索记忆。"""
    return recall_memory(
        query=query,
        memory_type=memory_type,
        top_k=top_k,
        reinforce=reinforce,
        include_decayed=include_decayed,
    )


def memory_reinforce(memory_id: int, amount: float = 1.0) -> dict:
    """显式强化一条记忆；高重要性或多次访问的短期记忆会固化为长期。"""
    return reinforce_memory(memory_id=memory_id, amount=amount)


def memory_decay(min_retention: float = 0.02, dry_run: bool = True) -> dict:
    """清理已过期或保留率过低的记忆；默认 dry_run。"""
    return decay_memory(min_retention=min_retention, dry_run=dry_run)


def memory_status() -> dict:
    """查看 SQLite 记忆库统计信息。"""
    return memory_stats()


# ============================================================================
# 工具注册 + 分发
# ============================================================================
TOOL_REGISTRY: dict[str, Any] = {
    "web_search": web_search, "calculate": calculate,
    "get_system_info": get_system_info, "get_temperature": get_temperature,
    "get_distance": get_distance, "get_gesture": get_gesture, "get_scene": get_scene,
    "memory_store": memory_store, "memory_recall": memory_recall,
    "memory_reinforce": memory_reinforce, "memory_decay": memory_decay,
    "memory_status": memory_status,
}


def call_tool(name: str, arguments: dict) -> Any:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"未知工具: {name}"}
    return fn(**(arguments or {}))


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("用法: python tools.py <tool_name> [json_args]")
        print(f"可用: {', '.join(TOOL_REGISTRY)}")
        sys.exit(0)
    print(json.dumps(call_tool(sys.argv[1],
                                json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}),
                     ensure_ascii=False, indent=2))
