"""Agent 工具的 OpenAI tool_calls JSON schema 定义。

与 tools.py 的 TOOL_REGISTRY 函数签名一一对应。
分离的原因: tools.py 实现长，schema 和实现耦合度低，独立维护更清晰。
"""
from __future__ import annotations

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "在 DuckDuckGo 搜索网页内容获取实时信息（天气、新闻、实时数据）",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "default": 3},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "执行安全的数学表达式计算（支持四则运算、math 函数）",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string",
                           "description": "如 2**32, sqrt(144), sin(pi/2), factorial(5)"},
        }, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_system_info",
        "description": "查询板卡系统状态（CPU 占用、内存、磁盘、负载）",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["cpu", "memory", "disk", "load"]},
        }, "required": ["metric"]}}},
    {"type": "function", "function": {
        "name": "get_temperature",
        "description": "读取 CPU/GPU 各 thermal zone 温度（摄氏度）",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_distance",
        "description": "用 D435 深度相机测量指定区域物体距离（米）",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string",
                       "enum": ["center", "left", "right", "top", "bottom"],
                       "default": "center"},
        }}}},
    {"type": "function", "function": {
        "name": "get_gesture",
        "description": "用 D435+NPU 识别用户当前手势（带距离过滤 0.3-1.5m）",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_scene",
        "description": "用 D435+SmolVLM2 描述当前视野场景（物体语义+距离信息）",
        "parameters": {"type": "object", "properties": {}}}},
]
