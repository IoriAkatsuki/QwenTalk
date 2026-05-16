"""Agent 工具的 OpenAI tool_calls JSON schema 定义。

与 tools.py 的 TOOL_REGISTRY 函数签名一一对应。
分离的原因: tools.py 实现长，schema 和实现耦合度低，独立维护更清晰。
"""
from __future__ import annotations

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": (
            "Bing 中国版 HTML 搜索（板卡国内可达）。通用网页搜索 fallback；"
            "天气问题请优先用 get_weather。"
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "default": 3},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": (
            "查询指定城市当前天气（温度/体感/描述/湿度/风速）。"
            "用户问天气类问题一律调此工具。city 用英文或拼音，如 Beijing/Shanghai/Tokyo。"
        ),
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市英文/拼音"},
        }, "required": ["city"]}}},
    # ===== 板卡感知 / TTS / 记忆 工具（生产暴露）=====
    {"type": "function", "function": {
        "name": "get_gesture",
        "description": "读当前手势识别状态（D435 深度相机 + NPU 已聚合的最新帧）。返回 gesture/confidence/wrist_distance_m/palm_3d_m。用户问'我在干什么''看到我的手了吗'时调",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_distance",
        "description": "读用户距摄像头距离。region='user' 整体距离, region='hand' 手部距离。用户问'我离屏幕多远''能看到我吗'时调",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string", "enum": ["user", "hand"], "default": "user"},
        }}}},
    {"type": "function", "function": {
        "name": "get_scene",
        "description": "综合视觉感知描述视野（人 + 距离 + 手势 + 表情 + 是否看屏幕）。用户问'你看到什么''描述一下'时调",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "identify_user",
        "description": "读当前用户面部 + 表情 + 头姿（NPU 已检测）。用户问'你认得我吗''我现在什么表情'时调",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "speak",
        "description": "让你的声音说出 text（前端 Web Speech API 触发，独立于聊天文字流）。在你想主动开口、强调情绪、或对方让你'说出来听听'时调。text 限 100 字以内",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "要说出来的文字（≤ 100 字）"},
        }, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "memory_store",
        "description": "把一条用户事实写入长期记忆（L2 persona facts）。当用户告诉你他的名字/偏好/生日/工作等需要长期记住的信息时调。importance 0-1（关键信息如名字用 0.8+）",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string", "description": "要记住的事实，自然语言"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        }, "required": ["content"]}}},
    {"type": "function", "function": {
        "name": "memory_recall",
        "description": "查长期记忆（L2 persona facts），返回按重要性排序的最近 N 条。用户问'你还记得我吗''之前我说过什么'时调",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "暂未实现 query 过滤，传空即可", "default": ""},
            "top_n": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        }}}},
    # ===== ROADMAP: 下列 schema 未在 webui 生产暴露，路线图保留供未来扩展 =====
    # 若实现接入生产，需同步检查 description/parameters 是否与当时 API 漂移
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
    {"type": "function", "function": {
        "name": "memory_store",
        "description": "写入板端 SQLite 记忆库，支持 short_term/long_term 与指数衰减半衰期",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string", "description": "需要记住的事实、偏好、事件或任务状态"},
            "memory_type": {"type": "string", "enum": ["short_term", "long_term"], "default": "short_term"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            "source": {"type": "string", "default": "agent"},
            "half_life_hours": {"type": "number",
                                "description": "可选自定义半衰期；默认短期 24h，长期 720h"},
            "summary": {"type": "string", "default": ""},
            "metadata": {"type": "object", "default": {}},
        }, "required": ["content"]}}},
    {"type": "function", "function": {
        "name": "memory_recall",
        "description": "从板端 SQLite 记忆库检索相关记忆；分数包含相关性、重要性、访问强化和 exp 衰减",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索关键词或自然语言问题"},
            "memory_type": {"type": "string", "enum": ["any", "short_term", "long_term"], "default": "any"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "reinforce": {"type": "boolean", "default": True,
                          "description": "检索命中后是否强化记忆，模拟提取练习"},
            "include_decayed": {"type": "boolean", "default": False},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "memory_reinforce",
        "description": "显式强化一条记忆；高重要性或多次访问的短期记忆会固化为长期",
        "parameters": {"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "amount": {"type": "number", "minimum": 0, "maximum": 3, "default": 1.0},
        }, "required": ["memory_id"]}}},
    {"type": "function", "function": {
        "name": "memory_decay",
        "description": "清理已过期或保留率过低的 SQLite 记忆；默认只预览不删除",
        "parameters": {"type": "object", "properties": {
            "min_retention": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.02},
            "dry_run": {"type": "boolean", "default": True},
        }}}},
    {"type": "function", "function": {
        "name": "memory_status",
        "description": "查看板端 SQLite 记忆库统计、默认半衰期和数据库路径",
        "parameters": {"type": "object", "properties": {}}}},
]
