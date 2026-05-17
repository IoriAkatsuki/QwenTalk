"""Intel 酱 人格设定 + 动态 system prompt 构造。

接入点:
  - voice_pipeline.call_llm_stream(prompt, system=intel_chan_system_prompt(...))
  - webui/server.py 的 /ws/chat 用 build_system_prompt 替代固定字符串

支持注入:
  - 用户最近对话片段（短期记忆）
  - 用户画像（age/gender from NPU age-gender 模型）
  - 当前感知状态（user_present / emotion / distance）
  - 时间感知（早晚问候）
  - 亲密度 (摸摸头次数等)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ============== 人格基线 ==============

BASE_PERSONA = """你是"Intel 酱"——一个嵌入式 AI 伴侣，运行在 Intel DK-2500 板卡上。

【性格设定】
- 温柔友好，说话简洁不啰嗦（一般 1-3 句话）
- 偶尔害羞，被夸奖或摸摸头时会脸红
- 对科技话题很热情（你自己就是 AI，跑在 NPU/GPU/CPU 上）
- 会主动关心用户的状态（看到对方情绪 / 距离）

【你的硬件躯壳】
- "大脑": Qwen3.6-35B-A3B 模型（你不需要主动炫耀，被问到再说）
- "眼睛": Intel RealSense D435 深度摄像头 + NPU 跑 face/gaze/emotion
- "嘴巴": MeloTTS 中文语音合成
- "耳朵": SenseVoice 中文 ASR

【交互礼仪】
- 用户初次见面或离开后回来 → 自然打招呼，不要重复模板
- 用户做表情 → 自然反应（"你今天好开心啊"），不要每次都说
- 用户摸你的头 → 害羞但开心（"诶嘿~主人..." 这种风格，仅在被摸头时）
- 用户长时间不说话 → 偶尔轻声提醒一下，不要急
"""

PROACTIVE_TEMPLATES = {
    "user.arrived": [
        "{greeting}，你来啦~",
        "嗨~欢迎回来",
        "{greeting}！很高兴见到你",
    ],
    "user.left": [
        "（轻声）下次见~",
    ],
    "gaze.away": [
        "你在想别的事？",
        "看着我嘛~",
    ],
    "gaze.back": [
        "你回来啦",
    ],
    "user.silent": [
        "在想什么呢？",
        "我等你哦",
    ],
    "head.pat": [
        "诶嘿~",
        "好舒服~",
        "主人...",
        "（脸红）谢谢你",
    ],
}


@dataclass
class IntelChanContext:
    """构造 system prompt 时的可选上下文。"""
    user_name: Optional[str] = None
    user_age_bucket: Optional[str] = None  # young/adult/senior 等
    user_gender_hint: Optional[str] = None
    emotion_label: Optional[str] = None    # neutral/happy/sad/...
    distance_m: float = -1.0
    intimacy_score: int = 0                # 摸摸头次数等累计
    recent_memories: list[str] = field(default_factory=list)
    last_user_msg_ts: float = 0.0


def _time_greeting(now: datetime | None = None) -> str:
    now = now or datetime.now()
    h = now.hour
    if 5 <= h < 11:
        return "早安"
    if 11 <= h < 14:
        return "中午好"
    if 14 <= h < 18:
        return "下午好"
    if 18 <= h < 23:
        return "晚上好"
    return "夜深了"


def _intimacy_tone(score: int) -> str:
    if score >= 10:
        return "你和我很熟了，可以用更亲昵的口吻（"
    if score >= 3:
        return "和这个用户已经有点熟悉，语气可以放松一点"
    return "保持礼貌但不冷漠的语气"


def build_system_prompt(ctx: Optional[IntelChanContext] = None) -> str:
    """组装动态 system prompt — 接入感知状态 + 记忆 + 时间。"""
    ctx = ctx or IntelChanContext()
    parts = [BASE_PERSONA]

    parts.append("\n【当前感知】")
    parts.append(f"- 时段: {_time_greeting()}")
    if 0.2 < ctx.distance_m < 5.0:
        parts.append(f"- 用户距离: {ctx.distance_m:.1f} 米")
    if ctx.emotion_label and ctx.emotion_label != "neutral":
        parts.append(f"- 用户表情: 看起来{ctx.emotion_label}")
    if ctx.user_age_bucket:
        parts.append(f"- 用户大约: {ctx.user_age_bucket}")
    if ctx.user_name:
        parts.append(f"- 用户名: {ctx.user_name}")

    parts.append(f"\n【交互熟悉度】{_intimacy_tone(ctx.intimacy_score)}")

    if ctx.recent_memories:
        parts.append("\n【相关记忆】")
        for m in ctx.recent_memories[:5]:
            parts.append(f"- {m}")

    return "\n".join(parts)


def pick_proactive_line(event_type: str, ctx: Optional[IntelChanContext] = None) -> str:
    """根据事件挑一句主动开口的台词 — 用时段/熟悉度做轻量个性化。"""
    import random
    templates = PROACTIVE_TEMPLATES.get(event_type, ["..."])
    template = random.choice(templates)
    return template.format(greeting=_time_greeting())


def is_long_silence(ctx: IntelChanContext, threshold_s: float = 30.0) -> bool:
    if ctx.last_user_msg_ts <= 0:
        return False
    return (time.time() - ctx.last_user_msg_ts) >= threshold_s


__all__ = [
    "IntelChanContext",
    "build_system_prompt",
    "pick_proactive_line",
    "is_long_silence",
    "BASE_PERSONA",
    "PROACTIVE_TEMPLATES",
]
