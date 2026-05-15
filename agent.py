#!/usr/bin/env python3
"""Qwen3.6 Agent — 多轮工具调用循环。

用法:
  # 1. 启动 LLM server (Qwen3.6 IQ2_M @ 8080)
  source /opt/intel/oneapi/setvars.sh
  ~/llama.cpp/build_sycl/bin/llama-server -m ~/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \\
      -ngl 99 -fa 0 --cache-type-k q8_0 --cache-type-v f16 \\
      -c 8192 --port 8080 --host 127.0.0.1 &

  # 2. (可选, get_scene 才需要) SmolVLM2 server @ 8081
  ~/llama.cpp/build_sycl/bin/llama-server -m ~/models/smolvlm2-2.2b-q4km.gguf \\
      --mmproj ~/models/smolvlm2-2.2b-mmproj-f16.gguf -ngl 0 -t 8 -c 2048 \\
      --port 8081 --host 127.0.0.1 &

  # 3. 跑 agent
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate openvino
  python agent.py                     # 交互式 REPL
  python agent.py "前面有什么物体？"   # 单次问答
  python agent.py --list-tools        # 列出可用工具
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests

from tools import TOOL_REGISTRY, TOOL_SCHEMAS, call_tool

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_TOOL_TURNS = 4
SYSTEM_PROMPT = (
    "你是 Intel DK-2500 边缘 AI 平台的智能助手。"
    "你可以调用以下工具获取实时信息：摄像头测距/手势/场景、系统状态/温度、网络搜索、数学计算。"
    "原则：能用工具时优先调用工具，回答简洁中文，不要展示思考过程。"
)


def health_check() -> bool:
    """LLM server 必须在线，VLM 可选。"""
    try:
        r = requests.get(LLM_URL.replace("/v1/chat/completions", "/health"), timeout=3)
        return r.ok
    except Exception:
        return False


def chat_once(messages: list[dict], *, use_tools: bool = True,
              max_tokens: int = 512, temperature: float = 0.3) -> dict:
    """一次 LLM 调用（关 thinking + 可选 tools）。"""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if use_tools:
        payload["tools"] = TOOL_SCHEMAS
        payload["tool_choice"] = "auto"
    r = requests.post(LLM_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def agent_loop(user_msg: str, *, verbose: bool = True) -> str:
    """多轮工具调用 → 最终答案。"""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    for turn in range(MAX_TOOL_TURNS):
        t0 = time.time()
        msg = chat_once(messages)
        elapsed = time.time() - t0

        tool_calls = msg.get("tool_calls") or []
        if verbose:
            content_preview = (msg.get("content") or "").strip()[:80]
            tools_preview = ", ".join(tc["function"]["name"] for tc in tool_calls) or "(无)"
            print(f"  [turn {turn+1}] {elapsed:.1f}s | tools={tools_preview} | content={content_preview!r}")

        if not tool_calls:
            return (msg.get("content") or "").strip() or "[空回复]"

        # 把 assistant 工具调用加进历史 (必须包含 tool_calls 字段)
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tool_calls})

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            t_tool = time.time()
            result = call_tool(name, args)
            t_tool = time.time() - t_tool
            if verbose:
                preview = json.dumps(result, ensure_ascii=False)[:120]
                print(f"      → {name}({args}) [{t_tool:.1f}s] = {preview}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", name),
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "[超过最大工具轮数]"


def repl():
    """交互式 REPL。"""
    print("Qwen3.6 Agent REPL — Ctrl+D 或 'exit' 退出")
    print(f"可用工具: {', '.join(TOOL_REGISTRY)}\n")
    while True:
        try:
            user_msg = input("👤 ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg or user_msg.lower() in ("exit", "quit"):
            break
        try:
            answer = agent_loop(user_msg)
            print(f"🤖 {answer}\n")
        except Exception as e:
            print(f"⚠ {e}\n")


def main():
    ap = argparse.ArgumentParser(description="Qwen3.6 Agent — 7 工具循环")
    ap.add_argument("question", nargs="?", help="单次问答 (省略进 REPL)")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--llm-url", default=LLM_URL)
    args = ap.parse_args()

    global LLM_URL
    LLM_URL = args.llm_url

    if args.list_tools:
        for s in TOOL_SCHEMAS:
            f = s["function"]
            params = list(f.get("parameters", {}).get("properties", {}).keys())
            print(f"  {f['name']:20s} - {f['description']} (参数: {params or '无'})")
        return

    if not health_check():
        print(f"❌ LLM server 不在线: {LLM_URL}")
        print("   启动: ~/llama.cpp/build_sycl/bin/llama-server -m Qwen3.6-IQ2_M.gguf -ngl 99 -fa 0 ...")
        sys.exit(1)

    if args.question:
        answer = agent_loop(args.question, verbose=not args.quiet)
        print(f"\n🤖 {answer}")
    else:
        repl()


if __name__ == "__main__":
    main()
