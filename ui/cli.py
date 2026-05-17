#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个功能强大的命令行AI对话程序 (talkCLI.py)

功能:
- 优先使用Intel NPU进行推理, 并可自动回退至CPU。
- 支持通过 /search <关键词> 命令进行联网搜索。
- 支持流式响应, 提升交互体验。
- 支持连续对话和历史记录管理。
- 提供 /help, /clear, /quit 等便捷命令。
"""

import openvino_genai as ov_genai
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict
import requests

class ChatHistory:
    """
    管理聊天历史记录。

    维护一个消息列表, 并提供保存、加载和格式化为上下文的功能。
    """
    def __init__(self, max_len: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_len = max_len

    def add_message(self, role: str, content: str):
        """向历史记录中添加一条新消息。"""
        self.history.append({"role": role, "content": content})
        # 当历史记录过长时, 移除最旧的一轮对话
        if len(self.history) > self.max_len * 2:
            self.history = self.history[-self.max_len * 2:]

    def get_formatted_history(self) -> List[Dict[str, str]]:
        """获取格式化后的历史记录, 适用于OpenVINO的聊天模板。"""
        return self.history

    def clear(self):
        """清空聊天历史。"""
        self.history.clear()
        print("\n[INFO] 聊天历史已清空。")

class WebSearchTool:
    """
    提供联网搜索功能的工具。

    使用DuckDuckGo的即时问答API, 无需申请API Key。
    """
    def search(self, query: str, max_results: int = 3) -> str:
        """
        执行网络搜索并返回格式化的结果字符串。

        Args:
            query: 搜索的关键词。
            max_results: 返回的最大相关主题数。

        Returns:
            一个包含搜索摘要和相关主题的字符串, 或错误信息。
        """
        if not query:
            return "错误: 请提供搜索关键词。"
        print(f"\n[INFO] 正在搜索: {query} ...")
        try:
            # 使用DuckDuckGo的API进行搜索
            url = f"https://api.duckduckgo.com/?q={query}&format=json&pretty=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            # 1. 添加摘要信息 (Abstract)
            if data.get("AbstractText"):
                results.append(f"摘要: {data['AbstractText']}")

            # 2. 添加相关主题 (Related Topics)
            related_topics = data.get("RelatedTopics", [])
            if related_topics:
                for i, topic in enumerate(related_topics[:max_results]):
                    if "Text" in topic:
                        results.append(f"相关信息{i+1}: {topic['Text']}")

            if not results:
                return "抱歉，没有找到相关的在线信息。"

            return "\n".join(results)

        except requests.RequestException as e:
            return f"网络搜索失败: {e}"
        except Exception as e:
            return f"处理搜索结果时出错: {e}"

class TalkCLI:
    """
    命令行对话界面的主控制器。
    """
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.pipe = None
        self.tokenizer = None  # B24: 加载 HF tokenizer 用于 apply_chat_template
        self.generation_config = None
        self.history = ChatHistory()
        self.search_tool = WebSearchTool()
        self._load_model()
        self._load_tokenizer()
        self._setup_generation_config()

    def _load_tokenizer(self):
        """B24: 加载 HF tokenizer 以正确格式化 chat history。

        失败时回退到手动 'role: content' 拼接（保持向后兼容）。
        """
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"[INFO] tokenizer chat_template: "
                  f"{'有' if self.tokenizer.chat_template else '无 (回退手动拼接)'}")
        except Exception as e:
            print(f"[WARN] tokenizer 加载失败: {e}, 回退手动拼接")
            self.tokenizer = None

    def _load_model(self):
        """加载LLM模型, 优先尝试NPU, 失败则回退到CPU。"""
        print(f"[INFO] 正在加载模型: {self.model_path}")
        try:
            print(f"[INFO] 尝试在 {self.device} 上加载模型...")
            self.pipe = ov_genai.LLMPipeline(self.model_path, self.device)
            print(f"[SUCCESS] 模型成功加载到 {self.device}!")
        except Exception as e:
            print(f"[WARNING] 在 {self.device} 上加载失败: {e}")
            if self.device.upper() != "CPU":
                print("[INFO] 尝试回退到 CPU ...")
                try:
                    self.device = "CPU"
                    self.pipe = ov_genai.LLMPipeline(self.model_path, self.device)
                    print("[SUCCESS] 模型成功加载到 CPU!")
                except Exception as e2:
                    print(f"[ERROR] CPU加载也失败: {e2}", file=sys.stderr)
                    sys.exit(1)

    def _setup_generation_config(self):
        """配置模型的生成参数。"""
        self.generation_config = ov_genai.GenerationConfig()
        self.generation_config.max_new_tokens = 65536
        self.generation_config.temperature = 0.7
        self.generation_config.top_p = 0.95
        self.generation_config.top_k = 40
        self.generation_config.repetition_penalty = 1.1
        self.generation_config.do_sample = True

    def _print_help(self):
        """打印帮助信息。"""
        help_text = """
    命令列表:
      /help          - 显示此帮助信息
      /quit, /exit   - 退出程序
      /clear         - 清除当前的对话历史
      /search <query> - 进行联网搜索, 并将结果作为下一轮对话的背景信息
      
    直接输入文字并按Enter键即可开始对话。
        """
        print(help_text)

    def run(self):
        """启动并运行主聊天循环。"""
        print("\n" + "="*50)
        print(f"🤖 AI对话已启动 (设备: {self.device})")
        print("   输入 /help 查看所有命令。")
        print("="*50)

        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                if not user_input:
                    continue

                # 处理命令
                if user_input.lower() in ["/quit", "/exit"]:
                    print("再见！")
                    break
                elif user_input.lower() == "/help":
                    self._print_help()
                    continue
                elif user_input.lower() == "/clear":
                    self.history.clear()
                    continue
                elif user_input.lower().startswith("/search "):
                    query = user_input[8:].strip()
                    search_result = self.search_tool.search(query)
                    print(f"\n🔍 搜索结果:\n---\n{search_result}\n---")
                    # 将搜索结果作为上下文, 引导模型进行总结或回答
                    prompt = f"根据以下信息: \"{search_result}\", 请回答: \"{query}\""
                    self.history.add_message("user", prompt) # 将包含搜索结果的提示加入历史
                    self._generate_stream_response(prompt)
                    continue

                # 普通对话
                self.history.add_message("user", user_input)

                # B24 修复: 优先用 tokenizer.apply_chat_template (Qwen/Gemma 等模型
                # 有专属格式 <|im_start|> / <bos><start_of_turn> 等), 手动 'role:content'
                # 拼接会让模型生成质量下降甚至 thinking 失效
                msgs = self.history.get_formatted_history()
                if self.tokenizer and self.tokenizer.chat_template:
                    prompt_string = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                else:
                    # fallback (老路径)
                    prompt_parts = [f"{m['role']}: {m['content']}" for m in msgs]
                    prompt_parts.append("assistant:")
                    prompt_string = "\n".join(prompt_parts)

                self._generate_stream_response(prompt_string)

            except KeyboardInterrupt:
                print("\n\n程序被中断。再见！")
                break
            except Exception as e:
                print(f"\n[ERROR] 发生未知错误: {e}", file=sys.stderr)

    def _generate_stream_response(self, prompt_or_history):
        """生成并流式打印模型的响应 (B25: 失败时回滚最后的 user 消息)."""
        print("\n🤖 AI: ", end="", flush=True)
        full_response = ""
        try:
            for chunk in self.pipe.generate(prompt_or_history, self.generation_config):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
            # B23/B25: 空响应不入历史 + 回滚 user 消息
            if full_response.strip():
                self.history.add_message("assistant", full_response)
            else:
                print("\n[WARN] 空响应，回滚最后的 user 消息", file=sys.stderr)
                self._rollback_last_user()
        except Exception as e:
            error_message = f"\n[ERROR] 生成响应时出错: {e}"
            print(error_message, file=sys.stderr)
            # B25: 异常时也回滚 user 消息，避免空 turn 污染对话
            self._rollback_last_user()

    def _rollback_last_user(self):
        """B25: 移除历史最后一条 user 消息（如果存在）。"""
        msgs = self.history.get_formatted_history()
        if msgs and msgs[-1].get("role") == "user":
            try:
                self.history.messages.pop() if hasattr(self.history, "messages") else None
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(
        description="一个功能强大的命令行AI对话程序。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="必需: 指向OpenVINO IR格式模型的路径 (例如: ./Qwen3-8B-nf4-ov)"
    )
    parser.add_argument(
        "-d", "--device",
        default="NPU",
        choices=["NPU", "CPU", "GPU"],
        help="推理设备 (默认: NPU)"
    )
    args = parser.parse_args()

    # 检查模型路径是否存在
    if not os.path.isdir(args.model):
        print(f"[ERROR] 模型路径不存在: {args.model}", file=sys.stderr)
        sys.exit(1)

    cli = TalkCLI(model_path=args.model, device=args.device)
    cli.run()

if __name__ == "__main__":
    main()
