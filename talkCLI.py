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
import subprocess

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

    通过调用 open-websearch MCP 工具执行搜索。
    """
    def search(self, query: str) -> str:
        """
        执行网络搜索并返回格式化的结果字符串。

        Args:
            query: 搜索的关键词。

        Returns:
            一个包含搜索结果的字符串, 或错误信息。
        """
        if not query:
            return "错误: 请提供搜索关键词。"
        print(f"\n[INFO] 正在通过 open-websearch 搜索: {query} ...")
        try:
            # 构建命令
            # 使用 npx 调用最新的 open-websearch 包
            command = [
                "npx", "-y", "open-websearch@latest",
                "--engine", "duckduckgo", # 根据用户之前的参考信息，使用duckduckgo
                "--query", query
            ]

            # 执行命令
            # capture_output=True 将 stdout 和 stderr 捕获到 result.stdout 和 result.stderr
            # text=True 将 stdout 和 stderr 解码为文本
            # check=True 如果命令返回非零退出码 (表示错误), 则会引发 CalledProcessError
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                shell=True, # 在Windows上使用shell=True可能有助于找到npx
                encoding='utf-8'
            )

            # 返回 stdout 的内容
            if not result.stdout.strip():
                return "抱歉，没有找到相关的在线信息。"
            return result.stdout.strip()

        except FileNotFoundError:
            return "错误: `npx` 命令未找到。请确保您已安装 Node.js。"
        except subprocess.CalledProcessError as e:
            # 如果命令执行失败, 返回 stderr 的内容
            return f"搜索时出错: {e.stderr}"
        except Exception as e:
            return f"处理搜索结果时发生未知错误: {e}"

class TalkCLI:
    """
    命令行对话界面的主控制器。
    """
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.pipe = None
        self.generation_config = None
        self.history = ChatHistory()
        self.search_tool = WebSearchTool()
        self._load_model()
        self._setup_generation_config()

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
                
                # --- FIX START ---
                # 手动将聊天记录格式化为单个字符串提示。
                # `generate` 函数需要一个字符串, 而不是字典列表。
                prompt_parts = []
                for message in self.history.get_formatted_history():
                    prompt_parts.append(f"{message['role']}: {message['content']}")
                prompt_parts.append("assistant:") # 提示AI开始回答
                prompt_string = "\n".join(prompt_parts)
                # --- FIX END ---

                self._generate_stream_response(prompt_string)

            except KeyboardInterrupt:
                print("\n\n程序被中断。再见！")
                break
            except Exception as e:
                print(f"\n[ERROR] 发生未知错误: {e}", file=sys.stderr)

    def _generate_stream_response(self, prompt_or_history):
        """
        生成并流式打印模型的响应。
        """
        print("\n🤖 AI: ", end="", flush=True)
        full_response = ""
        try:
            # 使用 generate 方法进行流式生成, 这是正确的函数
            for chunk in self.pipe.generate(prompt_or_history, self.generation_config):
                print(chunk, end="", flush=True)
                full_response += chunk
            print() # 换行
            # 仅在成功生成响应后才将其添加到历史记录中
            self.history.add_message("assistant", full_response)
        except Exception as e:
            error_message = f"\n[ERROR] 生成响应时出错: {e}"
            print(error_message, file=sys.stderr)
            # 不将错误信息添加到历史记录中, 以免污染上下文
            # self.history.add_message("assistant", error_message)

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
