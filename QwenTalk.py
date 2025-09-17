#!/usr/bin/env python3
"""
QwenTalk Core Chat Engine
"""
import openvino_genai as ov_genai
import time
from typing import List, Dict, Iterator, Tuple
from transformers import AutoTokenizer

import subprocess

class WebSearchTool:
    """
    提供联网搜索功能的工具。

    通过调用 open-websearch MCP 工具执行搜索。
    """
    def search(self, query: str) -> str:
        """
        执行网络搜索并返回格式化的结果字符串。
        """
        if not query:
            return "错误: 请提供搜索关键词。"
        print(f"\n[INFO] 正在通过 open-websearch 搜索: {query} ...")
        try:
            command = [
                "npx", "-y", "open-websearch@latest",
                "--engine", "duckduckgo",
                "--query", query
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                shell=True,
                encoding='utf-8'
            )
            if not result.stdout.strip():
                return "抱歉，没有找到相关的在线信息。"
            return result.stdout.strip()
        except FileNotFoundError:
            return "错误: `npx` 命令未找到。请确保您已安装 Node.js。"
        except subprocess.CalledProcessError as e:
            return f"搜索时出错: {e.stderr}"
        except Exception as e:
            return f"处理搜索结果时发生未知错误: {e}"

class ChatEngine:
    """
    The core chat engine for handling model loading, generation, and session management.
    """
    def __init__(self, model_path: str, device: str = "NPU"):
        self.model_path = model_path
        self.device = device
        self.pipe = None
        self.tokenizer = None
        self.generation_config = None
        self.search_tool = WebSearchTool()
        
        # Mode flags
        self.creative_mode = True
        self.search_enabled = False
        self.no_think_enabled = False

        # Session management
        self.sessions: Dict[str, List[Dict[str, str]]] = {"Default": []}
        self.active_session = "Default"
        self.max_history_len = 20

    def set_creative_mode(self, is_creative: bool):
        self.creative_mode = is_creative
        if self.generation_config:
            if is_creative:
                self.generation_config.do_sample = True
                self.generation_config.temperature = 0.7
            else:
                self.generation_config.do_sample = False
                self.generation_config.temperature = 0.01

    def set_search_mode(self, enabled: bool):
        self.search_enabled = enabled

    def set_no_think_mode(self, enabled: bool):
        self.no_think_enabled = enabled

    def load_model(self):
        """
        Loads the LLM model and tokenizer.
        """
        print(f"[INFO] Attempting to load model on device: {self.device}...")
        try:
            config = {"PERFORMANCE_HINT": "LATENCY", "MAX_PROMPT_LEN": 4096}
            self.pipe = ov_genai.LLMPipeline(self.model_path, self.device, config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            print(f"[SUCCESS] Model loaded successfully on {self.device}.")
        except Exception as e:
            print(f"[WARNING] Failed to load model on {self.device}: {e}")
            if self.device.upper() != "CPU":
                print("[INFO] Falling back to CPU...")
                try:
                    self.device = "CPU"
                    config = {"PERFORMANCE_HINT": "LATENCY", "MAX_PROMPT_LEN": 4096}
                    self.pipe = ov_genai.LLMPipeline(self.model_path, self.device, config)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                    print("[SUCCESS] Model loaded successfully on CPU.")
                except Exception as e2:
                    print(f"[ERROR] Failed to load model on CPU as well: {e2}")
                    raise e2
        
        self.generation_config = ov_genai.GenerationConfig()
        self.generation_config.max_new_tokens = 4096
        self.generation_config.top_p = 0.95
        self.generation_config.top_k = 40
        self.generation_config.repetition_penalty = 1.1
        self.set_creative_mode(self.creative_mode) # Apply initial mode

    # --- Session Management Methods ---
    def get_active_session_name(self) -> str:
        return self.active_session

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())

    def create_session(self, name: str) -> Tuple[bool, str]:
        if name in self.sessions:
            return False, "Session with this name already exists."
        if not name.strip():
            return False, "Session name cannot be empty."
        self.sessions[name] = []
        self.active_session = name
        return True, f"Session '{name}' created and activated."

    def switch_session(self, name: str) -> bool:
        if name in self.sessions:
            self.active_session = name
            return True
        return False

    def delete_session(self, name: str) -> Tuple[bool, str]:
        if name not in self.sessions:
            return False, "Session not found."
        if len(self.sessions) <= 1:
            return False, "Cannot delete the last remaining session."
        
        del self.sessions[name]
        
        if self.active_session == name:
            if "Default" in self.sessions:
                self.active_session = "Default"
            else:
                self.active_session = next(iter(self.sessions))
        
        return True, f"Session '{name}' deleted."

    def get_history(self, session_name: str = None) -> List[Dict[str, str]]:
        name = session_name or self.active_session
        return self.sessions.get(name, [])

    def add_to_history(self, role: str, content: str):
        history = self.sessions[self.active_session]
        history.append({"role": role, "content": content})
        if len(history) > self.max_history_len * 2:
            self.sessions[self.active_session] = history[-self.max_history_len * 2:]

    def clear_history(self):
        if self.active_session in self.sessions:
            self.sessions[self.active_session] = []

    # --- Generation Methods ---
    def generate_stream(self, user_input: str) -> Iterator[str]:
        if not self.pipe:
            raise RuntimeError("Model is not loaded.")

        self.add_to_history("user", user_input)
        prompt = ""

        if self.search_enabled:
            yield "[INFO] Searching online...\n"
            search_result = self.search_tool.search(user_input)
            yield f"[INFO] Search Results:\n---\n{search_result}\n---\n"
            # For search, we create a specific, one-off prompt
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': f'Based on these search results: "{search_result}", answer the following question: "{user_input}"'} ],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            history = []
            if self.no_think_enabled:
                # "No think" mode uses only the current user input
                history.append({'role': 'user', 'content': user_input})
            else:
                # Normal mode uses the active session history
                history = self.get_history()
            
            prompt = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )

        full_response = ""
        try:
            for token in self.pipe.generate(prompt, self.generation_config):
                full_response += token
                yield token
            
            self.add_to_history("assistant", full_response)

        except Exception as e:
            error_message = f"\n[ERROR] Failed during response generation: {e}"
            print(error_message)
            yield error_message
            # Roll back the user message if generation failed
            if self.sessions[self.active_session] and self.sessions[self.active_session][-1]['role'] == 'user':
                self.sessions[self.active_session].pop()

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return 0
        return len(self.tokenizer.encode(text))