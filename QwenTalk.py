#!/usr/bin/env python3
"""
QwenTalk Core Chat Engine
"""
import openvino_genai as ov_genai
import time
from typing import List, Dict, Iterator, Tuple
from transformers import AutoTokenizer

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
        
        # Session management
        self.sessions: Dict[str, List[Dict[str, str]]] = {"Default": []}
        self.active_session = "Default"
        self.max_history_len = 20  # Keep last 20 pairs of user/assistant messages
        self.creative_mode = True  # Default to creative

    def set_creative_mode(self, is_creative: bool):
        """Adjusts generation parameters based on creative mode."""
        self.creative_mode = is_creative
        if is_creative:
            self.generation_config.do_sample = True
            self.generation_config.temperature = 0.7
        else:
            self.generation_config.do_sample = False
            # For deterministic output, temperature is often set to 0, but we'll use a very low value
            self.generation_config.temperature = 0.01 

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
        self.generation_config.temperature = 0.7
        self.generation_config.top_p = 0.95
        self.generation_config.top_k = 40
        self.generation_config.repetition_penalty = 1.1
        self.generation_config.do_sample = True

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
            # Switch to Default or the first available session
            if "Default" in self.sessions:
                self.active_session = "Default"
            else:
                self.active_session = next(iter(self.sessions))
        
        return True, f"Session '{name}' deleted."

    def get_history(self, session_name: str = None) -> List[Dict[str, str]]:
        """Returns the history of the specified or active session."""
        name = session_name or self.active_session
        return self.sessions.get(name, [])

    def add_to_history(self, role: str, content: str):
        """Adds a message to the active session's history."""
        history = self.sessions[self.active_session]
        history.append({"role": role, "content": content})
        # Trim history if it exceeds max length
        if len(history) > self.max_history_len * 2:
            self.sessions[self.active_session] = history[-self.max_history_len * 2:]

    def clear_history(self):
        """Clears the active session's chat history."""
        if self.active_session in self.sessions:
            self.sessions[self.active_session] = []

    # --- Generation Methods ---
    def generate_stream(self, user_input: str) -> Iterator[str]:
        """
        Generates a response for the active session token by token.
        """
        if not self.pipe:
            raise RuntimeError("Model is not loaded.")

        self.add_to_history("user", user_input)
        
        active_history = self.get_history()
        prompt = self.tokenizer.apply_chat_template(
            active_history,
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
            if self.sessions[self.active_session] and self.sessions[self.active_session][-1]['role'] == 'user':
                self.sessions[self.active_session].pop()

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""
        if not self.tokenizer:
            return 0
        return len(self.tokenizer.encode(text))