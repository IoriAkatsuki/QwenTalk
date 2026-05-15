#!/usr/bin/env python3
"""端到端语音对话管线：ASR→LLM(streaming)→TTS(按句异步)。

Mic ── 16kHz 录音
  └─→ SenseVoice (CPU INT8, sensevoice_asr.py)
      └─→ llama-server (LLM streaming + 自适应 thinking)
          └─→ espeak-ng 按句异步说

用法:
    # 先启动 llama-server: llama-server -m model.gguf --port 8080
    python voice_pipeline.py
    python voice_pipeline.py --file in.wav
    python voice_pipeline.py --thinking off   # 强制 disable thinking
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import requests
import soundfile as sf

from sensevoice_asr import SenseVoiceASR

LLM_BASE = "http://127.0.0.1:8080"
LLM_URL = f"{LLM_BASE}/v1/chat/completions"
MODELS_URL = f"{LLM_BASE}/v1/models"
# 句尾切点：句末标点 OR 已累积 >=8 字后的逗号（拉长 TTS 节奏避免太碎）
SENTENCE_BREAK = re.compile(r"[。？！；?!;\n]+|[，,](?=.{8,})")
# 已知支持 thinking 模式的 model 名片段（disable 后省 reasoning token，拿首字延迟）
THINKING_MODELS = ("qwen3.6", "qwen3", "deepseek-r1", "qwq")


class LLMError(Exception):
    """LLM 调用失败标记。"""


def detect_thinking_support(mode: str) -> bool | None:
    """决定是否传 enable_thinking=False。

    Returns:
        True  → 传 enable_thinking=False（thinking model，省 reasoning token）
        False → 不传 chat_template_kwargs（非 thinking model）
        None  → 强制 on，让 model 自行 thinking
    """
    if mode == "off":
        return True
    if mode == "on":
        return None
    # auto: 探测后端（兼容 OpenAI .data[].id 与 Ollama-style .models[].name|model）
    try:
        r = requests.get(MODELS_URL, timeout=3)
        r.raise_for_status()
        payload = r.json()
        entries = payload.get("data") or payload.get("models") or []
        ids = [(e.get("id") or e.get("model") or e.get("name") or "").lower()
               for e in entries]
        for mid in ids:
            if any(p in mid for p in THINKING_MODELS):
                return True
        return False
    except requests.RequestException:
        return False


def _build_payload(prompt: str, system: str | None, max_tokens: int,
                   disable_thinking: bool | None) -> dict:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "messages": msgs, "max_tokens": max_tokens,
        "temperature": 0.7, "stream": True,
    }
    if disable_thinking is True:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _iter_sse_deltas(payload: dict):
    """SSE 流：yields delta content (string)。

    强制 UTF-8 解码：llama-server 不发 charset，requests 默认 ISO-8859-1 会
    把中文标点（如 。，！）破坏成 latin-1 多字符，导致下游 regex 找不到。
    """
    with requests.post(LLM_URL, json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                return
            try:
                j = json.loads(data)
                delta = j["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


def call_llm_stream(prompt: str, system: str | None = None,
                    max_tokens: int = 500, disable_thinking: bool | None = None):
    """generator: yields (sentence, full_so_far) 按句切割。"""
    payload = _build_payload(prompt, system, max_tokens, disable_thinking)
    buf = ""
    full = ""
    received_any = False
    for delta in _iter_sse_deltas(payload):
        received_any = True
        buf += delta
        full += delta
        while True:
            m = SENTENCE_BREAK.search(buf)
            if not m:
                break
            end = m.end()
            sentence = buf[:end].strip()
            buf = buf[end:]
            if sentence:
                yield sentence, full
    if buf.strip():
        yield buf.strip(), full
    if not received_any:
        raise LLMError("LLM stream 无任何 delta")


def speak(text: str) -> None:
    """espeak-ng 缺失时 fallback 打印。"""
    if not shutil.which("espeak-ng"):
        print(f"[TTS skip - espeak-ng 未安装] {text}")
        return
    subprocess.run(["espeak-ng", "-v", "zh", "-s", "150", text], check=False)


class TTSWorker:
    """后台线程串行说每个句子，避免阻塞 LLM stream。"""

    def __init__(self) -> None:
        self.q: queue.Queue[str | None] = queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def say(self, text: str) -> None:
        self.q.put(text)

    def close(self) -> None:
        self.q.put(None)
        self.t.join(timeout=60)

    def _run(self) -> None:
        while True:
            text = self.q.get()
            if text is None:
                return
            speak(text)


def record_until_silence(seconds: float = 5.0, sr: int = 16000) -> np.ndarray:
    """Lazy import sounddevice：--file 模式不需要录音库。"""
    import sounddevice as sd
    print(f"[REC] 录音 {seconds}s ...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def load_audio(file: str | None, rec_seconds: float) -> np.ndarray:
    if file:
        audio, sr = sf.read(file, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio
    return record_until_silence(rec_seconds)


def run_llm_with_tts(text: str, system: str, max_tokens: int,
                     disable_thinking: bool | None, use_tts: bool) -> None:
    """流式 LLM + 按句 TTS，打印 TTFA 与总耗时。"""
    tts = TTSWorker() if use_tts else None
    first_audio_t: float | None = None
    full_reply = ""
    t0 = time.perf_counter()
    try:
        for sentence, full in call_llm_stream(
                text, system=system, max_tokens=max_tokens,
                disable_thinking=disable_thinking):
            if first_audio_t is None:
                first_audio_t = time.perf_counter() - t0
                print()
            print(f"  → {sentence}")
            full_reply = full
            if tts is not None:
                tts.say(sentence)
    except LLMError as e:
        print(f"\n⚠ LLM 失败: {e}")
    except requests.RequestException as e:
        print(f"\n⚠ LLM 网络异常: {e}")
    finally:
        llm_total = time.perf_counter() - t0
        ttfa = f"{first_audio_t:.2f}s" if first_audio_t is not None else "n/a"
        print(f"\n🤖 {full_reply}")
        print(f"   (TTFA {ttfa}, total LLM {llm_total:.2f}s)")
        if tts is not None:
            tts.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="测试用 wav，省略则录音")
    parser.add_argument("--asr-device", default="CPU")
    parser.add_argument("--rec-seconds", type=float, default=5.0)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                        help="thinking 模式：auto 探测后端 / on 保持 / off 强制 disable")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--system", default="你是一个简洁的中文助手，用一两句话回答。")
    args = parser.parse_args()

    disable_thinking = detect_thinking_support(args.thinking)
    note = {True: "disabled (thinking-capable detected)",
            False: "n/a (non-thinking model)",
            None: "kept on (user override)"}[disable_thinking]
    print(f"[LLM] thinking mode → {note}")

    asr = SenseVoiceASR(device=args.asr_device)
    audio = load_audio(args.file, args.rec_seconds)

    res = asr.transcribe(audio)
    print(f"\n👤 [{', '.join(res['tags'])}] {res['text']}")
    print(f"   (ASR {res['infer_ms']:.0f}ms / {res['audio_s']:.1f}s 音频, RTFx {res['rtfx']:.1f})")

    if not res["text"]:
        print("[INFO] 没识别到内容，退出")
        return

    run_llm_with_tts(res["text"], args.system, args.max_tokens,
                     disable_thinking, use_tts=not args.no_tts)


if __name__ == "__main__":
    main()
