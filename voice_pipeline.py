#!/usr/bin/env python3
"""端到端语音对话管线骨架。

Mic ── PyAudio/sounddevice 16kHz录音
  └─→ SenseVoice (CPU INT8) → 中文 + 情感标签
      └─→ llama.cpp (Gemma 4 E4B) HTTP server → 回复
          └─→ espeak-ng (TTS 占位) → Speaker

后续：把 espeak 换成 MeloTTS NPU。

用法:
    # 准备：先启动 llama-server
    # llama-server -m /home/intel/models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 --port 8080
    python voice_pipeline.py            # 交互模式
    python voice_pipeline.py --file in.wav  # 测试模式
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import openvino as ov
import requests
import sentencepiece as spm
import sounddevice as sd
import soundfile as sf
import kaldi_native_fbank as knf

from _paths import SENSEVOICE_ROOT as ROOT  # B18: 不再硬编码 /home/intel
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
SPEC_TOK = re.compile(r"<\|[^|>]+\|>")  # SenseVoice 特殊 token


def extract_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """16kHz mono float32 → log-mel 80 → LFR 7+6 (T', 560)."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sr
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(sr, (audio * 32768).astype(np.float32))
    fb.input_finished()
    feats = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)], axis=0)
    # LFR 7+6
    T, D = feats.shape
    LFR_T = (T + 5) // 6
    out = np.zeros((LFR_T, D * 7), dtype=np.float32)
    for i in range(LFR_T):
        chunk = feats[i * 6 : i * 6 + 7]
        if chunk.shape[0] < 7:
            chunk = np.concatenate(
                [chunk, np.tile(feats[-1:], (7 - chunk.shape[0], 1))], axis=0
            )
        out[i] = chunk.reshape(-1)
    return out  # (T', 560)


def parse_mvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    means: list[float] = []
    vars_: list[float] = []
    state = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("<AddShift>"):
            state = "mean"; continue
        if line.startswith("<Rescale>"):
            state = "var"; continue
        if line.startswith("<LearnRateCoef>") and state in ("mean", "var"):
            m = re.search(r"\[(.*)\]", line)
            if m:
                vals = [float(x) for x in m.group(1).split() if x]
                if state == "mean": means = vals
                else: vars_ = vals
                state = None
    return np.asarray(means, dtype=np.float32), np.asarray(vars_, dtype=np.float32)


class SenseVoiceASR:
    def __init__(self, root: Path = ROOT, device: str = "CPU") -> None:
        print(f"[ASR] loading on {device} ...")
        t0 = time.perf_counter()
        core = ov.Core()
        model = core.read_model(str(root / "sense-voice-encoder-int8.onnx"))
        self.compiled = core.compile_model(model, device)
        self.tok = spm.SentencePieceProcessor()
        self.tok.load(str(root / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
        self.mean, self.var = parse_mvn(root / "am.mvn")
        print(f"[ASR] ready in {time.perf_counter()-t0:.1f}s")

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        feats = extract_features(audio, sr)  # (T, 560)
        if self.mean.size == feats.shape[1]:
            feats = (feats + self.mean) * self.var
        feats = feats[None, ...]
        feats_len = np.array([feats.shape[1]], dtype=np.int32)
        t0 = time.perf_counter()
        out = self.compiled({"speech": feats, "speech_lengths": feats_len})
        infer_ms = (time.perf_counter() - t0) * 1000
        logits = list(out.values())[0][0]
        pred = np.argmax(logits, axis=-1)
        decoded: list[int] = []
        prev = -1
        for tok in pred.tolist():
            if tok != prev and tok != 0:
                decoded.append(tok)
            prev = tok
        text = self.tok.decode(decoded)
        # 拆出特殊 token (lang/emotion/event)
        special = SPEC_TOK.findall(text)
        clean = SPEC_TOK.sub("", text).strip()
        audio_dur = len(audio) / sr
        rtfx = audio_dur * 1000 / infer_ms if infer_ms > 0 else 0
        return {
            "text": clean,
            "tags": special,
            "infer_ms": infer_ms,
            "audio_s": audio_dur,
            "rtfx": rtfx,
        }


class LLMError(Exception):
    """LLM 调用失败标记 (B20: 让上层能跳过 TTS)。"""


def call_llm(prompt: str, system: str | None = None, max_tokens: int = 500) -> str:
    """调 llama-server。

    B20 修复: 失败抛 LLMError 而非返回 "[ERROR]" 字符串污染对话。
    Qwen3.6 修复: enable_thinking=False 避免 reasoning 吃光 max_tokens。
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(LLM_URL, json=payload, timeout=180)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if content:
        return content
    raise LLMError("LLM 返回空 content (检查 enable_thinking / max_tokens / server 状态)")


def speak(text: str) -> None:
    """B21 修复: espeak-ng 缺失时优雅 fallback 而非 FileNotFoundError。"""
    if not shutil.which("espeak-ng"):
        print(f"[TTS skip - espeak-ng 未安装] {text}")
        return
    subprocess.run(["espeak-ng", "-v", "zh", "-s", "150", text], check=False)


def record_until_silence(seconds: float = 5.0, sr: int = 16000) -> np.ndarray:
    """简化版：固定时长录音。后续可加 VAD。"""
    print(f"[REC] 录音 {seconds}s ...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="测试用 wav，省略则录音")
    parser.add_argument("--asr-device", default="CPU")
    parser.add_argument("--rec-seconds", type=float, default=5.0)
    parser.add_argument("--no-tts", action="store_true", help="只跑 ASR+LLM 不发声")
    parser.add_argument("--system", default="你是一个简洁的中文助手，用一两句话回答。",
                        help="LLM system prompt")
    args = parser.parse_args()

    asr = SenseVoiceASR(device=args.asr_device)

    if args.file:
        audio, sr = sf.read(args.file, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio = record_until_silence(args.rec_seconds)

    # 1. ASR
    res = asr.transcribe(audio)
    print(f"\n👤 [{', '.join(res['tags'])}] {res['text']}")
    print(f"   (ASR {res['infer_ms']:.0f}ms / {res['audio_s']:.1f}s 音频, RTFx {res['rtfx']:.1f})")

    if not res["text"]:
        print("[INFO] 没识别到内容，退出")
        return

    # 2. LLM (B20: 失败跳过 TTS，不让错误字符串被合成)
    t0 = time.perf_counter()
    try:
        reply = call_llm(res["text"], system=args.system)
    except LLMError as e:
        print(f"\n⚠ LLM 失败: {e}")
        return
    except Exception as e:
        print(f"\n⚠ LLM 异常: {e}")
        return
    llm_ms = (time.perf_counter() - t0) * 1000
    print(f"\n🤖 {reply}")
    print(f"   (LLM {llm_ms:.0f}ms)")

    # 3. TTS
    if not args.no_tts:
        t0 = time.perf_counter()
        speak(reply)
        print(f"   (TTS {(time.perf_counter()-t0)*1000:.0f}ms)")


if __name__ == "__main__":
    main()
