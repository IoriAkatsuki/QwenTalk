#!/usr/bin/env python3
"""SenseVoice-Small ONNX → OpenVINO 推理测试。

参考 SenseVoice ONNX inference 流程：
1. wav → log-mel 80-dim features (kaldi-native-fbank)
2. mean/var normalize via am.mvn
3. encoder forward (features, language_id)
4. CTC greedy decode → BPE token IDs → text
5. parse special tokens: <|HAPPY|>/<|SAD|>/<|Speech|>/<|BGM|>/...

用法:
    python sensevoice_test.py --device CPU
    python sensevoice_test.py --device GPU
    python sensevoice_test.py --device NPU --model sense-voice-encoder-int8.onnx
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import openvino as ov

try:
    import sentencepiece as spm
except ImportError:
    spm = None


# 语言 token id (根据 SenseVoice 配置)
LANG_ID = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXT_NORM_ID = {"withitn": 14, "woitn": 15}


def load_mvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """解析 am.mvn 文件得到 mean/var (用于 80-dim Mel 特征归一化)。"""
    means: list[float] = []
    vars_: list[float] = []
    state = None
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("<AddShift>"):
                state = "mean"
                continue
            if line.startswith("<Rescale>"):
                state = "var"
                continue
            if line.startswith("<LearnRateCoef>") and state in ("mean", "var"):
                # 这一行末尾就是 [vector ...]
                m = re.search(r"\[(.*)\]", line)
                if m:
                    vals = [float(x) for x in m.group(1).split() if x]
                    if state == "mean":
                        means = vals
                    else:
                        vars_ = vals
                    state = None
    return np.asarray(means, dtype=np.float32), np.asarray(vars_, dtype=np.float32)


def extract_fbank(wav_path: Path) -> np.ndarray:
    """提取 80-dim log-mel filterbank (16kHz, 25ms window, 10ms hop)。"""
    import kaldi_native_fbank as knf
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # SenseVoice 用 fbank 80-dim
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    # KFB 期望 int16
    fbank.accept_waveform(16000, (audio * 32768).astype(np.float32))
    fbank.input_finished()
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], axis=0)
    return feats.astype(np.float32)  # (T, 80)


def lfr(feats: np.ndarray, m: int = 7, n: int = 6) -> np.ndarray:
    """Low Frame Rate: 每 n 帧取 m 帧拼接 (SenseVoice 用 7+6)。"""
    T, D = feats.shape
    LFR_T = (T + n - 1) // n
    out = np.zeros((LFR_T, D * m), dtype=np.float32)
    for i in range(LFR_T):
        start = i * n
        end = min(start + m, T)
        chunk = feats[start:end]
        if chunk.shape[0] < m:
            pad = np.tile(feats[-1:], (m - chunk.shape[0], 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        out[i] = chunk.reshape(-1)
    return out


def build_tokenizer(spm_model: Path) -> "spm.SentencePieceProcessor":
    if spm is None:
        raise RuntimeError("pip install sentencepiece")
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model))
    return sp


def main() -> None:
    parser = argparse.ArgumentParser()
    import os
    parser.add_argument("--root",
                        default=os.environ.get("SENSEVOICE_ROOT",
                                                str(Path.home() / "asr_tts" / "SenseVoice-onnx")),
                        help="SenseVoice ONNX 根目录 (env: SENSEVOICE_ROOT)")
    parser.add_argument("--model", default="sense-voice-encoder-int8.onnx",
                        help="ONNX file inside root")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU", "AUTO"])
    parser.add_argument("--wav", default=None, help="自定义 wav 路径，不传则用自带 zh sample")
    parser.add_argument("--lang", default="auto", choices=list(LANG_ID))
    parser.add_argument("--max-frames", type=int, default=300,
                        help="NPU 静态化时序长度（LFR 后帧数，300≈30s 音频）")
    args = parser.parse_args()

    root = Path(args.root)
    onnx_path = root / args.model
    wav_path = Path(args.wav) if args.wav else root / "asr_example_zh.wav"
    print(f"[INFO] model={onnx_path.name} device={args.device} wav={wav_path.name}")

    # 1) 特征提取
    t0 = time.perf_counter()
    feats = extract_fbank(wav_path)  # (T, 80)
    print(f"[FBANK] feats {feats.shape} in {(time.perf_counter()-t0)*1000:.1f} ms")

    mean, var = load_mvn(root / "am.mvn")
    if mean.size == 80 and var.size == 80:
        feats = (feats + mean) * var  # SenseVoice 风格：先 +mean 再 *scale
    elif mean.size > 80:
        # 已经是 LFR 后的 mvn (560 = 80*7)
        feats_lfr = lfr(feats, 7, 6)
        feats = (feats_lfr + mean) * var
    else:
        feats = lfr(feats, 7, 6)

    # 确保是 LFR 后的形状 (T', 560)
    if feats.shape[1] == 80:
        feats = lfr(feats, 7, 6)
    feats = feats[None, ...]  # (1, T, 560)
    feats_len = np.array([feats.shape[1]], dtype=np.int32)
    print(f"[LFR ] feats after LFR {feats.shape}")

    # 2) 加载模型到 OpenVINO
    core = ov.Core()
    print(f"[OV  ] devices={core.available_devices}")
    t0 = time.perf_counter()
    model = core.read_model(str(onnx_path))

    # NPU 要求静态 shape：把 speech 时间维度固定到 max_frames
    if args.device == "NPU":
        T = args.max_frames
        model.reshape({"speech": [1, T, 560], "speech_lengths": [1]})
        print(f"[OV  ] NPU 静态化 shape: speech=(1,{T},560)")
        # 把 feats 右侧 padding 到 T
        if feats.shape[1] > T:
            print(f"[WARN] 音频太长 ({feats.shape[1]} > {T} 帧)，截断")
            feats = feats[:, :T]
            feats_len = np.array([T], dtype=np.int32)
        elif feats.shape[1] < T:
            pad = np.zeros((1, T - feats.shape[1], 560), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=1)

    compiled = core.compile_model(model, args.device)
    print(f"[OV  ] compile on {args.device} in {time.perf_counter()-t0:.2f} s")

    # 3) 准备语言 + text_norm token
    lang = np.array([[LANG_ID[args.lang]]], dtype=np.int32)
    text_norm = np.array([[TEXT_NORM_ID["woitn"]]], dtype=np.int32)

    # 4) 推理
    inputs = {n.any_name: None for n in compiled.inputs}
    print(f"[OV  ] input names: {list(inputs.keys())}")
    print(f"[OV  ] input shapes: {[(n.any_name, list(n.partial_shape)) for n in compiled.inputs]}")

    # 实际填充：SenseVoice ONNX 通常输入是 (speech, speech_lengths, language, textnorm)
    for n in compiled.inputs:
        name = n.any_name
        if name in ("speech", "feats", "feature", "input"):
            inputs[name] = feats
        elif "len" in name.lower():
            inputs[name] = feats_len
        elif "lang" in name.lower():
            inputs[name] = lang
        elif "norm" in name.lower() or "itn" in name.lower():
            inputs[name] = text_norm

    t0 = time.perf_counter()
    outputs = compiled(inputs)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"[OV  ] inference {infer_ms:.1f} ms")
    audio_dur = feats.shape[1] * 6 / 100  # LFR=6, 10ms/frame
    print(f"[STAT] audio_dur~{audio_dur:.1f}s, RTFx={audio_dur*1000/infer_ms:.1f}")

    # 5) CTC greedy decode → tokens → text
    logits = list(outputs.values())[0]  # (1, T, V)
    print(f"[OUT ] logits shape {logits.shape}")
    pred = np.argmax(logits[0], axis=-1)
    # CTC blank=0, 去重去 blank
    decoded: list[int] = []
    prev = -1
    for tok in pred.tolist():
        if tok != prev and tok != 0:
            decoded.append(tok)
        prev = tok
    tok = build_tokenizer(root / "chn_jpn_yue_eng_ko_spectok.bpe.model")
    text = tok.decode(decoded)
    print(f"\n[ASR] {text}")


if __name__ == "__main__":
    main()
