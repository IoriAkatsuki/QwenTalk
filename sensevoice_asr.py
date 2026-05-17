"""SenseVoice CPU INT8 ASR：log-mel + LFR 特征 + OpenVINO 推理。"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import openvino as ov
import sentencepiece as spm
import kaldi_native_fbank as knf

from _paths import SENSEVOICE_ROOT as ROOT

SPEC_TOK = re.compile(r"<\|[^|>]+\|>")


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
    return out


def parse_mvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """解析 SenseVoice am.mvn 均值/方差文件。"""
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
    """SenseVoice INT8 OpenVINO 包装：transcribe(audio)→dict(text/tags/timing)."""

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
        feats = extract_features(audio, sr)
        if self.mean.size == feats.shape[1]:
            feats = (feats + self.mean) * self.var
        feats = feats[None, ...]
        feats_len = np.array([feats.shape[1]], dtype=np.int32)
        t0 = time.perf_counter()
        out = self.compiled({"speech": feats, "speech_lengths": feats_len})
        infer_ms = (time.perf_counter() - t0) * 1000
        logits = list(out.values())[0][0]
        text = self._ctc_decode(logits)
        special = SPEC_TOK.findall(text)
        clean = SPEC_TOK.sub("", text).strip()
        audio_dur = len(audio) / sr
        rtfx = audio_dur * 1000 / infer_ms if infer_ms > 0 else 0
        return {
            "text": clean, "tags": special,
            "infer_ms": infer_ms, "audio_s": audio_dur, "rtfx": rtfx,
        }

    def _ctc_decode(self, logits: np.ndarray) -> str:
        """CTC greedy decode：去重 + 去 blank。"""
        pred = np.argmax(logits, axis=-1)
        decoded: list[int] = []
        prev = -1
        for tok in pred.tolist():
            if tok != prev and tok != 0:
                decoded.append(tok)
            prev = tok
        return self.tok.decode(decoded)
