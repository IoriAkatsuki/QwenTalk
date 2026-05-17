"""硬件 JPEG 编码器：通过 ffmpeg subprocess 把 BGR24 NumPy 帧送进
mjpeg_vaapi（Xe Media Engine），把吐出的 JPEG 字节流按 SOI/EOI 切分。

为什么这样实现：
  - Arrow Lake-U + iHD 25.4.6 暴露 JPEGBaseline + VAEntrypointEncPicture，
    但 PyAV / ffmpeg-python 在 Fedora 标准源里没现成包，subprocess 最稳。
  - 写线程把 raw BGR24 喂 stdin，读线程从 stdout 扫 0xFFD8/0xFFD9 切帧。
  - cv2.imencode (libjpeg-turbo CPU) 作为 fallback：环境变量 QWENTALK_ENCODER=cpu。
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
from typing import Iterator

import cv2
import numpy as np

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

DRI_DEVICE = os.environ.get("QWENTALK_VAAPI_DEV", "/dev/dri/renderD128")
ENCODER_BACKEND = os.environ.get("QWENTALK_ENCODER", "vaapi")  # vaapi | cpu
JPEG_QUALITY = int(os.environ.get("QWENTALK_JPEG_Q", "75"))


class JpegEncoder:
    """统一接口：encode(bgr) -> bytes (JPEG)。"""

    def encode(self, bgr: np.ndarray) -> bytes | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def backend(self) -> str:
        return "abstract"


class CpuJpegEncoder(JpegEncoder):
    """libjpeg-turbo SIMD（cv2.imencode）。"""

    def encode(self, bgr: np.ndarray) -> bytes | None:
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return jpg.tobytes() if ok else None

    @property
    def backend(self) -> str:
        return "cpu-libjpeg-turbo"


class VaapiJpegEncoder(JpegEncoder):
    """ffmpeg + mjpeg_vaapi 走 Xe Media Engine 硬件编码。"""

    def __init__(self, width: int, height: int, fps: int = 30, quality: int = JPEG_QUALITY):
        self.width, self.height, self.fps = width, height, fps
        self._frame_size = width * height * 3
        self._proc = self._spawn_ffmpeg(quality)
        self._read_buf = bytearray()
        self._out_queue: queue.Queue[bytes] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, name="vaapi-jpeg-rx", daemon=True)
        self._reader.start()

    def _spawn_ffmpeg(self, quality: int) -> subprocess.Popen:
        env = os.environ.copy()
        env["LIBVA_DRIVER_NAME"] = "iHD"
        log_path = os.environ.get("QWENTALK_FFMPEG_LOG", "/tmp/qwentalk_ffmpeg.log")
        self._stderr_log = open(log_path, "w", buffering=1)
        # 输入声明 nv12（pipeline 已用 cv2 SIMD 把 BGR 转好 + 重排为 NV12 layout）
        # → ffmpeg 内只做 hwupload，省掉 swscale，CPU 从 ~50% 降到 ~5%
        # BGR24 输入 + 多线程 swscale。NV12 输入 hwupload 拿不到 frame ctx
        # (https://trac.ffmpeg.org/wiki/Hardware/VAAPI 已知)，所以让 ffmpeg
        # 自己做 BGR→NV12，但开 -filter_threads 4 把 swscale 拆到 4 核。
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-filter_threads", "4",
            "-init_hw_device", f"vaapi=va:{DRI_DEVICE}",
            "-filter_hw_device", "va",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", "pipe:0",
            "-vf", "format=nv12,hwupload",
            "-c:v", "mjpeg_vaapi",
            "-global_quality", str(quality),
            "-f", "mjpeg", "pipe:1",
        ]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr_log, bufsize=0, env=env,
        )

    def _reader_loop(self) -> None:
        assert self._proc.stdout is not None
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                break
            self._read_buf.extend(chunk)
            for jpg in self._extract_jpegs():
                try:
                    self._out_queue.put_nowait(jpg)
                except queue.Full:
                    try:
                        self._out_queue.get_nowait()
                        self._out_queue.put_nowait(jpg)
                    except (queue.Empty, queue.Full):
                        pass

    def _extract_jpegs(self) -> Iterator[bytes]:
        """从 read_buf 顺序抠出完整 JPEG 帧。"""
        while True:
            soi = self._read_buf.find(JPEG_SOI)
            if soi < 0:
                self._read_buf.clear()
                return
            eoi = self._read_buf.find(JPEG_EOI, soi + 2)
            if eoi < 0:
                if soi > 0:
                    del self._read_buf[:soi]
                return
            end = eoi + 2
            yield bytes(self._read_buf[soi:end])
            del self._read_buf[:end]

    def encode(self, bgr: np.ndarray) -> bytes | None:
        if self._proc.poll() is not None or self._proc.stdin is None:
            return None
        if bgr.shape != (self.height, self.width, 3) or bgr.dtype != np.uint8:
            return None
        try:
            # 直接喂 BGR24 raw bytes，让 ffmpeg 多线程 swscale 转 NV12
            self._proc.stdin.write(bgr.tobytes())
        except (BrokenPipeError, ValueError):
            return None
        try:
            return self._out_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def _i420_to_nv12_bytes(self, i420: np.ndarray) -> bytes:
        """I420 (YYY..UUU..VVV) → NV12 (YYY..UVUVUV..) 平面重排。"""
        h, w = self.height, self.width
        y = i420[:h, :w]
        uv_h, uv_w = h // 2, w // 2
        u = i420[h:h + uv_h, :uv_w].reshape(uv_h, uv_w)
        v = i420[h + uv_h:h + 2 * uv_h, :uv_w].reshape(uv_h, uv_w)
        uv_interleaved = np.empty((uv_h, uv_w * 2), dtype=np.uint8)
        uv_interleaved[:, 0::2] = u
        uv_interleaved[:, 1::2] = v
        return y.tobytes() + uv_interleaved.tobytes()

    def close(self) -> None:
        self._stop.set()
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        if self._proc:
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    @property
    def backend(self) -> str:
        return "vaapi-mjpeg (Xe Media Engine)"


def make_encoder(width: int, height: int, fps: int = 30) -> JpegEncoder:
    """工厂：尝试 VA-API，失败回退 CPU。"""
    if ENCODER_BACKEND == "cpu":
        return CpuJpegEncoder()
    try:
        enc = VaapiJpegEncoder(width, height, fps)
        # 自检：连发 5 帧给 ffmpeg 灌满 pipeline，期待至少拿到 1 帧
        probe = np.zeros((height, width, 3), dtype=np.uint8)
        got = None
        for _ in range(5):
            got = enc.encode(probe)
            if got:
                break
        if not got:
            enc.close()
            log_path = os.environ.get("QWENTALK_FFMPEG_LOG", "/tmp/qwentalk_ffmpeg.log")
            with open(log_path, "r", errors="ignore") as f:
                tail = f.read()[-500:]
            raise RuntimeError(f"VA-API probe failed; ffmpeg stderr:\n{tail}")
        return enc
    except Exception as exc:
        print(f"[encoder] VA-API 不可用，回退 CPU: {exc}")
        return CpuJpegEncoder()
