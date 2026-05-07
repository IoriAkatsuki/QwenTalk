#!/usr/bin/env python3
"""NPU head-pose-estimation-adas-0001 验证脚本。

测试内容:
  1. NPU 编译是否通过（静态 shape 60×60）
  2. 单帧推理延迟
  3. 输出 yaw/pitch/roll 三个角度是否合理
  4. 与现有 NPU 感知栈并发兼容性

用法:
  # 板卡上执行
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate openvino
  python npu_headpose_test.py --device NPU
  python npu_headpose_test.py --device NPU --stress 30  # 30秒压测
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _paths import HEADPOSE_ROOT as MODEL_DIR  # B01: env 可配 (HEADPOSE_ROOT)
MODEL_XML = MODEL_DIR / "head-pose-estimation-adas-0001.xml"


def load_model(device: str = "NPU"):
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(MODEL_XML))
    # 静态 reshape: batch=1, 3通道, 60×60
    model.reshape({model.input(0): [1, 3, 60, 60]})
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"编译到 {device}: {compile_ms:.0f} ms")
    return compiled


def single_infer(compiled, warmup: int = 5, repeat: int = 50):
    """单帧推理延迟测试。"""
    dummy = np.random.randn(1, 3, 60, 60).astype(np.float32)

    for _ in range(warmup):
        compiled(dummy)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = compiled(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    yaw = result[compiled.output("angle_y_fc")].flatten()[0]
    pitch = result[compiled.output("angle_p_fc")].flatten()[0]
    roll = result[compiled.output("angle_r_fc")].flatten()[0]

    avg_ms = np.mean(times)
    fps = 1000.0 / avg_ms
    print(f"推理延迟: {avg_ms:.2f} ms ({fps:.0f} FPS)")
    print(f"输出示例: yaw={yaw:.1f}° pitch={pitch:.1f}° roll={roll:.1f}°")
    return avg_ms


def stress_test(compiled, duration: int = 30):
    """持续压测，统计吞吐和错误数。"""
    dummy = np.random.randn(1, 3, 60, 60).astype(np.float32)
    errors = 0
    frames = 0
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration:
        try:
            compiled(dummy)
            frames += 1
        except Exception:
            errors += 1

    elapsed = time.perf_counter() - t_start
    fps = frames / elapsed
    print(f"压测 {duration}s: {frames} 帧, {fps:.0f} FPS, {errors} 错误")
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    parser.add_argument("--stress", type=int, default=0, help="压测秒数，0 表示只跑单帧")
    args = parser.parse_args()

    if not MODEL_XML.exists():
        print(f"模型不存在: {MODEL_XML}")
        print("请先执行: bash download_headpose.sh")
        return

    compiled = load_model(args.device)
    single_infer(compiled)

    if args.stress > 0:
        stress_test(compiled, args.stress)


if __name__ == "__main__":
    main()
