"""NPU 多路虚拟视频流 stress：看 NPU 单 yolov8n-face 模型最多能撑几路 30fps
单 model handle 也支持多请求并发吗？还是必须每路单独 compile_model？

测两种模式：
  A. 单 compiled_model + N 路 worker 并发 infer (无锁串行)
  B. N 个独立 compiled_model + N 路 worker 并发
"""
import time, threading
import numpy as np
import openvino as ov
from concurrent.futures import ThreadPoolExecutor

XML = "/home/intel/QwenTalk/models/omz/yolov8n-face_static.xml"
DURATION = 20

core = ov.Core()
print(f"OV {ov.__version__}", flush=True)

# Mode A: 单 compiled, N workers
def test_mode(num_streams, share_compiled):
    m = core.read_model(XML)
    if share_compiled:
        compiled = core.compile_model(m, "NPU")
        compileds = [compiled] * num_streams
    else:
        compileds = [core.compile_model(m, "NPU") for _ in range(num_streams)]
    inps = [np.random.uniform(0, 1, (1,3,640,640)).astype(np.float32) for _ in range(num_streams)]
    # warmup
    for c, i in zip(compileds, inps):
        for _ in range(2): c([i])

    counters = [0] * num_streams
    errors = [0] * num_streams
    target_fps = 30
    interval = 1.0 / target_fps
    stop_at = time.time() + DURATION

    def worker(idx):
        next_t = time.time()
        while time.time() < stop_at:
            next_t += interval
            try:
                compileds[idx]([inps[idx]])
                counters[idx] += 1
            except Exception:
                errors[idx] += 1
            sleep = next_t - time.time()
            if sleep > 0: time.sleep(sleep)

    with ThreadPoolExecutor(max_workers=num_streams) as ex:
        fs = [ex.submit(worker, i) for i in range(num_streams)]
        for f in fs: f.result()

    total = sum(counters) / DURATION
    perfect = num_streams * target_fps
    drop = (1 - total/perfect) * 100
    return total, perfect, drop, sum(errors)

# Test multi-stream scenarios
print("\n=== Mode A: 单 compiled_model + N workers ===", flush=True)
for n in [1, 2, 4]:
    actual, perfect, drop, err = test_mode(n, share_compiled=True)
    print(f"  {n} streams (target {perfect} FPS): actual {actual:.1f} FPS, drop {drop:.1f}%, errors {err}", flush=True)

print("\n=== Mode B: N 个独立 compiled_model + N workers ===", flush=True)
for n in [1, 2, 4, 6]:
    try:
        actual, perfect, drop, err = test_mode(n, share_compiled=False)
        print(f"  {n} streams (target {perfect} FPS): actual {actual:.1f} FPS, drop {drop:.1f}%, errors {err}", flush=True)
    except Exception as e:
        print(f"  {n} streams: FAIL - {e}", flush=True)
        break
