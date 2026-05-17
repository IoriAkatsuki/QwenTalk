"""NPU 三模型并发 smoke test
模拟真实 demo 场景：YOLOv8n-face + facial-landmarks-98 + emotions 同时跑 NPU
验证 OpenVINO NPU plugin 多 compiled_model 共存稳定性
"""
import sys, time
import numpy as np
import openvino as ov
from concurrent.futures import ThreadPoolExecutor

MODELS = {
    "yolo_face":   "/home/intel/QwenTalk/models/omz/yolov8n-face_static.xml",
    "landmarks98": "/home/intel/QwenTalk/models/omz/facial-landmarks-98-detection-0001.xml",
    "emotions":    "/home/intel/QwenTalk/models/omz/emotions-recognition-retail-0003.xml",
}

INPUT_SHAPES = {
    "yolo_face":   [1, 3, 640, 640],
    "landmarks98": [1, 3, 64, 64],   # 实测 model.input shape
    "emotions":    [1, 3, 64, 64],
}

DEVICE = "NPU"
core = ov.Core()
print(f"OpenVINO {ov.__version__}", flush=True)
print(f"devices: {core.available_devices}", flush=True)

# 1. 顺序编译三模型
compiled = {}
for name, xml in MODELS.items():
    print(f"\n=== compile {name} @ {DEVICE} ===", flush=True)
    model = core.read_model(xml)
    print(f"   orig input: {model.input(0).get_partial_shape()}", flush=True)
    try:
        model.reshape(INPUT_SHAPES[name])
    except Exception as e:
        print(f"   reshape skipped: {e}", flush=True)
    print(f"   reshaped to: {model.input(0).get_partial_shape()}", flush=True)
    t0 = time.time()
    compiled[name] = core.compile_model(model, DEVICE)
    print(f"   compile {time.time()-t0:.2f}s OK", flush=True)

# 2. 准备输入
rng = np.random.default_rng(42)
inputs = {n: rng.uniform(0, 1, s).astype(np.float32) for n, s in INPUT_SHAPES.items()}

# 3. 串行 baseline (each warmup + bench)
print(f"\n=== serial baseline ===", flush=True)
serial_fps = {}
for name, model in compiled.items():
    inp = inputs[name]
    for _ in range(3): model([inp])  # warmup
    n = 100 if name != "yolo_face" else 50
    t0 = time.time()
    for _ in range(n):
        model([inp])
    elapsed = time.time() - t0
    fps = n / elapsed
    serial_fps[name] = fps
    print(f"   {name:12s}  {fps:6.1f} FPS  ({elapsed*1000/n:.2f} ms)", flush=True)

# 4. 并发 stress (simulate real pipeline: yolo 30fps + landmark 30fps + emo 10fps)
print(f"\n=== concurrent stress (30s) ===", flush=True)
ratios = {"yolo_face": 30, "landmarks98": 30, "emotions": 10}
counters = {n: 0 for n in MODELS}
errors = {n: 0 for n in MODELS}
stop_at = time.time() + 30

def worker(name):
    model = compiled[name]
    inp = inputs[name]
    interval = 1.0 / ratios[name]
    next_t = time.time()
    while time.time() < stop_at:
        next_t += interval
        try:
            model([inp])
            counters[name] += 1
        except Exception as e:
            errors[name] += 1
            if errors[name] < 3:
                print(f"   ERR {name}: {e}", flush=True)
        sleep = next_t - time.time()
        if sleep > 0: time.sleep(sleep)

with ThreadPoolExecutor(max_workers=3) as ex:
    futures = [ex.submit(worker, n) for n in MODELS]
    for f in futures: f.result()

duration = 30
print(f"\n=== concurrent results ({duration}s) ===", flush=True)
for name in MODELS:
    fps = counters[name] / duration
    target = ratios[name]
    pct = fps / target * 100
    print(f"   {name:12s}  {fps:5.1f} FPS / {target} target ({pct:.0f}%)  errors={errors[name]}", flush=True)
print(flush=True)
