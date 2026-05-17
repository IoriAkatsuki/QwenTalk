"""4-模 NPU 并发 stress (30s):
yolo_face + landmark98 + emotion + palm + handpose 同时跑
模拟真实"人脸感知 + 手势交互"双链路场景"""
import time
import numpy as np
import openvino as ov
from concurrent.futures import ThreadPoolExecutor

MODELS = {
    "yolo_face":   ("/home/intel/QwenTalk/models/omz/yolov8n-face_static.xml", [1,3,640,640]),
    "landmarks98": ("/home/intel/QwenTalk/models/omz/facial-landmarks-98-detection-0001.xml", [1,3,64,64]),
    "emotion":     ("/home/intel/QwenTalk/models/omz/emotions-recognition-retail-0003.xml", [1,3,64,64]),
    "palm":        ("/home/intel/QwenTalk/models/omz/palm_detection_mediapipe_2023feb.onnx", [1,192,192,3]),
    "handpose":    ("/home/intel/QwenTalk/models/omz/handpose_estimation_mediapipe_2023feb.onnx", [1,224,224,3]),
}

# 模拟真实场景：每秒采样数
TARGETS = {"yolo_face": 30, "landmarks98": 30, "emotion": 10, "palm": 30, "handpose": 60}

core = ov.Core()
print(f"OV {ov.__version__}", flush=True)

# Compile all
compiled = {}
inputs = {}
print("=== compile NPU ===", flush=True)
for name, (xml, shape) in MODELS.items():
    m = core.read_model(xml)
    cur = m.input(0).get_partial_shape()
    if list(cur) != shape:
        try: m.reshape(shape)
        except Exception as e: print(f"  {name} reshape FAIL: {e}", flush=True)
    t0 = time.time()
    compiled[name] = core.compile_model(m, "NPU")
    print(f"  {name:12s} compile {time.time()-t0:.2f}s", flush=True)
    inputs[name] = np.random.uniform(0, 1, shape).astype(np.float32)

# Warmup
for n, m in compiled.items():
    for _ in range(2): m([inputs[n]])

# Concurrent stress
DURATION = 30
counters = {n: 0 for n in MODELS}
errors = {n: 0 for n in MODELS}
stop_at = time.time() + DURATION

def worker(name):
    model = compiled[name]
    inp = inputs[name]
    interval = 1.0 / TARGETS[name]
    next_t = time.time()
    while time.time() < stop_at:
        next_t += interval
        try:
            model([inp])
            counters[name] += 1
        except Exception as e:
            errors[name] += 1
            if errors[name] < 3: print(f"  ERR {name}: {e}", flush=True)
        sleep = next_t - time.time()
        if sleep > 0: time.sleep(sleep)

print(f"\n=== concurrent stress ({DURATION}s, 5 workers) ===", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=5) as ex:
    fs = [ex.submit(worker, n) for n in MODELS]
    for f in fs: f.result()
elapsed = time.time() - t0

print(f"\n=== RESULTS ({elapsed:.1f}s elapsed) ===", flush=True)
total_compute_ms = 0
for name in MODELS:
    fps = counters[name] / DURATION
    target = TARGETS[name]
    pct = fps / target * 100
    # Estimate NPU compute time
    bench_ms = {"yolo_face": 8.4, "landmarks98": 2.66, "emotion": 0.51, "palm": 2.55, "handpose": 1.38}
    spent = counters[name] * bench_ms.get(name, 0)
    total_compute_ms += spent
    print(f"  {name:12s}  {fps:5.1f}/{target} FPS  ({pct:.0f}%)  errors={errors[name]}  compute≈{spent:.0f}ms", flush=True)

print(f"\nNPU 总计算时间: {total_compute_ms:.0f} ms / {DURATION*1000} ms ({total_compute_ms/(DURATION*10):.1f}%)", flush=True)
print(f"NPU 余量: ~{100 - total_compute_ms/(DURATION*10):.0f}%", flush=True)
