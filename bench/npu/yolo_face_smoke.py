"""NPU smoke: yolov8n-face (lindevs) - 8.1 GFLOPs, 640x640, NPU 主场试金石"""
import sys, time
import numpy as np
import openvino as ov

ONNX = "/home/intel/QwenTalk/models/omz/yolov8n-face.onnx"
LABELS = ["face"]

print(f"[1/5] OpenVINO {ov.__version__}", flush=True)
core = ov.Core()
print(f"      devices: {core.available_devices}", flush=True)

print(f"[2/5] read ONNX + reshape to [1,3,640,640]", flush=True)
model = core.read_model(ONNX)
print(f"      input: {model.input(0).get_partial_shape()}, output: {model.output(0).get_partial_shape()}", flush=True)
model.reshape([1, 3, 640, 640])

# Save as OV IR for reuse (FP16)
ir_xml = "/home/intel/QwenTalk/models/omz/yolov8n-face_static.xml"
ov.save_model(model, ir_xml)
print(f"      saved IR: {ir_xml}", flush=True)

results = {}
rng = np.random.default_rng(42)
inp = rng.uniform(0, 1, (1, 3, 640, 640)).astype(np.float32)

for device in ["NPU", "GPU", "CPU"]:
    if device not in core.available_devices:
        continue
    print(f"\n[3/5] compile @ {device}...", flush=True)
    t0 = time.time()
    try:
        compiled = core.compile_model(model, device)
        compile_t = time.time() - t0
        print(f"      compile ok in {compile_t:.2f}s", flush=True)
    except Exception as e:
        print(f"      ✗ compile FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        continue

    print(f"      warmup 3 iters...", flush=True)
    for _ in range(3):
        compiled([inp])

    n = 50 if device == "CPU" else 100
    print(f"      bench {n} iters...", flush=True)
    t0 = time.time()
    for _ in range(n):
        out = compiled([inp])[compiled.output(0)]
    elapsed = time.time() - t0
    fps = n / elapsed
    print(f"      → {fps:.1f} FPS ({elapsed*1000/n:.2f} ms/inf), out shape={np.array(out).shape}", flush=True)
    results[device] = fps

print(f"\n[5/5] Summary YOLOv8n-face @ 640x640:", flush=True)
for d, fps in results.items():
    print(f"  {d:5s} {fps:6.1f} FPS  ({1000/fps:.1f} ms/inf)", flush=True)
