"""NPU smoke test: emotions-recognition-retail-0003
最小 NPU CNN 通路验证 (0.13 GFLOPs, 64x64 input)
"""
import sys, time
import numpy as np
import openvino as ov

MODEL_DIR = "/home/intel/QwenTalk/models/omz"
XML = f"{MODEL_DIR}/emotions-recognition-retail-0003.xml"
LABELS = ["neutral", "happy", "sad", "surprise", "anger"]

print(f"[1/4] OpenVINO version: {ov.__version__}", flush=True)
core = ov.Core()
print(f"      devices: {core.available_devices}", flush=True)
print(f"      NPU name: {core.get_property('NPU', 'FULL_DEVICE_NAME') if 'NPU' in core.available_devices else 'N/A'}", flush=True)

print(f"[2/4] read model + reshape to [1,3,64,64]", flush=True)
model = core.read_model(XML)
print(f"      input: {model.input(0).get_partial_shape()}, output: {model.output(0).get_partial_shape()}", flush=True)
model.reshape([1, 3, 64, 64])

results = {}
for device in ["NPU", "GPU", "CPU"]:
    if device not in core.available_devices:
        continue
    print(f"\n[3/4] compile @ {device}...", flush=True)
    t0 = time.time()
    try:
        compiled = core.compile_model(model, device)
        compile_t = time.time() - t0
        print(f"      compile ok in {compile_t:.2f}s", flush=True)
    except Exception as e:
        print(f"      ✗ compile FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        continue

    # Warm + bench (200 iters of random face crop)
    rng = np.random.default_rng(42)
    inp = rng.standard_normal((1, 3, 64, 64), dtype=np.float32) * 50 + 128
    inp = inp.astype(np.float32)

    print(f"      warmup 5 iters...", flush=True)
    for _ in range(5):
        compiled([inp])

    print(f"      bench 200 iters...", flush=True)
    t0 = time.time()
    for _ in range(200):
        out = compiled([inp])[compiled.output(0)]
    elapsed = time.time() - t0
    fps = 200 / elapsed
    label = LABELS[int(np.array(out).reshape(-1).argmax())]
    print(f"      → {fps:.1f} FPS ({elapsed*1000/200:.2f} ms/inf), label={label}", flush=True)
    results[device] = fps

print(f"\n[4/4] Summary:", flush=True)
for d, fps in results.items():
    print(f"  {d:5s} {fps:6.1f} FPS", flush=True)
