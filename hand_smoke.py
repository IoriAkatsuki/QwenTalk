"""WL3 MediaPipe Hands NPU smoke
palm: [1,192,192,3] NHWC → 21 anchors
handpose: [1,224,224,3] NHWC → 21 keypoints (x,y,z) flattened to 63
"""
import sys, time
import numpy as np
import openvino as ov

PALM_ONNX = "/home/intel/QwenTalk/models/omz/palm_detection_mediapipe_2023feb.onnx"
POSE_ONNX = "/home/intel/QwenTalk/models/omz/handpose_estimation_mediapipe_2023feb.onnx"

core = ov.Core()
print(f"OV {ov.__version__} devices={core.available_devices}", flush=True)

for label, onnx, shape in [
    ("palm",     PALM_ONNX, (1, 192, 192, 3)),
    ("handpose", POSE_ONNX, (1, 224, 224, 3)),
]:
    print(f"\n=== {label} ===", flush=True)
    m = core.read_model(onnx)
    # Already static, no reshape needed

    inp = np.random.uniform(0, 1, shape).astype(np.float32)

    for device in ["NPU", "GPU", "CPU"]:
        if device not in core.available_devices: continue
        try:
            t0 = time.time()
            compiled = core.compile_model(m, device)
            comp_t = time.time() - t0
        except Exception as e:
            print(f"  {device:5s}: compile FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue

        for _ in range(3):
            compiled([inp])

        n = 100 if device != "CPU" else 50
        t0 = time.time()
        for _ in range(n):
            compiled([inp])
        elapsed = time.time() - t0
        fps = n / elapsed
        ms = elapsed*1000/n
        print(f"  {device:5s}: compile {comp_t:5.2f}s | {ms:6.2f} ms/inf | {fps:6.1f} FPS", flush=True)
