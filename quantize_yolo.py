"""NNCF INT8 量化 yolov8n-face → 测 NPU FPS 提升

策略：合成 face-like 图作 calibration（不追求精度，只量速度）
真实精度需 WIDER-FACE 或 COCO-Person 样本，但 FPS 提升趋势可信
"""
import sys, time
import numpy as np
import openvino as ov
import nncf

XML_FP16 = "/home/intel/QwenTalk/models/omz/yolov8n-face_static.xml"
XML_INT8 = "/home/intel/QwenTalk/models/omz/yolov8n-face_int8.xml"

print(f"OV {ov.__version__}, NNCF {nncf.__version__}", flush=True)

# Calibration: 50 random face-like patches (灰度噪声 + 椭圆肤色块模拟人脸)
def make_calib_image():
    img = np.random.uniform(0.2, 0.5, (1, 3, 640, 640)).astype(np.float32)
    # 模拟人脸肤色椭圆
    cy, cx = np.random.randint(150, 490, 2)
    r = np.random.randint(60, 140)
    yy, xx = np.ogrid[:640, :640]
    mask = ((yy-cy)/r)**2 + ((xx-cx)/r*1.2)**2 < 1
    img[0, 0, mask] = np.random.uniform(0.7, 0.9)  # R high
    img[0, 1, mask] = np.random.uniform(0.5, 0.7)  # G mid
    img[0, 2, mask] = np.random.uniform(0.4, 0.6)  # B low
    return img

print("[1/3] 生成 50 张 calibration 图", flush=True)
calib = [make_calib_image() for _ in range(50)]
calib_dataset = nncf.Dataset(calib, lambda x: x)

print("[2/3] 量化 INT8...", flush=True)
core = ov.Core()
fp16_model = core.read_model(XML_FP16)
t0 = time.time()
int8_model = nncf.quantize(
    fp16_model, calib_dataset,
    target_device=nncf.TargetDevice.NPU,
    preset=nncf.QuantizationPreset.PERFORMANCE,
)
ov.save_model(int8_model, XML_INT8)
print(f"      量化 done in {time.time()-t0:.1f}s", flush=True)

print("[3/3] FP16 vs INT8 bench", flush=True)
results = {}
inp = make_calib_image()

for label, xml in [("FP16", XML_FP16), ("INT8", XML_INT8)]:
    print(f"  --- {label} ---", flush=True)
    model = core.read_model(xml)
    for device in ["NPU", "GPU", "CPU"]:
        try:
            compiled = core.compile_model(model, device)
        except Exception as e:
            print(f"    {device}: compile FAIL {type(e).__name__}", flush=True)
            continue
        for _ in range(3): compiled([inp])  # warmup
        n = 100 if device != "CPU" else 50
        t0 = time.time()
        for _ in range(n):
            compiled([inp])
        elapsed = time.time() - t0
        fps = n / elapsed
        results[(label, device)] = fps
        print(f"    {device:5s} {fps:6.1f} FPS ({elapsed*1000/n:.2f} ms)", flush=True)

print("\n=== Summary (FP16 vs INT8) ===", flush=True)
for device in ["NPU", "GPU", "CPU"]:
    fp = results.get(("FP16", device))
    iq = results.get(("INT8", device))
    if fp and iq:
        speedup = iq / fp
        print(f"  {device:5s} FP16 {fp:6.1f} → INT8 {iq:6.1f} FPS ({speedup:.2f}× speedup)", flush=True)
