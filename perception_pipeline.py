"""端到端感知 pipeline (NPU 三模串联):
  YOLOv8n-face → 取最大脸框 → crop+resize 64x64 → landmarks98 + emotions

模拟连续 100 帧测吞吐 + 验证 pipeline 链路完整性
"""
import sys, time
import numpy as np
import openvino as ov

MODELS = {
    "face":     "/home/intel/QwenTalk/models/omz/yolov8n-face_static.xml",
    "landmark": "/home/intel/QwenTalk/models/omz/facial-landmarks-98-detection-0001.xml",
    "emotion":  "/home/intel/QwenTalk/models/omz/emotions-recognition-retail-0003.xml",
}

EMO_LABELS = ["neutral", "happy", "sad", "surprise", "anger"]

def post_yolo_face(out, conf=0.30, img_w=640, img_h=640):
    """YOLOv8 face: out shape [1, 5, 8400] = [batch, x_y_w_h_conf, anchors]"""
    o = np.array(out).squeeze(0)  # (5, 8400)
    confs = o[4]
    mask = confs > conf
    if not mask.any():
        return []
    xs, ys, ws, hs = o[0,mask], o[1,mask], o[2,mask], o[3,mask]
    boxes = []
    for x, y, w, h, c in zip(xs, ys, ws, hs, confs[mask]):
        x1 = max(0, int(x - w/2))
        y1 = max(0, int(y - h/2))
        x2 = min(img_w, int(x + w/2))
        y2 = min(img_h, int(y + h/2))
        boxes.append((x1, y1, x2, y2, float(c)))
    return boxes

def crop_resize(img, box, target=64):
    """640x640 输入图 + box → 64x64 crop"""
    x1, y1, x2, y2, _ = box
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img[:, :, y1:y2, x1:x2]  # (1,3,h,w)
    # naïve nearest resize via slicing
    h, w = crop.shape[2], crop.shape[3]
    yi = (np.arange(target) * h / target).astype(int)
    xi = (np.arange(target) * w / target).astype(int)
    return crop[:, :, yi, :][:, :, :, xi]

def main():
    core = ov.Core()
    print(f"OV {ov.__version__} devices={core.available_devices}", flush=True)

    print("=== compile NPU ===", flush=True)
    compiled = {}
    for name, xml in MODELS.items():
        m = core.read_model(xml)
        if name == "face":
            m.reshape([1,3,640,640])
        else:
            m.reshape([1,3,64,64])
        t0 = time.time()
        compiled[name] = core.compile_model(m, "NPU")
        print(f"  {name:10s} compiled {time.time()-t0:.2f}s", flush=True)

    # Synthetic input: random + some "face-like" structure
    rng = np.random.default_rng(0)
    test_img = rng.uniform(0, 1, (1, 3, 640, 640)).astype(np.float32)

    # Warmup
    print("=== warmup ===", flush=True)
    for _ in range(3):
        face_out = compiled["face"]([test_img])[compiled["face"].output(0)]

    # Run pipeline N=100 times
    N = 100
    print(f"=== pipeline run x{N} ===", flush=True)
    timings = {"face": 0, "crop": 0, "landmark": 0, "emotion": 0}
    n_face_detected = 0

    t_total = time.time()
    for i in range(N):
        # Face detection
        t0 = time.time()
        face_out = compiled["face"]([test_img])[compiled["face"].output(0)]
        timings["face"] += time.time() - t0

        boxes = post_yolo_face(face_out, conf=0.05)  # 低 conf 因合成图无真脸
        if not boxes:
            # 用合成 64x64 patch 代替 (验证 pipeline 通路)
            face_crop = test_img[:, :, 100:164, 100:164]
        else:
            n_face_detected += 1
            t0 = time.time()
            biggest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            face_crop = crop_resize(test_img, biggest, target=64)
            timings["crop"] += time.time() - t0
            if face_crop is None:
                face_crop = test_img[:, :, 100:164, 100:164]

        # Landmark 98 points
        t0 = time.time()
        lm_out = compiled["landmark"]([face_crop])[compiled["landmark"].output(0)]
        timings["landmark"] += time.time() - t0

        # Emotion classification
        t0 = time.time()
        emo_out = compiled["emotion"]([face_crop])[compiled["emotion"].output(0)]
        timings["emotion"] += time.time() - t0

        if i == N - 1:
            emo_label = EMO_LABELS[int(np.array(emo_out).reshape(-1).argmax())]
            print(f"  last frame: {len(boxes)} faces, emotion={emo_label}, lm shape={np.array(lm_out).shape}", flush=True)

    total = time.time() - t_total
    fps = N / total

    print(f"\n=== RESULT ===", flush=True)
    print(f"throughput: {fps:.1f} FPS  ({total*1000/N:.2f} ms/frame avg)", flush=True)
    print(f"face-detected frames: {n_face_detected}/{N}", flush=True)
    for stage, t in timings.items():
        if t > 0:
            print(f"  {stage:10s} {t*1000/N:6.2f} ms/frame", flush=True)

if __name__ == "__main__":
    main()
