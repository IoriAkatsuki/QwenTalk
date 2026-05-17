#!/usr/bin/env python3
"""D435 深度摄像头 + NPU 3D 手势识别集成测试。

管线:
  D435 RGB ──→ NPU palm_detection (192×192)
            ──→ NPU handpose_estimation (224×224) → 21 个 2D 关键点
  D435 Depth ──→ CPU rs2_deproject → 21 个 3D 坐标 (x,y,z 米)
              ──→ 规则分类器: 5 种手势 + 距离过滤 + 3D 校验

测试模式:
  --mode check    检查 D435 连接 + 模型可用性
  --mode bench    NPU 推理速度基准（无需 D435）
  --mode live     D435 实时 3D 手势识别（GUI）
  --mode headless 无 GUI 版 live（SSH 远程用）

用法:
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate openvino
  python gesture3d_test.py --mode check
  python gesture3d_test.py --mode live --device NPU
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _paths import OMZ_ROOT as MODEL_ROOT  # B01: 不再硬编码 /home/intel
PALM_XML = MODEL_ROOT / "palm_detection_mediapipe_2023feb.onnx"
HAND_XML = MODEL_ROOT / "handpose_estimation_mediapipe_2023feb.onnx"

# 21 个 MediaPipe 手部关键点索引
WRIST = 0
THUMB_CMC = 1   # B05 修复: 拇指根部腕掌关节
THUMB_MCP = 2   # B05 修复: 拇指掌指关节 (用于 thumb 伸展判断)
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

# 距离过滤范围（米）：典型车舱手势 0.3-1.5m，过近过远不响应
DIST_MIN_M = 0.3
DIST_MAX_M = 1.5
# 真实手大小范围（米）：腕→中指 MCP 距离 5-15cm
HAND_SIZE_MIN = 0.05
HAND_SIZE_MAX = 0.15


def check_env():
    """检查 D435 和模型文件是否就绪。"""
    print("=== 环境检查 ===")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            for d in devs:
                name = d.get_info(rs.camera_info.name)
                sn = d.get_info(rs.camera_info.serial_number)
                print(f"  ✓ RealSense: {name} (SN: {sn})")
        else:
            print("  ✗ 未检测到 RealSense 设备")
    except ImportError:
        print("  ✗ pyrealsense2 未安装")

    for name, path in [("palm_detection", PALM_XML), ("handpose", HAND_XML)]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  ✓ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name}: {path} 不存在")

    try:
        import openvino as ov
        core = ov.Core()
        print(f"  OpenVINO 设备: {core.available_devices}")
    except Exception as e:
        print(f"  ✗ OpenVINO: {e}")


def load_npu_models(device: str = "NPU"):
    """编译 palm + handpose 模型到指定设备（NHWC 输入）。"""
    import openvino as ov
    core = ov.Core()

    palm = core.read_model(str(PALM_XML))
    palm.reshape({palm.input(0): [1, 192, 192, 3]})
    t0 = time.perf_counter()
    palm_c = core.compile_model(palm, device)
    print(f"palm_detection 编译到 {device}: {(time.perf_counter()-t0)*1000:.0f} ms")

    hand = core.read_model(str(HAND_XML))
    hand.reshape({hand.input(0): [1, 224, 224, 3]})
    t0 = time.perf_counter()
    hand_c = core.compile_model(hand, device)
    print(f"handpose 编译到 {device}: {(time.perf_counter()-t0)*1000:.0f} ms")

    return palm_c, hand_c


def preprocess_palm(frame_bgr, size=192):
    import cv2
    img = cv2.resize(frame_bgr, (size, size)).astype(np.float32) / 255.0
    return img[np.newaxis]


def preprocess_hand(crop_bgr, size=224):
    import cv2
    img = cv2.resize(crop_bgr, (size, size)).astype(np.float32) / 255.0
    return img[np.newaxis]


def estimate_handpose(hand_compiled, crop_bgr):
    """估计 21 个关键点，返回 (21,3) 归一化坐标。"""
    inp = preprocess_hand(crop_bgr)
    result = hand_compiled(inp)
    return result[hand_compiled.output(0)].reshape(-1, 3)


def palm_detection_score(palm_compiled, frame_bgr):
    """B02 修复: 跑 palm detector 拿最大检测分数, 用作 handpose 闸门。

    MediaPipe palm_detection 输出 (1, 2944, 18) regressors + (1, 2944, 1) scores
    返回最大 sigmoid(score), 0-1 范围。
    """
    inp = preprocess_palm(frame_bgr)
    result = palm_compiled(inp)
    # 找置信度输出 (单通道, 通常是 shape (1, N, 1))
    for output in palm_compiled.outputs:
        arr = result[output]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            # sigmoid 反归一化前的 logits
            max_logit = float(arr.max())
            return float(1.0 / (1.0 + np.exp(-max_logit)))
    return 0.0  # 找不到 score 输出, 不做 gating


def _classify_2d(pts):
    """2D 图像平面规则（手必须竖直朝上才准确）。"""
    thumb_out = pts[THUMB_TIP][0] > pts[INDEX_MCP][0]
    index_up = pts[INDEX_TIP][1] < pts[INDEX_MCP][1]
    middle_up = pts[MIDDLE_TIP][1] < pts[MIDDLE_MCP][1]
    ring_up = pts[RING_TIP][1] < pts[RING_MCP][1]
    pinky_up = pts[PINKY_TIP][1] < pts[PINKY_MCP][1]
    n = sum([index_up, middle_up, ring_up, pinky_up])
    return n, thumb_out, index_up, middle_up, ring_up, pinky_up


def _classify_3d(pts_3d):
    """3D 向量长度规则（旋转不变，姿态无关）。

    四指伸展: |tip - wrist| > |mcp - wrist| × 1.25
    拇指伸展 (B05 修复): 沿 CMC→MCP→IP→TIP 链，伸展时
              |tip - cmc| > |mcp - cmc| × 2.0 (弯曲时 tip 靠近 mcp)
    """
    wrist = pts_3d[WRIST]

    def finger_ext(tip_idx, mcp_idx):
        tip_d = float(np.linalg.norm(pts_3d[tip_idx] - wrist))
        mcp_d = float(np.linalg.norm(pts_3d[mcp_idx] - wrist))
        return tip_d > mcp_d * 1.25

    # 拇指: 链式判断（不再用 INDEX_MCP 作错误参考）
    cmc_to_tip = float(np.linalg.norm(pts_3d[THUMB_TIP] - pts_3d[THUMB_CMC]))
    cmc_to_mcp = float(np.linalg.norm(pts_3d[THUMB_MCP] - pts_3d[THUMB_CMC]))
    thumb_out = cmc_to_tip > cmc_to_mcp * 2.0 if cmc_to_mcp > 0.005 else False

    index_up = finger_ext(INDEX_TIP, INDEX_MCP)
    middle_up = finger_ext(MIDDLE_TIP, MIDDLE_MCP)
    ring_up = finger_ext(RING_TIP, RING_MCP)
    pinky_up = finger_ext(PINKY_TIP, PINKY_MCP)
    n = sum([index_up, middle_up, ring_up, pinky_up])
    return n, thumb_out, index_up, middle_up, ring_up, pinky_up


def estimate_wrist_distance(pts_3d):
    """B04 修复: wrist 深度无效时 fallback 到关键点中位深度。"""
    if pts_3d is None or len(pts_3d) < 21:
        return -1.0
    if pts_3d[WRIST][2] > 0.05:
        return float(pts_3d[WRIST][2])
    valid_z = pts_3d[:, 2]
    valid = valid_z[valid_z > 0.05]
    if len(valid) >= 5:
        return float(np.median(valid))
    return -1.0


def classify_gesture(landmarks_2d, pts_3d=None, wrist_distance=None):
    """规则手势分类（带距离过滤 + 3D 校验）。

    landmarks_2d: (21, 2/3) 归一化 xy（ONNX 输出）
    pts_3d: (21, 3) 反投影后的米制坐标，可选
    wrist_distance: 手腕到相机距离（米），可选；None 时尝试从 pts_3d 估计

    返回: (gesture_name, confidence)
        open_palm/fist/point/thumbs_up/peace/unknown/out_of_range/too_close/no_hand
    """
    # B04: wrist_distance 缺失时从 pts_3d fallback
    if wrist_distance is None or wrist_distance <= 0:
        wrist_distance = estimate_wrist_distance(pts_3d)

    # 距离闸门：超出范围不响应（必须有有效距离才能进入分类，避免无深度时绕过闸门）
    if wrist_distance <= 0:
        return "no_depth", 0.0
    if wrist_distance > DIST_MAX_M:
        return "out_of_range", 0.0
    if wrist_distance < DIST_MIN_M:
        return "too_close", 0.0

    if landmarks_2d.shape[0] < 21:
        return "no_hand", 0.0

    # 优先 3D 分类（旋转不变），失败 fallback 2D
    use_3d = False
    if pts_3d is not None and len(pts_3d) >= 21:
        valid_z_count = int(np.sum(pts_3d[:, 2] > 0.05))
        if valid_z_count >= 15:
            wrist_3d = pts_3d[WRIST]
            mcp_3d = pts_3d[MIDDLE_MCP]
            if wrist_3d[2] > 0 and mcp_3d[2] > 0:
                hand_size = float(np.linalg.norm(wrist_3d - mcp_3d))
                if HAND_SIZE_MIN <= hand_size <= HAND_SIZE_MAX:
                    use_3d = True

    if use_3d:
        n, thumb, idx, mid, ring, pky = _classify_3d(pts_3d)
        boost = 0.05
    else:
        n, thumb, idx, mid, ring, pky = _classify_2d(landmarks_2d[:, :2])
        boost = 0.0

    if n >= 4 and thumb:
        return "open_palm", 0.9 + boost
    if n == 0 and not thumb:
        return "fist", 0.9 + boost
    if idx and not mid and not ring and not pky:
        return "point", 0.85 + boost
    if thumb and n == 0:
        return "thumbs_up", 0.8 + boost
    if idx and mid and not ring and not pky:
        return "peace", 0.8 + boost
    return "unknown", 0.3


def bench_mode(device: str):
    """NPU 推理速度基准（无需 D435）。"""
    palm_c, hand_c = load_npu_models(device)

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    for _ in range(10):
        palm_c(preprocess_palm(dummy))
        hand_c(preprocess_hand(crop))

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        palm_c(preprocess_palm(dummy))
        times.append((time.perf_counter() - t0) * 1000)
    print(f"palm_detection: {np.mean(times):.2f} ms ({1000/np.mean(times):.0f} FPS)")

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        hand_c(preprocess_hand(crop))
        times.append((time.perf_counter() - t0) * 1000)
    print(f"handpose: {np.mean(times):.2f} ms ({1000/np.mean(times):.0f} FPS)")

    dummy_lm = np.random.rand(21, 3).astype(np.float32)
    dummy_3d = np.random.rand(21, 3).astype(np.float32) * 0.5 + 0.3
    t0 = time.perf_counter()
    for _ in range(10000):
        classify_gesture(dummy_lm, dummy_3d, 0.5)
    total = (time.perf_counter() - t0) * 1000
    print(f"classify (3D): {total/10000:.3f} ms")

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        palm_c(preprocess_palm(dummy))
        hand_c(preprocess_hand(crop))
        classify_gesture(dummy_lm, dummy_3d, 0.5)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"全流水: {np.mean(times):.2f} ms ({1000/np.mean(times):.0f} FPS)")


PALM_CONF_THRESHOLD = 0.5  # B02: palm detection 分数 < 此值跳过 handpose


def live_mode(device: str, headless: bool = False, max_frames: int = 0,
              palm_threshold: float = PALM_CONF_THRESHOLD):
    """D435 实时 3D 手势识别（P0-4 复用 resources.py + P1 palm gating）。

    max_frames=0: 持续运行（GUI 按 q 退出，headless 跑到 Ctrl+C）
    max_frames=N: 跑 N 帧后退出
    palm_threshold: palm detection 分数阈值, 低于则不跑 handpose (节省 NPU)
    """
    import pyrealsense2 as rs
    import cv2
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from resources import ensure_d435, grab_frames

    print("D435 启动 (含 hardware_reset)...")
    try:
        d435 = ensure_d435(warmup_frames=10)
    except RuntimeError as e:
        print(f"✗ D435 初始化失败: {e}")
        return

    print("D435 流就绪，加载 NPU 模型...")
    palm_c, hand_c = load_npu_models(device)

    print(f"D435: 640×480@30fps, depth_scale={d435['depth_scale']:.4f}")
    print(f"距离闸门: {DIST_MIN_M:.1f}m - {DIST_MAX_M:.1f}m | palm 阈值: {palm_threshold:.2f}")
    print("手势: open_palm fist point thumbs_up peace")
    if not headless:
        print("按 q 退出")

    frame_count = 0
    fps_t0 = time.perf_counter()
    stats = {"out_of_range": 0, "too_close": 0, "no_hand": 0,
             "no_depth": 0, "no_palm": 0, "valid": 0}

    try:
        while True:
            try:
                color_img, depth_m = grab_frames()
            except RuntimeError as e:
                print(f"⚠ 帧超时: {e}，重试...")
                continue

            h, w = color_img.shape[:2]

            # B02: palm detection score gating (低分跳过 handpose, 节省 NPU)
            palm_score = palm_detection_score(palm_c, color_img)
            if palm_score < palm_threshold:
                stats["no_palm"] += 1
                gesture, conf = "no_palm", 0.0
                landmarks = None
                pts_3d = None
                wrist_dist = -1.0
            else:
                margin = min(h, w) // 4
                crop = color_img[margin:h-margin, margin:w-margin]
                landmarks = estimate_handpose(hand_c, crop)

                pts_3d = np.zeros((21, 3), dtype=np.float32)
                for i in range(min(21, landmarks.shape[0])):
                    px = int(np.clip(landmarks[i][0] * (w - 2*margin) + margin, 0, w-1))
                    py = int(np.clip(landmarks[i][1] * (h - 2*margin) + margin, 0, h-1))
                    d_m = float(depth_m[py, px])
                    if 0.05 < d_m < 5.0:
                        pts_3d[i] = rs.rs2_deproject_pixel_to_point(
                            d435["intrinsics"], [float(px), float(py)], d_m,
                        )

                gesture, conf = classify_gesture(landmarks, pts_3d=pts_3d)
                wrist_dist = estimate_wrist_distance(pts_3d)
                stats[gesture] = stats.get(gesture, 0) + 1
                if gesture not in ("out_of_range", "too_close", "no_hand", "no_depth"):
                    stats["valid"] = stats.get("valid", 0) + 1

            frame_count += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                wrist_str = f"{wrist_dist:.2f}m" if wrist_dist > 0 else "N/A"
                if headless:
                    print(f"  {fps:.1f} FPS | palm={palm_score:.2f} | "
                          f"{gesture:12s} ({conf:.0%}) | wrist={wrist_str}")
                frame_count = 0
                fps_t0 = time.perf_counter()

            if not headless and landmarks is not None:
                color = (0, 255, 0) if conf > 0.5 else (128, 128, 128)
                margin = min(h, w) // 4
                for i in range(min(21, landmarks.shape[0])):
                    px = int(np.clip(landmarks[i][0] * (w - 2*margin) + margin, 0, w-1))
                    py = int(np.clip(landmarks[i][1] * (h - 2*margin) + margin, 0, h-1))
                    cv2.circle(color_img, (px, py), 3, color, -1)
                cv2.putText(color_img, f"{gesture} ({conf:.0%})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                wrist_text = f"Z={wrist_dist:.2f}m palm={palm_score:.2f}"
                color_z = (0, 255, 0) if DIST_MIN_M <= wrist_dist <= DIST_MAX_M else (0, 0, 255)
                cv2.putText(color_img, wrist_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_z, 2)
            if not headless:
                cv2.imshow("3D Gesture", color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            total_frames = sum(stats.values()) - stats["valid"]  # 不重复计数 valid
            if max_frames > 0 and total_frames >= max_frames:
                break
    finally:
        if not headless:
            cv2.destroyAllWindows()
        # D435 由 resources.py 的 atexit 释放 (P0-4 统一)
        print("\n=== 统计 ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="D435 + NPU 3D 手势识别")
    parser.add_argument("--mode", default="check",
                        choices=["check", "bench", "live", "headless"])
    parser.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    parser.add_argument("--frames", type=int, default=0,
                        help="帧数限制（0=持续运行）")
    args = parser.parse_args()

    if args.mode == "check":
        check_env()
    elif args.mode == "bench":
        bench_mode(args.device)
    elif args.mode == "live":
        live_mode(args.device, headless=False, max_frames=args.frames)
    elif args.mode == "headless":
        live_mode(args.device, headless=True, max_frames=args.frames)


if __name__ == "__main__":
    main()
