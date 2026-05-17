#!/usr/bin/env python3
"""D435 RGB-D + SmolVLM2 三维场景理解。

核心思路：
  - SmolVLM2 看不到深度，只识别物体语义
  - D435 提供精确 mm 级深度
  - 把深度统计（前景/背景/9 宫格区域）注入到提示词，让 VLM 给出
    "什么物体 + 在多远" 的结构化场景理解

输出:
  scene_text: SmolVLM2 生成的描述
  depth_stats: 9 宫格 + 前景/背景的中位距离
  closest_object: 距离 + 像素位置 + 视觉块 id

依赖前置:
  llama-server 已用 SmolVLM2-2.2B 启动在 :8081
  $LLAMA_SERVER -m smolvlm2-2.2b-q4km.gguf --mmproj smolvlm2-2.2b-mmproj-f16.gguf \
      --port 8081 --host 127.0.0.1 -ngl 0 -t 8 -c 2048

用法:
  python scene3d.py --mode single        # 拍一帧分析
  python scene3d.py --mode loop --interval 3  # 每 3 秒一次
  python scene3d.py --mode dryrun        # 不调 VLM，只看深度统计
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

VLM_URL = "http://127.0.0.1:8081/completion"
SAVE_DIR = Path("/tmp/scene3d")
SAVE_DIR.mkdir(exist_ok=True)


def capture_rgbd(width=640, height=480, fps=30, warmup=10, reset=True):
    """捕获一帧 RGB + Depth（米单位）+ 内参。

    P0-4 修复: 复用 resources.py 单例，避免重复实现 hardware_reset。
    reset=True 时强制 force_reset (用于 single 模式确保新鲜状态)。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from resources import ensure_d435, grab_frames

    print("D435 启动 (含 hardware_reset)..." if reset else "D435 复用单例...")
    d435 = ensure_d435(force_reset=reset, warmup_frames=warmup)
    color, depth_m = grab_frames()
    return color, depth_m, d435["intrinsics"]


def depth_stats_grid(depth_m, rows=3, cols=3):
    """将深度图分 9 宫格，每格中位距离 + 有效像素占比。"""
    h, w = depth_m.shape
    cells = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            patch = depth_m[y0:y1, x0:x1]
            valid = patch[(patch > 0.1) & (patch < 8.0)]
            if len(valid) > patch.size * 0.05:
                cells.append({
                    "row": r, "col": c,
                    "median_m": float(np.median(valid)),
                    "min_m": float(np.min(valid)),
                    "valid_ratio": len(valid) / patch.size,
                })
            else:
                cells.append({"row": r, "col": c, "median_m": -1.0, "min_m": -1.0, "valid_ratio": 0.0})
    return cells


def closest_region(depth_m, patch=20):
    """找最近物体（去除噪声后），返回 (cx, cy, dist_m)。"""
    valid_mask = (depth_m > 0.2) & (depth_m < 5.0)
    if valid_mask.sum() < 100:
        return None
    # 用形态学开运算去掉零星噪声点
    blurred = cv2.medianBlur((depth_m * 1000).astype(np.uint16), 5)
    masked = np.where(valid_mask, blurred.astype(np.float32) / 1000, np.inf)
    cy, cx = np.unravel_index(np.argmin(masked), masked.shape)
    h, w = depth_m.shape
    y0, y1 = max(0, cy - patch), min(h, cy + patch)
    x0, x1 = max(0, cx - patch), min(w, cx + patch)
    region = depth_m[y0:y1, x0:x1]
    region_valid = region[(region > 0.2) & (region < 5.0)]
    if len(region_valid) == 0:
        return None
    return (int(cx), int(cy), float(np.median(region_valid)))


def annotate_image(color_bgr, cells, closest):
    """在 RGB 图上画距离信息。"""
    img = color_bgr.copy()
    h, w = img.shape[:2]
    # 9 宫格
    for cell in cells:
        if cell["median_m"] < 0:
            continue
        x = int((cell["col"] + 0.5) * w / 3)
        y = int((cell["row"] + 0.5) * h / 3)
        text = f"{cell['median_m']:.1f}m"
        cv2.putText(img, text, (x - 25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # 最近点
    if closest:
        cx, cy, d = closest
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), 3)
        cv2.putText(img, f"closest: {d:.2f}m",
                    (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    return img


def build_depth_prompt(cells, closest):
    """生成给 VLM 的深度上下文提示词（中文紧凑）。"""
    lines = []
    pos_names = [
        "左上", "正上", "右上",
        "左中", "中央", "右中",
        "左下", "正下", "右下",
    ]
    valid = [(pos_names[i], c["median_m"]) for i, c in enumerate(cells) if c["median_m"] > 0]
    valid_str = " ".join(f"{n}={d:.1f}m" for n, d in valid)

    lines.append("以下是深度传感器测得的距离信息（米）：")
    lines.append(f"九宫格: {valid_str}")
    if closest:
        cx, cy, d = closest
        lines.append(f"最近物体: 距相机 {d:.2f}m，位置约 ({cx},{cy})")
    return "\n".join(lines)


def call_vlm(image_bgr, depth_context, *, timeout=120):
    """调用 SmolVLM2 server 完成场景理解。

    用 /completion endpoint 直接传 image_data + 显式 <__image__> marker，
    /v1/chat/completions 在当前 llama.cpp 模板下不会注入 marker (返回 400)。
    """
    import requests

    _, jpeg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    user_text = (
        f"{depth_context}\n"
        "结合图像和距离信息，用中文一句话描述场景：物体、相对位置、最近物体。"
    )

    prompt = (
        f"<|im_start|>User: <__image__>{user_text}<end_of_utterance>\n"
        f"<|im_start|>Assistant:"
    )

    payload = {
        "prompt": prompt,
        "image_data": [{"data": b64, "id": 1}],
        "n_predict": 120,
        "temperature": 0.3,
        "stop": ["<end_of_utterance>", "<|im_end|>"],
    }
    r = requests.post(VLM_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    content = (body.get("content") or "").strip()
    if not content:
        # B08 修复: 空 content 应当作错误而非"成功"
        raise RuntimeError(
            f"VLM 返回空 content (HTTP {r.status_code}, "
            f"stop_type={body.get('stop_type','?')}, "
            f"tokens_predicted={body.get('tokens_predicted','?')})"
        )
    return content


def analyze_once(*, save=True, call_llm=True):
    """单帧端到端: 采集→深度统计→VLM→输出。"""
    print("[1/3] 采集 D435 RGB-D...")
    t0 = time.perf_counter()
    color, depth_m, _intr = capture_rgbd()
    print(f"    {(time.perf_counter()-t0)*1000:.0f} ms | shape={color.shape} depth_valid={int((depth_m>0.1).sum()/depth_m.size*100)}%")

    print("[2/3] 计算深度统计...")
    t0 = time.perf_counter()
    cells = depth_stats_grid(depth_m)
    closest = closest_region(depth_m)
    depth_ctx = build_depth_prompt(cells, closest)
    print(f"    {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"    {depth_ctx}")

    annotated = annotate_image(color, cells, closest)
    if save:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(SAVE_DIR / f"scene_{ts}_color.jpg"), color)
        cv2.imwrite(str(SAVE_DIR / f"scene_{ts}_anno.jpg"), annotated)
        print(f"    保存到 {SAVE_DIR}/scene_{ts}_*.jpg")

    if not call_llm:
        return {"depth_context": depth_ctx, "scene_text": None, "cells": cells}

    print("[3/3] SmolVLM2 推理...")
    t0 = time.perf_counter()
    try:
        scene_text = call_vlm(color, depth_ctx)
        print(f"    {(time.perf_counter()-t0):.1f}s")
        print(f"\n→ {scene_text}\n")
    except Exception as e:
        print(f"    VLM 失败: {e}")
        scene_text = None

    return {"depth_context": depth_ctx, "scene_text": scene_text, "cells": cells, "closest": closest}


def loop_mode(interval: float):
    """每 N 秒分析一次（持久 pipeline 复用 resources.py 单例），Ctrl+C 退出。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from resources import ensure_d435, grab_frames

    print("D435 启动 (P0-4: 复用 resources 单例)...")
    ensure_d435(warmup_frames=10)

    n = 0
    try:
        while True:
            print(f"\n========== 第 {n+1} 次分析 ==========")
            color, depth_m = grab_frames()

            cells = depth_stats_grid(depth_m)
            closest = closest_region(depth_m)
            depth_ctx = build_depth_prompt(cells, closest)
            print(depth_ctx)

            ts = time.strftime("%Y%m%d_%H%M%S")
            anno = annotate_image(color, cells, closest)
            cv2.imwrite(str(SAVE_DIR / f"loop_{ts}.jpg"), anno)

            try:
                t0 = time.perf_counter()
                scene = call_vlm(color, depth_ctx)
                if not scene:
                    print(f"VLM ({(time.perf_counter()-t0):.1f}s) → ⚠ 空响应")
                else:
                    print(f"VLM ({(time.perf_counter()-t0):.1f}s) → {scene}")
            except Exception as e:
                print(f"VLM 失败: {e}")
            n += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n停止，共完成 {n} 次")
    # D435 由 resources.atexit 自动 release


def main():
    p = argparse.ArgumentParser(description="D435 + SmolVLM2 三维场景理解")
    p.add_argument("--mode", choices=["single", "loop", "dryrun"], default="single")
    p.add_argument("--interval", type=float, default=3.0, help="loop 模式间隔秒")
    p.add_argument("--no-save", action="store_true", help="不保存图像")
    args = p.parse_args()

    if args.mode == "single":
        result = analyze_once(save=not args.no_save, call_llm=True)
        print("\n=== JSON 输出 ===")
        print(json.dumps({
            "scene_text": result["scene_text"],
            "closest": result.get("closest"),
            "depth_grid": [
                f"{c['median_m']:.2f}" if c['median_m'] > 0 else "N/A"
                for c in result["cells"]
            ],
        }, ensure_ascii=False, indent=2))
    elif args.mode == "loop":
        loop_mode(args.interval)
    elif args.mode == "dryrun":
        analyze_once(save=not args.no_save, call_llm=False)


if __name__ == "__main__":
    main()
